"""Paged KV cache correctness: paged decode must equal the M1 reference exactly.

The tiny random model makes these tests fast and download-free while still
exercising GQA, RoPE positions, block-boundary crossings, and ragged batching.
"""

import os

import pytest
import torch

from engine.block_manager import BlockManager
from engine.config import ModelConfig
from engine.inference import (
    paged_decode_step,
    paged_generate,
    paged_prefill,
    reference_generate,
)
from engine.kv_cache import PagedKVCache
from engine.model import MiniQwenForCausalLM


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: 2 query heads per KV head
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        hidden_act="silu",
        attention_bias=True,
        mlp_bias=False,
        tie_word_embeddings=False,
    )


@pytest.fixture(scope="module")
def tiny_model() -> MiniQwenForCausalLM:
    torch.manual_seed(1234)
    return MiniQwenForCausalLM(tiny_config()).eval()


def fresh_cache(model: MiniQwenForCausalLM, num_blocks: int = 32, block_size: int = 4):
    # block_size 4 forces many block-boundary crossings in short tests.
    cache = PagedKVCache(
        model.config,
        num_blocks=num_blocks,
        block_size=block_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    bm = BlockManager(num_blocks=num_blocks, block_size=block_size)
    return cache, bm


@pytest.mark.parametrize("prompt_len,new_tokens", [(3, 5), (4, 9), (7, 20), (13, 6)])
def test_paged_generate_matches_reference(tiny_model, prompt_len, new_tokens):
    torch.manual_seed(prompt_len)
    prompt = torch.randint(0, 64, (1, prompt_len))

    expected = reference_generate(tiny_model, prompt, max_new_tokens=new_tokens)[0].tolist()

    cache, bm = fresh_cache(tiny_model)
    actual = paged_generate(
        tiny_model, cache, bm, prompt[0].tolist(), max_new_tokens=new_tokens
    )

    assert actual == expected
    assert bm.num_free_blocks == bm.num_blocks  # everything freed afterwards


def test_batched_decode_matches_individual_runs(tiny_model):
    """Two ragged sequences decoded in one batch must match their solo runs.

    This is the M3 precursor: it catches cross-sequence masking and block-table
    padding bugs that single-sequence tests cannot see.
    """

    torch.manual_seed(7)
    prompt_a = torch.randint(0, 64, (1, 3))
    prompt_b = torch.randint(0, 64, (1, 11))  # different length on purpose
    new_tokens = 8

    solo = {}
    for name, prompt in (("a", prompt_a), ("b", prompt_b)):
        cache, bm = fresh_cache(tiny_model)
        solo[name] = paged_generate(
            tiny_model, cache, bm, prompt[0].tolist(), max_new_tokens=new_tokens
        )

    cache, bm = fresh_cache(tiny_model)
    last = {}
    outputs = {"a": prompt_a[0].tolist(), "b": prompt_b[0].tolist()}
    for seq_id, name, prompt in ((0, "a", prompt_a), (1, "b", prompt_b)):
        logits = paged_prefill(tiny_model, cache, bm, seq_id, prompt[0].tolist())
        last[name] = int(torch.argmax(logits).item())
        outputs[name].append(last[name])

    for _ in range(new_tokens - 1):
        logits = paged_decode_step(tiny_model, cache, bm, [0, 1], [last["a"], last["b"]])
        for row, name in ((0, "a"), (1, "b")):
            last[name] = int(torch.argmax(logits[row]).item())
            outputs[name].append(last[name])

    assert outputs["a"] == solo["a"]
    assert outputs["b"] == solo["b"]


def test_cache_reuse_after_sequence_end(tiny_model):
    """Blocks freed by one sequence are reused by the next without stale reads."""

    torch.manual_seed(3)
    prompt1 = torch.randint(0, 64, (1, 9))
    prompt2 = torch.randint(0, 64, (1, 6))

    cache, bm = fresh_cache(tiny_model, num_blocks=4, block_size=4)  # tight pool forces reuse
    out1 = paged_generate(tiny_model, cache, bm, prompt1[0].tolist(), max_new_tokens=5)
    out2 = paged_generate(tiny_model, cache, bm, prompt2[0].tolist(), max_new_tokens=5)

    assert out1 == reference_generate(tiny_model, prompt1, max_new_tokens=5)[0].tolist()
    assert out2 == reference_generate(tiny_model, prompt2, max_new_tokens=5)[0].tolist()


@pytest.mark.slow
def test_paged_generate_matches_reference_on_real_model():
    """Full Qwen2.5-0.5B paged decode vs reference, gated like the other slow tests."""

    if os.getenv("MINI_VLLM_RUN_PARITY") != "1":
        pytest.skip("set MINI_VLLM_RUN_PARITY=1 to run full Qwen parity tests")

    from engine.config import resolve_device, resolve_dtype
    from engine.inference import DEFAULT_MODEL_ID, MiniVLLMEngine
    from engine.weights import resolve_model_path

    model_dir = resolve_model_path(os.getenv("MINI_VLLM_MODEL", DEFAULT_MODEL_ID))
    device = resolve_device(os.getenv("MINI_VLLM_TEST_DEVICE"))
    dtype = resolve_dtype(os.getenv("MINI_VLLM_TEST_DTYPE", "auto"), device)
    engine = MiniVLLMEngine.from_pretrained(str(model_dir), device=device, dtype=dtype)

    prompt = "The capital of France is"
    prompt_ids = engine.tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_ids], device=device)

    expected = reference_generate(
        engine.model, input_ids, max_new_tokens=16, eos_token_id=engine.tokenizer.eos_token_id
    )[0].tolist()

    cache = PagedKVCache(
        engine.model.config, num_blocks=64, block_size=16, device=device, dtype=dtype
    )
    bm = BlockManager(num_blocks=64, block_size=16)
    actual = paged_generate(
        engine.model,
        cache,
        bm,
        prompt_ids,
        max_new_tokens=16,
        eos_token_id=engine.tokenizer.eos_token_id,
    )

    assert actual == expected
