import copy
import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.config import resolve_device, resolve_dtype
from engine.inference import DEFAULT_MODEL_ID, MiniVLLMEngine
from engine.weights import resolve_model_path


pytestmark = pytest.mark.slow

PROMPTS = [
    "The capital of France is",
    "Write a Python function that adds two numbers.",
    "In one sentence, explain gravity:",
    "Translate to Spanish: good morning",
    "List three colors:",
]


def require_parity_enabled():
    if os.getenv("MINI_VLLM_RUN_PARITY") != "1":
        pytest.skip("set MINI_VLLM_RUN_PARITY=1 to run full Qwen parity tests")


def test_greedy_decode_matches_transformers_generate_token_for_token():
    require_parity_enabled()
    model_id = os.getenv("MINI_VLLM_MODEL", DEFAULT_MODEL_ID)
    model_dir = resolve_model_path(model_id)
    device = resolve_device(os.getenv("MINI_VLLM_TEST_DEVICE"))
    dtype = resolve_dtype(os.getenv("MINI_VLLM_TEST_DTYPE", "auto"), device)

    engine = MiniVLLMEngine.from_pretrained(str(model_dir), device=device, dtype=dtype)
    hf_tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        trust_remote_code=False,
    ).to(device)
    hf_model.eval()
    greedy_config = copy.deepcopy(hf_model.generation_config)
    greedy_config.do_sample = False
    greedy_config.repetition_penalty = 1.0
    greedy_config.temperature = None
    greedy_config.top_p = None
    greedy_config.top_k = None

    for prompt in PROMPTS:
        inputs = hf_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        expected = hf_model.generate(
            **inputs,
            generation_config=greedy_config,
            max_new_tokens=8,
            pad_token_id=hf_tokenizer.eos_token_id,
        )

        actual_ids = engine.generate_greedy(prompt, max_new_tokens=8, add_special_tokens=True)

        assert actual_ids == expected[0].detach().cpu().tolist()


def test_prompt_logits_match_transformers_forward():
    require_parity_enabled()
    model_id = os.getenv("MINI_VLLM_MODEL", DEFAULT_MODEL_ID)
    model_dir = resolve_model_path(model_id)
    device = resolve_device(os.getenv("MINI_VLLM_TEST_DEVICE"))
    dtype = resolve_dtype(os.getenv("MINI_VLLM_TEST_DTYPE", "auto"), device)

    prompt = PROMPTS[0]
    engine = MiniVLLMEngine.from_pretrained(str(model_dir), device=device, dtype=dtype)
    hf_tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        trust_remote_code=False,
    ).to(device)
    hf_model.eval()

    inputs = hf_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        expected = hf_model(**inputs).logits
        actual = engine.model(inputs["input_ids"])

    assert torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3) is None
