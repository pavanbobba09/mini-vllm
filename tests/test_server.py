"""M4 integration: the official openai client against a live uvicorn server.

Uses the tiny random model with a whitespace-integer stub tokenizer, so the
test is fast and download-free while exercising the full HTTP + engine path.
"""

import socket
import threading
import time
from typing import List

import pytest
import torch
import uvicorn
from openai import OpenAI

from engine.config import EngineConfig, ModelConfig
from engine.inference import reference_generate
from engine.model import MiniQwenForCausalLM
from server.api import EngineWorker, create_app


class StubTokenizer:
    """Maps '1 2 3' <-> [1, 2, 3]; enough to drive the tiny model over HTTP."""

    eos_token_id = None
    pad_token_id = None

    def encode(self, prompt: str, add_special_tokens: bool = True) -> List[int]:
        return [int(tok) for tok in prompt.split()]

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return " ".join(str(int(t)) for t in token_ids)


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        hidden_act="silu",
        attention_bias=True,
        mlp_bias=False,
        tie_word_embeddings=False,
    )


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def tiny_model() -> MiniQwenForCausalLM:
    torch.manual_seed(1234)
    return MiniQwenForCausalLM(tiny_config()).eval()


@pytest.fixture(scope="module")
def worker(tiny_model):
    return EngineWorker(
        tiny_model,
        StubTokenizer(),
        EngineConfig(block_size=4, num_blocks=64, max_num_seqs=4, watermark_blocks=2),
    )


@pytest.fixture(scope="module")
def server(worker):
    app = create_app(worker, model_name="tiny-test-model")
    port = free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    uvicorn_server = uvicorn.Server(config)
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not uvicorn_server.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/v1"
    uvicorn_server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def client(server):
    return OpenAI(base_url=server, api_key="not-needed")


def expected_text(tiny_model, prompt: str, max_tokens: int) -> str:
    prompt_ids = [int(t) for t in prompt.split()]
    out = reference_generate(
        tiny_model, torch.tensor([prompt_ids]), max_new_tokens=max_tokens
    )[0].tolist()
    return " ".join(str(t) for t in out[len(prompt_ids):])


def test_non_streaming_completion_matches_reference(tiny_model, client):
    prompt = "1 2 3 4 5"
    completion = client.completions.create(
        model="tiny-test-model", prompt=prompt, max_tokens=6, temperature=0.0
    )
    assert completion.object == "text_completion"
    assert completion.choices[0].finish_reason == "length"
    assert completion.usage.prompt_tokens == 5
    assert completion.usage.completion_tokens == 6
    assert completion.choices[0].text.strip() == expected_text(tiny_model, prompt, 6)


def test_streaming_completion_matches_reference(tiny_model, client):
    prompt = "7 8 9"
    parts = []
    finish_reasons = []
    stream = client.completions.create(
        model="tiny-test-model", prompt=prompt, max_tokens=5, temperature=0.0, stream=True
    )
    for chunk in stream:
        assert chunk.object == "text_completion"
        parts.append(chunk.choices[0].text)
        if chunk.choices[0].finish_reason is not None:
            finish_reasons.append(chunk.choices[0].finish_reason)

    assert finish_reasons == ["length"]
    assert "".join(parts).strip() == expected_text(tiny_model, prompt, 5)
    assert len([p for p in parts if p]) > 1, "streaming must deliver multiple chunks"


def test_multi_prompt_batch_shares_the_engine(tiny_model, client):
    prompts = ["1 2 3", "10 11 12 13", "20 21"]
    completion = client.completions.create(
        model="tiny-test-model", prompt=prompts, max_tokens=4, temperature=0.0
    )
    assert len(completion.choices) == 3
    by_index = sorted(completion.choices, key=lambda c: c.index)
    for prompt, choice in zip(prompts, by_index):
        assert choice.text.strip() == expected_text(tiny_model, prompt, 4)


def test_sampling_with_seed_is_reproducible(client):
    kwargs = dict(
        model="tiny-test-model", prompt="1 2 3", max_tokens=6, temperature=0.9, top_p=0.8, seed=7
    )
    first = client.completions.create(**kwargs)
    second = client.completions.create(**kwargs)
    assert first.choices[0].text == second.choices[0].text


def test_models_endpoint(client):
    models = client.models.list()
    assert models.data[0].id == "tiny-test-model"


@pytest.fixture(scope="module")
def secured_server(worker):
    app = create_app(worker, model_name="tiny-test-model", api_key="sk-test-key")
    port = free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    uvicorn_server = uvicorn.Server(config)
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not uvicorn_server.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    uvicorn_server.should_exit = True
    thread.join(timeout=5)


def test_api_key_required_when_configured(secured_server):
    import httpx
    from openai import AuthenticationError

    wrong = OpenAI(base_url=f"{secured_server}/v1", api_key="wrong-key", max_retries=0)
    with pytest.raises(AuthenticationError):
        wrong.completions.create(model="tiny-test-model", prompt="1 2 3", max_tokens=2)

    right = OpenAI(base_url=f"{secured_server}/v1", api_key="sk-test-key", max_retries=0)
    completion = right.completions.create(
        model="tiny-test-model", prompt="1 2 3", max_tokens=2, temperature=0.0
    )
    assert completion.choices[0].text

    # Health endpoint stays open for platform checks.
    assert httpx.get(f"{secured_server}/health").status_code == 200


def test_client_disconnect_aborts_request(client, worker):
    """Closing an SSE stream early must free the engine slot, not run to max_tokens."""

    stream = client.completions.create(
        model="tiny-test-model", prompt="1 2 3", max_tokens=500, temperature=0.0, stream=True
    )
    seen = 0
    for _ in stream:
        seen += 1
        if seen >= 2:
            break
    stream.close()

    pool = worker.scheduler.block_manager
    deadline = time.time() + 5
    while time.time() < deadline:
        with worker._lock:
            if not worker.scheduler.has_unfinished and pool.num_free_blocks == pool.num_blocks:
                return
        time.sleep(0.05)
    pytest.fail("aborted request did not release its blocks")
