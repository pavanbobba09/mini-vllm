"""M6 observability: metrics and logs must follow the engine lifecycle safely."""

import io
import json
import logging

import torch
from fastapi.testclient import TestClient

from engine.config import EngineConfig, ModelConfig
from engine.model import MiniQwenForCausalLM
from engine.scheduler import Request, Scheduler
from server.api import EngineWorker, create_app
from server.observability import JsonLogFormatter, PrometheusTelemetry


class StubTokenizer:
    eos_token_id = None
    pad_token_id = None

    def encode(self, prompt: str, add_special_tokens: bool = True):
        return [int(token) for token in prompt.split()]

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return " ".join(str(int(token)) for token in token_ids)


def tiny_model() -> MiniQwenForCausalLM:
    torch.manual_seed(1234)
    config = ModelConfig(
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
    return MiniQwenForCausalLM(config).eval()


def telemetry_with_logs():
    output = io.StringIO()
    logger = logging.Logger("mini-vllm-observability-test", level=logging.INFO)
    handler = logging.StreamHandler(output)
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
    return PrometheusTelemetry("tiny-test-model", logger=logger), output


def metric_value(telemetry: PrometheusTelemetry, name: str, **labels):
    return telemetry.registry.get_sample_value(name, labels)


def test_scheduler_lifecycle_metrics_and_cache_cleanup():
    telemetry, _ = telemetry_with_logs()
    config = EngineConfig(block_size=4, num_blocks=32, max_num_seqs=2, watermark_blocks=1)
    scheduler = Scheduler(tiny_model(), config, telemetry=telemetry)
    request = Request(request_id=7, prompt_ids=[1, 2, 3], max_new_tokens=4)
    scheduler.add_request(request)

    while scheduler.has_unfinished:
        scheduler.step()

    model = {"model": "tiny-test-model"}
    assert metric_value(telemetry, "mini_vllm_requests_total", **model) == 1
    assert metric_value(telemetry, "mini_vllm_prompt_tokens_total", **model) == 3
    assert metric_value(telemetry, "mini_vllm_generated_tokens_total", **model) == 4
    assert metric_value(
        telemetry,
        "mini_vllm_requests_finished_total",
        model="tiny-test-model",
        finish_reason="length",
    ) == 1
    assert metric_value(telemetry, "mini_vllm_ttft_seconds_count", **model) == 1
    assert metric_value(telemetry, "mini_vllm_prefill_duration_seconds_count", **model) == 1
    assert metric_value(telemetry, "mini_vllm_decode_step_duration_seconds_count", **model) == 3
    assert metric_value(telemetry, "mini_vllm_kv_blocks_free", **model) == 32
    assert metric_value(telemetry, "mini_vllm_kv_blocks_used", **model) == 0
    assert scheduler.block_manager.num_free_blocks == config.num_blocks


def test_abort_is_counted_and_releases_blocks():
    telemetry, _ = telemetry_with_logs()
    config = EngineConfig(block_size=4, num_blocks=32, max_num_seqs=2, watermark_blocks=1)
    scheduler = Scheduler(tiny_model(), config, telemetry=telemetry)
    request = Request(request_id=9, prompt_ids=[1, 2, 3], max_new_tokens=20)
    scheduler.add_request(request)
    scheduler.step()

    assert scheduler.abort_request(request.request_id)
    assert metric_value(
        telemetry, "mini_vllm_requests_aborted_total", model="tiny-test-model"
    ) == 1
    assert scheduler.block_manager.num_free_blocks == config.num_blocks


def test_lifecycle_logs_are_structured_and_do_not_contain_token_payloads():
    telemetry, output = telemetry_with_logs()
    scheduler = Scheduler(
        tiny_model(),
        EngineConfig(block_size=4, num_blocks=32, max_num_seqs=2, watermark_blocks=1),
        telemetry=telemetry,
    )
    scheduler.add_request(Request(request_id=11, prompt_ids=[61, 62, 63], max_new_tokens=2))
    while scheduler.has_unfinished:
        scheduler.step()

    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [record["event"] for record in records] == [
        "request_submitted",
        "request_admitted",
        "request_first_token",
        "request_finished",
    ]
    serialized = output.getvalue()
    assert "prompt_ids" not in serialized
    assert "generated_ids" not in serialized
    assert "61 62 63" not in serialized


def test_each_observability_instance_has_an_isolated_registry():
    first = PrometheusTelemetry("first")
    second = PrometheusTelemetry("second")

    first.http_request("health", 200, 0.001)
    second.http_request("health", 503, 0.002)

    assert metric_value(
        first, "mini_vllm_http_requests_total", endpoint="health", status="200"
    ) == 1
    assert metric_value(
        first, "mini_vllm_http_requests_total", endpoint="health", status="503"
    ) is None
    assert metric_value(
        second, "mini_vllm_http_requests_total", endpoint="health", status="503"
    ) == 1


def test_metrics_and_health_endpoints():
    telemetry, _ = telemetry_with_logs()
    worker = EngineWorker(
        tiny_model(),
        StubTokenizer(),
        EngineConfig(block_size=4, num_blocks=32, max_num_seqs=2, watermark_blocks=1),
        telemetry=telemetry,
    )
    app = create_app(worker, model_name="tiny-test-model", observability=telemetry)

    with TestClient(app, follow_redirects=True) as client:
        metrics = client.get("/metrics")
        readiness = client.get("/health/ready")

    assert metrics.status_code == 200
    assert "mini_vllm_requests_total" in metrics.text
    assert 'mini_vllm_kv_blocks_free{model="tiny-test-model"} 32.0' in metrics.text
    assert readiness.status_code == 200
    assert readiness.json()["worker_alive"] is True
    assert readiness.json()["free_kv_blocks"] == 32


def test_disabled_observability_does_not_read_hot_path_clocks(monkeypatch):
    scheduler = Scheduler(
        tiny_model(), EngineConfig(block_size=4, num_blocks=32, max_num_seqs=2)
    )

    def unexpected_clock_read():
        raise AssertionError("disabled observability must not time scheduler operations")

    monkeypatch.setattr("engine.scheduler.time.perf_counter", unexpected_clock_read)
    request = Request(request_id=12, prompt_ids=[1, 2, 3], max_new_tokens=3)
    scheduler.add_request(request)
    while scheduler.has_unfinished:
        scheduler.step()

    assert request.submitted_at is None
    assert request.first_token_at is None
