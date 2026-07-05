"""Scheduler invariants: outputs match the reference, blocks never leak,
budget is never exceeded, long requests do not block short ones, and
preempted requests still finish correctly."""

import pytest
import torch

from engine.config import EngineConfig, ModelConfig
from engine.inference import reference_generate
from engine.model import MiniQwenForCausalLM
from engine.scheduler import Request, RequestState, Scheduler


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


@pytest.fixture(scope="module")
def tiny_model() -> MiniQwenForCausalLM:
    torch.manual_seed(1234)
    return MiniQwenForCausalLM(tiny_config()).eval()


def make_request(request_id: int, prompt_len: int, max_new_tokens: int, seed: int) -> Request:
    torch.manual_seed(seed)
    prompt = torch.randint(0, 64, (prompt_len,)).tolist()
    return Request(request_id=request_id, prompt_ids=prompt, max_new_tokens=max_new_tokens)


def run_to_completion(scheduler: Scheduler, max_steps: int = 500):
    steps = 0
    while scheduler.has_unfinished:
        scheduler.step()
        steps += 1
        assert steps < max_steps, "scheduler did not converge: possible starvation"
    return steps


def expected_output(model, request: Request):
    prompt = torch.tensor([request.prompt_ids])
    return reference_generate(
        model, prompt, max_new_tokens=request.max_new_tokens, eos_token_id=request.eos_token_id
    )[0].tolist()


def test_outputs_match_reference_under_batching(tiny_model):
    engine_config = EngineConfig(block_size=4, num_blocks=64, max_num_seqs=4, watermark_blocks=2)
    scheduler = Scheduler(tiny_model, engine_config)
    requests = [make_request(i, prompt_len=3 + i * 2, max_new_tokens=6 + i, seed=i) for i in range(5)]
    for r in requests:
        scheduler.add_request(r)

    run_to_completion(scheduler)

    for r in requests:
        assert r.state is RequestState.FINISHED
        assert r.prompt_ids + r.generated_ids == expected_output(tiny_model, r)
    assert scheduler.block_manager.num_free_blocks == engine_config.num_blocks


def test_admission_never_exceeds_budget(tiny_model):
    engine_config = EngineConfig(block_size=4, num_blocks=16, max_num_seqs=3, watermark_blocks=1)
    scheduler = Scheduler(tiny_model, engine_config)
    for i in range(6):
        scheduler.add_request(make_request(i, prompt_len=6, max_new_tokens=5, seed=10 + i))

    while scheduler.has_unfinished:
        scheduler.step()
        assert len(scheduler.running) <= engine_config.max_num_seqs
        used = engine_config.num_blocks - scheduler.block_manager.num_free_blocks
        assert used <= engine_config.num_blocks

    assert scheduler.block_manager.num_free_blocks == engine_config.num_blocks


def test_short_requests_finish_while_long_one_runs(tiny_model):
    """The continuous batching claim: a long request must not gate short ones."""

    engine_config = EngineConfig(block_size=4, num_blocks=64, max_num_seqs=4, watermark_blocks=2)
    scheduler = Scheduler(tiny_model, engine_config)
    long_req = make_request(0, prompt_len=4, max_new_tokens=60, seed=0)
    shorts = [make_request(i, prompt_len=4, max_new_tokens=3, seed=i) for i in range(1, 4)]
    scheduler.add_request(long_req)
    for r in shorts:
        scheduler.add_request(r)

    finish_step = {}
    step = 0
    while scheduler.has_unfinished:
        step += 1
        out = scheduler.step()
        for r in out.finished:
            finish_step[r.request_id] = step

    assert all(finish_step[r.request_id] < finish_step[0] for r in shorts)


def test_late_arrival_is_admitted_mid_flight(tiny_model):
    engine_config = EngineConfig(block_size=4, num_blocks=64, max_num_seqs=4, watermark_blocks=2)
    scheduler = Scheduler(tiny_model, engine_config)
    scheduler.add_request(make_request(0, prompt_len=4, max_new_tokens=30, seed=0))

    for _ in range(5):
        scheduler.step()

    late = make_request(1, prompt_len=5, max_new_tokens=4, seed=1)
    scheduler.add_request(late)
    out = scheduler.step()
    # Admitted and produced its first token on the very next step.
    assert late.request_id in out.new_tokens

    run_to_completion(scheduler)
    assert late.prompt_ids + late.generated_ids == expected_output(tiny_model, late)


def test_preemption_recovers_and_output_is_unchanged(tiny_model):
    """A pool too small for all admitted sequences forces preemption; greedy
    recompute on resume must land on the identical output."""

    engine_config = EngineConfig(block_size=4, num_blocks=10, max_num_seqs=3, watermark_blocks=1)
    scheduler = Scheduler(tiny_model, engine_config)
    requests = [make_request(i, prompt_len=6, max_new_tokens=14, seed=20 + i) for i in range(3)]
    for r in requests:
        scheduler.add_request(r)

    run_to_completion(scheduler)

    assert sum(r.num_preemptions for r in requests) > 0, "test must actually exercise preemption"
    for r in requests:
        assert r.prompt_ids + r.generated_ids == expected_output(tiny_model, r)
    assert scheduler.block_manager.num_free_blocks == engine_config.num_blocks


def test_eos_finishes_request_early(tiny_model):
    engine_config = EngineConfig(block_size=4, num_blocks=32, max_num_seqs=2, watermark_blocks=1)
    scheduler = Scheduler(tiny_model, engine_config)
    probe = make_request(0, prompt_len=5, max_new_tokens=8, seed=3)
    expected = expected_output(tiny_model, probe)
    # Use the reference run's second generated token as a fake EOS.
    eos = expected[len(probe.prompt_ids) + 1]
    request = Request(request_id=0, prompt_ids=probe.prompt_ids, max_new_tokens=8, eos_token_id=eos)
    scheduler.add_request(request)

    run_to_completion(scheduler)

    assert request.generated_ids[-1] == eos
    assert len(request.generated_ids) == 2
    assert scheduler.block_manager.num_free_blocks == engine_config.num_blocks


def test_identical_prompts_are_distinct_requests(tiny_model):
    """Two requests with identical fields must not be conflated by membership checks."""

    engine_config = EngineConfig(block_size=4, num_blocks=64, max_num_seqs=4, watermark_blocks=2)
    scheduler = Scheduler(tiny_model, engine_config)
    twin_a = make_request(0, prompt_len=5, max_new_tokens=6, seed=42)
    twin_b = make_request(1, prompt_len=5, max_new_tokens=6, seed=42)
    twin_b.request_id = 1
    assert twin_a.prompt_ids == twin_b.prompt_ids
    scheduler.add_request(twin_a)
    scheduler.add_request(twin_b)

    run_to_completion(scheduler)

    expected = expected_output(tiny_model, twin_a)
    assert twin_a.prompt_ids + twin_a.generated_ids == expected
    assert twin_b.prompt_ids + twin_b.generated_ids == expected
    assert scheduler.block_manager.num_free_blocks == engine_config.num_blocks


def test_oversized_prompt_rejected(tiny_model):
    engine_config = EngineConfig(block_size=4, num_blocks=4, max_num_seqs=2, watermark_blocks=1)
    scheduler = Scheduler(tiny_model, engine_config)
    with pytest.raises(ValueError):
        scheduler.add_request(make_request(0, prompt_len=40, max_new_tokens=4, seed=0))


def test_context_limit_enforced(tiny_model):
    """Sequences must never outgrow max_position_embeddings (256 for tiny_config)."""

    engine_config = EngineConfig(block_size=4, num_blocks=128, max_num_seqs=2, watermark_blocks=1)
    scheduler = Scheduler(tiny_model, engine_config)

    over = make_request(0, prompt_len=256, max_new_tokens=4, seed=0)
    with pytest.raises(ValueError):
        scheduler.add_request(over)

    near = make_request(1, prompt_len=250, max_new_tokens=50, seed=1)
    scheduler.add_request(near)
    assert near.max_new_tokens == 6  # clamped to the context limit
    run_to_completion(scheduler)
    assert len(near.prompt_ids) + len(near.generated_ids) <= 256
    assert scheduler.block_manager.num_free_blocks == engine_config.num_blocks


def test_abort_running_request_frees_blocks(tiny_model):
    engine_config = EngineConfig(block_size=4, num_blocks=64, max_num_seqs=4, watermark_blocks=2)
    scheduler = Scheduler(tiny_model, engine_config)
    victim = make_request(0, prompt_len=5, max_new_tokens=40, seed=0)
    survivor = make_request(1, prompt_len=7, max_new_tokens=6, seed=1)
    scheduler.add_request(victim)
    scheduler.add_request(survivor)
    scheduler.step()  # both admitted and running

    assert scheduler.abort_request(0) is True
    assert victim.state is RequestState.ABORTED

    run_to_completion(scheduler)
    assert survivor.prompt_ids + survivor.generated_ids == expected_output(tiny_model, survivor)
    assert scheduler.block_manager.num_free_blocks == engine_config.num_blocks
    assert scheduler.abort_request(0) is False  # already gone


def test_abort_waiting_request(tiny_model):
    engine_config = EngineConfig(block_size=4, num_blocks=8, max_num_seqs=1, watermark_blocks=1)
    scheduler = Scheduler(tiny_model, engine_config)
    first = make_request(0, prompt_len=5, max_new_tokens=6, seed=0)
    queued = make_request(1, prompt_len=5, max_new_tokens=6, seed=1)
    scheduler.add_request(first)
    scheduler.add_request(queued)
    scheduler.step()  # only first admitted (max_num_seqs=1)

    assert scheduler.abort_request(1) is True
    assert queued.state is RequestState.ABORTED

    run_to_completion(scheduler)
    assert first.state is RequestState.FINISHED
    assert not queued.generated_ids
    assert scheduler.block_manager.num_free_blocks == engine_config.num_blocks
