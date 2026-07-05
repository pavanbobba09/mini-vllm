"""Benchmark mini-vllm continuous batching vs vanilla transformers static batching.

Workload: 2x concurrency requests with uneven prompt lengths and uneven
max_tokens, all submitted at t=0. The HF baseline serves them in fixed batches
of size `concurrency` via model.generate (the naive serving pattern); mini-vllm
schedules them through one continuously batched engine.

Usage: .venv/bin/python -m bench.benchmark --concurrency 1 8 32
"""

from __future__ import annotations

import argparse
import gc
import random
import statistics
import time
from dataclasses import dataclass
from typing import List

import torch

from engine.config import EngineConfig, resolve_device
from engine.inference import DEFAULT_MODEL_ID, MiniVLLMEngine
from engine.scheduler import Request, Scheduler

PROMPT_POOL = [
    "The capital of France is",
    "Write a Python function that adds two numbers.",
    "In one sentence, explain gravity:",
    "Translate to Spanish: good morning",
    "List three colors:",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What is the difference between a list and a tuple in Python? Answer briefly.",
    "Continue the story: Once upon a time, a small robot woke up alone in a warehouse and",
    "Explain what an operating system page table does, in plain words, for a beginner:",
    "Give me a haiku about the ocean.",
]


@dataclass
class Workload:
    prompt: str
    prompt_ids: List[int]
    max_tokens: int


def _pct(values: List[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    return float(statistics.quantiles(values, n=100)[int(pct) - 1])


@dataclass
class RunResult:
    engine: str
    concurrency: int
    num_requests: int
    wall_s: float
    output_tokens: int
    latencies_s: List[float]
    ttfts_s: List[float]

    @property
    def tokens_per_s(self) -> float:
        return self.output_tokens / self.wall_s

    def latency_pct(self, pct: float) -> float:
        return _pct(self.latencies_s, pct)

    def ttft_pct(self, pct: float) -> float:
        return _pct(self.ttfts_s, pct)


def build_workload(tokenizer, concurrency: int, seed: int = 0) -> List[Workload]:
    rng = random.Random(seed)
    items = []
    for i in range(2 * concurrency):
        prompt = PROMPT_POOL[i % len(PROMPT_POOL)]
        items.append(
            Workload(
                prompt=prompt,
                prompt_ids=tokenizer.encode(prompt),
                max_tokens=rng.randint(8, 64),
            )
        )
    return items


def warmup_mini_vllm(engine: MiniVLLMEngine) -> None:
    scheduler = Scheduler(engine.model, EngineConfig(num_blocks=64, max_num_seqs=2))
    scheduler.add_request(
        Request(request_id=0, prompt_ids=engine.tokenizer.encode("warm up"), max_new_tokens=4)
    )
    while scheduler.has_unfinished:
        scheduler.step()


def run_mini_vllm(engine: MiniVLLMEngine, workload: List[Workload], concurrency: int) -> RunResult:
    # 384 blocks = 6144 cached tokens, plenty for 32 seqs while fitting 8 GB RAM.
    engine_config = EngineConfig(
        block_size=16, num_blocks=384, max_num_seqs=concurrency, watermark_blocks=4
    )
    scheduler = Scheduler(engine.model, engine_config)
    eos = engine.tokenizer.eos_token_id
    requests = [
        Request(
            request_id=i,
            prompt_ids=item.prompt_ids,
            max_new_tokens=item.max_tokens,
            eos_token_id=eos,
        )
        for i, item in enumerate(workload)
    ]

    start = time.perf_counter()
    for request in requests:
        scheduler.add_request(request)
    finish_at = {}
    first_token_at = {}
    while scheduler.has_unfinished:
        out = scheduler.step()
        now = time.perf_counter()
        for request_id in out.new_tokens:
            first_token_at.setdefault(request_id, now)
        for request in out.finished:
            finish_at[request.request_id] = now
    wall = time.perf_counter() - start

    output_tokens = sum(len(r.generated_ids) for r in requests)
    latencies = [finish_at[r.request_id] - start for r in requests]
    ttfts = [first_token_at[r.request_id] - start for r in requests]
    return RunResult("mini-vllm", concurrency, len(requests), wall, output_tokens, latencies, ttfts)


def load_transformers(model_dir: str, device: torch.device, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    tokenizer.padding_side = "left"  # decoder-only generation pads on the left
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype).to(device)
    model.eval()
    # Untimed warmup so one-time kernel compilation does not pollute the first run.
    warm = tokenizer(["warm up"], return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(**warm, do_sample=False, max_new_tokens=4, pad_token_id=tokenizer.pad_token_id)
    return model, tokenizer


def run_transformers(model, tokenizer, device: torch.device, workload: List[Workload], concurrency: int) -> RunResult:
    start = time.perf_counter()
    latencies: List[float] = []
    output_tokens = 0
    # Static batching: fixed batches run to completion; the whole batch waits
    # for its slowest member, and the next batch waits for the whole batch.
    for batch_start in range(0, len(workload), concurrency):
        batch = workload[batch_start : batch_start + concurrency]
        inputs = tokenizer([w.prompt for w in batch], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max(w.max_tokens for w in batch),
                pad_token_id=tokenizer.pad_token_id,
            )
        batch_done = time.perf_counter()
        prompt_len = inputs["input_ids"].shape[1]
        for row, item in enumerate(batch):
            generated = out[row, prompt_len:].tolist()
            if tokenizer.eos_token_id in generated:
                generated = generated[: generated.index(tokenizer.eos_token_id) + 1]
            # A request only ever asked for its own max_tokens.
            generated = generated[: item.max_tokens]
            output_tokens += len(generated)
            # In static batching every request waits for its whole batch; plain
            # generate also cannot stream, so TTFT equals full batch latency.
            latencies.append(batch_done - start)
    wall = time.perf_counter() - start
    return RunResult(
        "transformers", concurrency, len(workload), wall, output_tokens, latencies, list(latencies)
    )


def to_markdown(results: List[RunResult]) -> str:
    lines = [
        "| concurrency | engine | requests | wall (s) | output tokens | tokens/s | p50 TTFT (s) | p50 latency (s) | p99 latency (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.concurrency} | {r.engine} | {r.num_requests} | {r.wall_s:.1f} | "
            f"{r.output_tokens} | {r.tokens_per_s:.1f} | {r.ttft_pct(50):.1f} | "
            f"{r.latency_pct(50):.1f} | {r.latency_pct(99):.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--output", default="bench/results.md")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    print(f"device={device.type} dtype={args.dtype} model={args.model}")

    from engine.weights import resolve_model_path

    model_dir = str(resolve_model_path(args.model))

    # One model resident at a time: with 8 GB unified memory, keeping both
    # engines loaded pushes the machine into swap and poisons the numbers.
    engine = MiniVLLMEngine.from_pretrained(args.model, device=device, dtype=dtype)
    warmup_mini_vllm(engine)
    workloads = {c: build_workload(engine.tokenizer, c) for c in args.concurrency}
    mini_results = []
    for concurrency in args.concurrency:
        print(f"[concurrency {concurrency}] mini-vllm ...", flush=True)
        mini_results.append(run_mini_vllm(engine, workloads[concurrency], concurrency))
        print(f"  {mini_results[-1].tokens_per_s:.1f} tok/s in {mini_results[-1].wall_s:.1f}s", flush=True)
    del engine
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    hf_model, hf_tokenizer = load_transformers(model_dir, device, dtype)
    hf_results = []
    for concurrency in args.concurrency:
        print(f"[concurrency {concurrency}] transformers ...", flush=True)
        hf_results.append(run_transformers(hf_model, hf_tokenizer, device, workloads[concurrency], concurrency))
        print(f"  {hf_results[-1].tokens_per_s:.1f} tok/s in {hf_results[-1].wall_s:.1f}s", flush=True)
    del hf_model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    results = [r for pair in zip(mini_results, hf_results) for r in pair]
    table = to_markdown(results)
    header = f"# Benchmark results\n\ndevice: {device.type}, dtype: {args.dtype}, model: {args.model}\n\n"
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(header + table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
