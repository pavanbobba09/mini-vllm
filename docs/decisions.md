# Design Decisions

## Milestone 1 - Correct Forward Pass

### Goal

Implement Qwen2.5-0.5B-Instruct inference with our own PyTorch modules while using HuggingFace only for tokenization, checkpoint download, and test oracle behavior.

### Options Considered

1. Parse `config.json` ourselves vs. use `transformers.AutoConfig` vs. hard-code Qwen2.5-0.5B dimensions.
2. Require a local checkpoint path vs. auto-download from HuggingFace vs. support both.
3. Verify raw logits vs. verify generated text vs. verify generated token IDs.

### Choice

We parse `config.json` into `ModelConfig`, support both local paths and HuggingFace repo IDs, and test token-level greedy parity against `transformers.generate`. A logits tolerance test is also included because token parity alone can hide small forward-pass mistakes.

The module names intentionally mirror HuggingFace state-dict keys, so strict loading becomes an architecture check. If we get attention dimensions, biases, layer names, or MLP structure wrong, weight loading fails before generation.

### What Changes At 10x Model Size

Milestone 1 loads all safetensors into a CPU state dict before moving the model. That is simple and correct for a 0.5B model, but it doubles peak memory during load. For a 10x model, we would stream shards into an initialized model, consider meta-device construction, and avoid holding a full duplicate state dict.

### What Changes At 100x Traffic

Milestone 1 recomputes the full prefix on every decode step. That is intentionally inefficient because it gives us a clean correctness baseline. At high traffic, the KV cache, PagedAttention, and continuous batching from later milestones become mandatory.

## Milestone 2 - PagedAttention KV Cache

### Goal

Replace full-prefix recompute with a KV cache that does not reserve max-length
contiguous tensors per sequence.

### Why paging beats one contiguous KV tensor per sequence

A contiguous cache must reserve `max_seq_len * kv_size` per sequence up front
because a tensor cannot grow in place. Most sequences stop early, so most of the
reservation is dead memory (internal fragmentation); vLLM measured 60-80% waste.
Slack-then-reallocate schemes instead scatter variable-size holes across the pool
(external fragmentation). Paging is the same fix operating systems use for RAM:
fixed-size blocks (16 tokens), a per-sequence block table as the page table, and
logical position i living at slot `table[i // 16] * 16 + i % 16`. Waste is bounded
by less than one block (15 slots max) per sequence, and any free block can serve
any sequence, so external fragmentation cannot exist.

### Why GQA makes the cache 7x smaller

The cache stores K/V per KV head, and Qwen2.5-0.5B has 2 KV heads serving 14
query heads. An MHA layout would cache 14 heads; GQA caches 2 and re-expands
them (repeat_kv) after reading. Same attention math, 7x less cache memory,
which directly multiplies how many sequences fit in the block pool.

### Options considered

1. Cache layout: per-block tensors `[num_blocks, block_size, heads, dim]` vs one
   flat slot array `[num_blocks * block_size, heads, dim]` per layer. Chose flat:
   writes become a single `index_copy_` over a slot mapping and reads a single
   fancy-index gather, which are exactly the two spots vLLM replaces with the
   `reshape_and_cache` and `paged_attention` CUDA kernels. The comments in
   `kv_cache.py` and `layers.py` mark those seams.
2. Prefill batching: batch-of-1 prefill vs padded multi-prompt prefill. Chose
   batch-of-1: no padding logic in the causal path, and the M3 scheduler admits
   prompts one at a time anyway. Padded prefill is a throughput optimization to
   revisit with chunked prefill.
3. Where RoPE meets the cache: store post-RoPE keys (chosen, matches vLLM) vs
   rotate at read time. Rotating at read would re-rotate the whole context every
   step, costing compute for no correctness gain at fixed positions.

### Verification

Token-for-token equality against `reference_generate` (the M1 full-recompute
oracle) on a tiny random GQA model with block_size 4 to force boundary
crossings, a ragged two-sequence batched decode against solo runs, block reuse
after free with a tight 4-block pool, and a gated real-Qwen paged-vs-reference
test. Allocator invariants (no leak, no double free, reuse, disjoint tables)
are unit-tested separately.

### What changes at 10x model size

The block pool becomes the dominant GPU allocation, so `num_blocks` must be
derived from measured free VRAM after weights load instead of a constant. The
gather-based attention also materializes padded K/V per step, which at 10x
hidden sizes wastes bandwidth; that is when the fused paged-attention kernel
stops being optional.

### What changes at 100x traffic

Prefix sharing across requests (same system prompt cached once) needs per-block
reference counts and copy-on-write, which the BlockManager deliberately does not
have yet. Fused QKV projections are the other deferred optimization.

## Milestone 3 - Continuous Batching Scheduler

### Continuous vs static batching

Static batching picks N requests, runs them to completion, and only then admits
more. Every request in the batch waits for the slowest one, and freed capacity
sits idle until the whole batch drains. Continuous batching re-decides membership
every decode step: finished sequences retire mid-batch and free their blocks
immediately, and waiting requests join the very next step. Per-step admission is
what keeps the GPU working on every live sequence instead of on padding and
stragglers, which is where the big utilization wins come from.

### Eviction policy

Chosen: preempt the youngest (most recently admitted) sequence and recompute its
full context on resume. Youngest-first means the oldest request always makes
progress, which gives the no-starvation guarantee; preempted requests rejoin the
FRONT of the waiting queue so they resume before never-admitted ones. Recompute
is correct under greedy decoding because replaying prompt + generated tokens
lands on the same continuation, and seeded sampling stays reproducible because
the per-token generator is derived from seed + tokens generated. The alternative,
swapping KV blocks to CPU RAM and back, avoids the recompute FLOPs at the cost of
PCIe transfers and a second allocator; vLLM supports both and defaults to
recompute for exactly this simplicity reason.

### Step shape

Each step: (1) admit from the FIFO queue while the watermark allows, running one
prefill per admission (batch of 1) which also emits that request's first token;
(2) one batched decode step for previously admitted sequences; (3) retire
finished sequences and free their blocks in the same step. Capacity for decode
is checked before the batch runs and fixed by preempting, never by letting the
allocator throw mid-batch, because a partial batch append would corrupt state.

### Verification

Outputs under batching, late arrival, tight-pool preemption, and duplicate
prompts all match reference_generate exactly; blocks return to a full pool after
every scenario; short requests provably finish while a long one keeps running.

### What changes at 10x model size

Prefill cost dominates admission, so batch-of-1 prefill becomes the bottleneck:
chunked prefill (splitting long prompts across steps, interleaved with decode)
keeps decode latency flat. Preemption-by-recompute also gets more expensive,
tilting the tradeoff toward swap-to-CPU.

### What changes at 100x traffic

FIFO admission ignores prompt length, so one giant prompt can delay many small
ones; a real system adds priority or shortest-job-first queues plus fairness
budgets, and enforces per-tenant quotas. The scheduler loop itself (pure Python,
one lock) would need to move off the request thread entirely.

## Milestone 4 - OpenAI-Compatible Server

### Options considered

1. Engine placement: run scheduler steps inside request handlers vs a dedicated
   engine thread. Chose the thread: torch forward passes are synchronous and
   would starve the event loop, and a single owner thread means the scheduler
   needs one lock and no async rewrite. Handlers talk to it through per-request
   asyncio queues fed via call_soon_threadsafe.
2. Streaming detokenization: decode per token vs decode-all-and-slice. Chose
   decode-all-and-slice (emit text[len(previous):]) because BPE merges and
   multi-byte characters make single-token decode wrong at chunk boundaries.
3. Test transport: ASGI in-process transport vs real uvicorn on a port. Chose
   real uvicorn with the official openai client pointed at base_url, because
   "the openai client works unmodified" is the milestone's acceptance bar.

### Verification

Integration tests drive the official openai client for non-streaming, SSE
streaming (multiple chunks asserted), multi-prompt batches served concurrently
by one engine, seeded sampling reproducibility, and /v1/models. Greedy outputs
are asserted token-exact against reference_generate through the whole HTTP path.

### What changes at 10x model size / 100x traffic

One engine thread per process is the scaling unit; bigger models add tensor
parallelism inside the step, more traffic adds replicas behind a load balancer.
Missing for production: request timeouts and client-disconnect cancellation,
backpressure on the waiting queue, chat completions endpoint, stop strings,
logprobs, and metrics.

## Milestone 5 - Benchmarks

### Methodology

Workload per concurrency level: 2x concurrency requests with uneven prompt
lengths and uneven max_tokens (8-64, seeded), all available at t=0. The
transformers baseline is the naive serving pattern: fixed batches of size
`concurrency` through `model.generate`, next batch starts only when the current
one finishes, and a request's latency is the time until its whole batch returns.
mini-vllm serves the same workload through the continuous batching scheduler.
Throughput counts each request's useful tokens (up to its own max_tokens or EOS).

### Fairness decisions

1. Both engines warm up untimed before measurement, because the first MPS
   forward pass includes one-time kernel compilation that otherwise lands on
   whichever engine runs first (it distorted first-run numbers by 5-10x).
2. Only one model is resident at a time, with gc plus torch.mps.empty_cache()
   between phases. This machine has 8 GB unified memory; two resident fp32
   models pushed it into swap and produced a 450x collapse in one HF run.
3. fp16 for both engines on MPS, for the same memory reason. Correctness tests
   stay fp32 on CPU; the benchmark only measures speed.

### Honest caveats

This is pure-PyTorch paged attention with a per-step Python scheduler, so decode
does a block-table gather and rebuilds step metadata in Python every token. The
per-step overhead is why single-stream throughput is unimpressive; the win the
benchmark demonstrates is scheduling (no batch-drain stalls, per-request
max_tokens, immediate backfill), not kernel speed. vLLM gets both because its
paged attention is a fused CUDA kernel and its scheduler overhead is amortized
with CUDA graphs.

### Results (fp16, MPS, 8 GB machine)

Concurrency 1: tie (20.4 vs 19.4 tok/s), the single-stream sanity check.
Concurrency 8: mini-vllm wins throughput by 28% (54.5 vs 42.7 tok/s) and cuts
p50 latency from 12.3s to 8.9s; retire-and-backfill beats batch-drain.
Concurrency 32: transformers wins raw throughput (290 vs 203 tok/s) because
mini-vllm is step-rate bound (identical 12.0s wall at concurrency 8 and 32);
fused kernels beat per-step Python once batches are wide. The scheduler win and
the kernel gap are separable effects, and this benchmark shows both.

## Optimization and Hardening Pass (post-M5)

The first benchmark showed the engine step-rate bound: identical 12.0s wall at
concurrency 8 and 32. Profiling the step path by inspection found two per-step
wastes and two serving holes.

### 1. RoPE table shared across layers

Before: each of the 24 attention layers owned a RotaryEmbedding and recomputed
einsum + cos + sin for the same positions every forward, so identical angles
were computed 24x per step. After: the model owns one RotaryEmbedding that
precomputes fp32 cos/sin tables for all positions at init; each step does one
table lookup shared by every layer. Tables stay fp32 because long-position
angles lose precision in fp16.

### 2. Fused QKV projection

Three separate q/k/v matmuls per layer became one fused projection (the
optimization deferred in Milestone 1). The loader now carries an explicit HF ->
ours remap that concatenates q/k/v checkpoint tensors in the exact order the
forward splits them. Effect of 1 + 2 together: the CPU fp32 parity suite dropped
from 93s to 52s, about 1.8x faster per decode step.

### 3. Context-length enforcement (correctness hole)

Nothing stopped a sequence from outgrowing max_position_embeddings; past it the
RoPE lookup would fail or silently misbehave. add_request now rejects prompts at
or over the limit and clamps max_new_tokens to the remaining budget, so the
clamp happens once at admission instead of being checked every step.

### 4. Request abort (serving hole)

A disconnected client used to keep its sequence decoding to max_tokens, holding
blocks and burning compute for output nobody reads. scheduler.abort_request
drops a request from waiting or running and frees its blocks; the server calls
it when an SSE stream is dropped or a handler is cancelled. Verified by a test
that closes a 500-token stream after 2 chunks and asserts the pool returns to
full within a bounded wait.

### Benchmark methodology addition

TTFT (time to first token) p50 was added to the table. For mini-vllm it is the
step at which a request's first token appears; for the transformers baseline it
equals full batch latency because plain generate cannot stream. That column is
the clearest view of what per-step admission buys.

### Measured effect

Same workload and machine as the M5 run. mini-vllm went from 20.4/54.5/203.1 to
28.5/74.0/239.0 tok/s at concurrency 1/8/32 (+40%/+36%/+18%), overtaking the
transformers baseline at concurrency 1 and 8. p50 TTFT holds at 1.1-1.6s at
every level while the static baseline sits at 3.8-10.7s. The concurrency-32
throughput gap to fused kernels remains (239 vs 297 tok/s), as expected without
a fused paged-attention kernel.
