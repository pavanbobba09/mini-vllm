# Observability

mini-vllm exposes engine metrics and structured request lifecycle logs without
coupling the inference core to a monitoring vendor. `engine.telemetry` defines
the events; `server.observability` implements them with Prometheus and JSON
logging. The default `NullTelemetry` keeps library and benchmark use unchanged.

## Data flow

```mermaid
flowchart LR
    API[FastAPI] --> W[EngineWorker]
    W --> S[Scheduler]
    S --> M[Model]
    S --> KV[Block manager and KV cache]
    API -. bounded HTTP status .-> T[Telemetry sink]
    S -. lifecycle and step snapshots .-> T
    T --> P[Prometheus registry]
    T --> L[JSON lifecycle logs]
    P --> E[/metrics]
```

The scheduler records monotonic lifecycle timestamps:

```text
submitted -> admitted -> first token -> finished / aborted
           | queue     | TTFT                    |
           |----------- total request duration -|
```

Queue time ends when admission begins. TTFT includes prefill and sampling.
Host-side prefill and decode measurements deliberately avoid extra GPU
synchronization; accurate CUDA-kernel timing can later be sampled with CUDA
events.

## Metrics

| metric | meaning |
|---|---|
| `mini_vllm_requests_total` | requests accepted by the scheduler |
| `mini_vllm_requests_finished_total` | completions split by bounded finish reason |
| `mini_vllm_requests_aborted_total` | requests cancelled before completion |
| `mini_vllm_request_duration_seconds` | submission-to-finish duration |
| `mini_vllm_queue_duration_seconds` | submission-to-first-admission duration |
| `mini_vllm_ttft_seconds` | time to first generated token |
| `mini_vllm_inter_token_seconds` | time between tokens for a sequence |
| `mini_vllm_prefill_duration_seconds` | host-observed paged prefill duration |
| `mini_vllm_decode_step_duration_seconds` | host-observed batched decode duration |
| `mini_vllm_scheduler_step_duration_seconds` | full admit/decode/retire step duration |
| `mini_vllm_decode_batch_size` | sequences in each actual decode call |
| `mini_vllm_prompt_tokens_total` | original prompt tokens accepted |
| `mini_vllm_prefill_tokens_total` | all prefill work, including preemption replay |
| `mini_vllm_generated_tokens_total` | generated output tokens |
| `mini_vllm_requests_waiting` | current admission queue depth |
| `mini_vllm_requests_running` | current active sequence count |
| `mini_vllm_kv_blocks_free` | unallocated physical KV blocks |
| `mini_vllm_kv_blocks_used` | allocated physical KV blocks |
| `mini_vllm_preemptions_total` | sequences evicted and queued for recompute |
| `mini_vllm_http_requests_total` | HTTP requests by normalized endpoint and status |

Run the server, then scrape `http://localhost:8000/metrics`. Health probes are
available at `/health/live` and `/health/ready`; the existing `/health` endpoint
remains supported.

## Logs and privacy

The CLI configures one JSON record for submission, first admission, first token,
preemption, completion, and abort. Records contain request IDs, timing, token
counts, finish reason, and preemption count. They never contain prompt text,
generated text, token arrays, authorization headers, or API keys.

Prometheus labels are intentionally bounded. Model, finish reason, normalized
endpoint, and HTTP status are allowed. Request IDs, raw paths, prompts, users,
and exception messages must never become labels.

## Engineering loop

Every performance change follows the same loop:

1. Capture a baseline for output parity, TTFT, inter-token latency, throughput,
   scheduler duration, and KV utilization.
2. Make one scoped engine or kernel change.
3. Run the tiny-model correctness and lifecycle suite.
4. Compare the same workload against the baseline.
5. Keep the change only when correctness holds and the intended metric improves
   without an unacceptable regression elsewhere.

This makes later chunked prefill, prefix caching, and fused PagedAttention work
measurable instead of relying on aggregate tokens/second alone.

## Initial alert candidates

- p99 TTFT exceeds its benchmark-derived objective.
- Waiting requests grow continuously while decode throughput is flat.
- Free KV blocks remain below the scheduler watermark.
- Preemption or abort rate rises sharply.
- Requests are running but generated-token counters stop increasing.
- `/health/ready` returns 503 because the worker thread is unavailable.
