# Benchmark results

device: mps, dtype: float16, model: Qwen/Qwen2.5-0.5B-Instruct

| concurrency | engine | requests | wall (s) | output tokens | tokens/s | p50 TTFT (s) | p50 latency (s) | p99 latency (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | mini-vllm | 2 | 3.3 | 94 | 28.5 | 1.2 | 2.7 | 4.4 |
| 1 | transformers | 2 | 4.3 | 94 | 21.7 | 3.8 | 3.8 | 5.3 |
| 8 | mini-vllm | 16 | 8.8 | 653 | 74.0 | 1.1 | 6.2 | 9.0 |
| 8 | transformers | 16 | 13.5 | 653 | 48.2 | 10.7 | 10.7 | 13.5 |
| 32 | mini-vllm | 64 | 10.2 | 2440 | 239.0 | 1.6 | 5.4 | 10.2 |
| 32 | transformers | 64 | 8.2 | 2440 | 297.2 | 6.7 | 6.7 | 8.2 |
