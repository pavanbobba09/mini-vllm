# mini-vllm

An educational LLM inference engine built from scratch in Python and PyTorch.

Current target: `Qwen/Qwen2.5-0.5B-Instruct`.

## Milestone 1

Milestone 1 implements a correct non-cached Qwen2 forward pass:

- safetensors weight loading
- custom RMSNorm
- custom RoPE
- custom grouped-query causal attention
- custom SwiGLU MLP
- greedy decoding without using `transformers` generation utilities in the engine

The public helper for the milestone is:

```python
from engine import logits_for_prompt

logits = logits_for_prompt("The capital of France is")
```

For repeated calls, load the model once:

```python
from engine import MiniVLLMEngine

engine = MiniVLLMEngine.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
logits = engine.logits("The capital of France is")
ids = engine.generate_greedy("The capital of France is", max_new_tokens=8)
```

## Tests

Fast smoke tests:

```bash
python3 -m pytest
```

Full parity tests against HuggingFace `generate`:

```bash
MINI_VLLM_RUN_PARITY=1 python3 -m pytest tests/test_greedy_parity.py
```

Use `MINI_VLLM_TEST_DEVICE=cpu` to force CPU correctness tests.
