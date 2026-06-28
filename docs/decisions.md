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
