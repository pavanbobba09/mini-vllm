"""Qwen2 building blocks implemented directly in PyTorch."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from engine.config import ModelConfig
from engine.kv_cache import PagedBatchMeta, PagedKVCache
from engine.rope import apply_rotary_pos_emb


class QwenRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # RMSNorm normalizes by root-mean-square only; it does not subtract a mean.
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class QwenMLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation {config.hidden_act!r}; Qwen2.5 uses 'silu'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU gates the up projection, which improves capacity at similar FLOPs.
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped KV heads to one KV head per query head."""

    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class QwenAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.attention_head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        # One fused projection instead of three: fewer kernel launches per step.
        # The loader concatenates HF q/k/v tensors in this exact order.
        self.q_size = config.query_hidden_size
        self.kv_size = config.kv_hidden_size
        self.qkv_proj = nn.Linear(
            config.hidden_size, self.q_size + 2 * self.kv_size, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(config.query_hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[PagedKVCache] = None,
        meta: Optional[PagedBatchMeta] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = rope
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            assert meta is not None, "paged attention needs PagedBatchMeta"
            # Cache stores post-RoPE K/V, KV heads only; repeat_kv happens after reads.
            key_flat = key_states.transpose(1, 2).reshape(batch_size * seq_len, self.num_key_value_heads, self.head_dim)
            value_flat = value_states.transpose(1, 2).reshape(
                batch_size * seq_len, self.num_key_value_heads, self.head_dim
            )
            kv_cache.write(self.layer_idx, key_flat, value_flat, meta.slot_mapping)
            if not meta.is_prefill:
                return self._decode_with_cache(query_states, kv_cache, meta)
            # Prefill always starts at position 0 in v1, so the in-step K/V is the
            # whole context and the regular causal path below applies unchanged.

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = self._apply_masks(attn_weights, attention_mask)

        # Softmax in fp32 avoids avoidable precision loss, then returns to model dtype.
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)

    def _decode_with_cache(
        self,
        query_states: torch.Tensor,
        kv_cache: PagedKVCache,
        meta: PagedBatchMeta,
    ) -> torch.Tensor:
        """Decode attention over the paged cache; vLLM's fused paged_attention
        kernel does this in one pass without materializing padded K/V."""

        batch_size = query_states.shape[0]
        key_states, value_states = kv_cache.gather(self.layer_idx, meta.block_tables)
        key_states = key_states.transpose(1, 2)  # [batch, kv_heads, padded_len, head_dim]
        value_states = value_states.transpose(1, 2)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Padding blocks and unwritten slots hold garbage; mask beyond true length.
        padded_len = key_states.shape[2]
        positions = torch.arange(padded_len, device=query_states.device)
        invalid = positions[None, :] >= meta.context_lens[:, None]  # [batch, padded_len]
        attn_weights = attn_weights.masked_fill(
            invalid[:, None, None, :], torch.finfo(attn_weights.dtype).min
        )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, 1, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)

    def _apply_masks(
        self,
        attn_weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        seq_len = attn_weights.shape[-1]
        min_value = torch.finfo(attn_weights.dtype).min

        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=attn_weights.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask, min_value)

        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].to(torch.bool)
            attn_weights = attn_weights.masked_fill(~key_mask, min_value)
        return attn_weights


class QwenDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = QwenAttention(config=config, layer_idx=layer_idx)
        self.mlp = QwenMLP(config)
        self.input_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[PagedKVCache] = None,
        meta: Optional[PagedBatchMeta] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            rope=rope,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            meta=meta,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states
