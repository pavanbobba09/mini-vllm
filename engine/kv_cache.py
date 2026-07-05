"""Paged KV cache tensors and the per-step metadata attention needs to use them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from engine.config import ModelConfig


@dataclass
class PagedBatchMeta:
    """Per-step cache metadata, built once by the runner/scheduler and shared by all layers."""

    slot_mapping: torch.Tensor  # long [total_input_tokens], physical slot per token
    block_tables: torch.Tensor  # long [batch, max_blocks_per_seq], 0-padded
    context_lens: torch.Tensor  # long [batch], cached tokens after this step's writes
    is_prefill: bool


class PagedKVCache:
    """Flat per-layer K/V slot arrays: [num_blocks * block_size, num_kv_heads, head_dim].

    Storing only the KV heads (2 vs 14 query heads on Qwen2.5-0.5B) is the 7x GQA saving.
    """

    def __init__(
        self,
        config: ModelConfig,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_slots = num_blocks * block_size
        shape = (self.num_slots, config.num_key_value_heads, config.attention_head_dim)
        self.k_cache: List[torch.Tensor] = [
            torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)
        ]
        self.v_cache: List[torch.Tensor] = [
            torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)
        ]

    def write(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """Scatter post-RoPE K/V [total_tokens, kv_heads, head_dim] into slots.

        This is where vLLM's reshape_and_cache CUDA kernel slots in. Indexed
        assignment (index_put_) instead of index_copy_ because MPS lacks the latter.
        """

        self.k_cache[layer_idx][slot_mapping] = key
        self.v_cache[layer_idx][slot_mapping] = value

    def gather(self, layer_idx: int, block_tables: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return padded dense K/V [batch, max_blocks * block_size, kv_heads, head_dim].

        Pure-PyTorch stand-in for the fused paged_attention kernel, which would
        read blocks in place instead of gathering.
        """

        batch = block_tables.shape[0]
        offsets = torch.arange(self.block_size, device=block_tables.device)
        slots = (block_tables.unsqueeze(-1) * self.block_size + offsets).reshape(batch, -1)
        return self.k_cache[layer_idx][slots], self.v_cache[layer_idx][slots]


def build_block_tables_tensor(tables: List[List[int]], device: torch.device) -> torch.Tensor:
    """Pad ragged block tables with 0; padded reads are masked out via context_lens."""

    max_len = max(len(t) for t in tables)
    padded = [t + [0] * (max_len - len(t)) for t in tables]
    return torch.tensor(padded, dtype=torch.long, device=device)


def make_meta(
    slot_mapping: List[int],
    block_tables: List[List[int]],
    context_lens: List[int],
    is_prefill: bool,
    device: torch.device,
) -> PagedBatchMeta:
    return PagedBatchMeta(
        slot_mapping=torch.tensor(slot_mapping, dtype=torch.long, device=device),
        block_tables=build_block_tables_tensor(block_tables, device),
        context_lens=torch.tensor(context_lens, dtype=torch.long, device=device),
        is_prefill=is_prefill,
    )
