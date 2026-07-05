"""Block allocator and per-sequence block tables for the paged KV cache.

OS analogy: physical blocks are page frames, a block table is a page table, and
logical position i lives at slot table[i // block_size] * block_size + i % block_size.
"""

from __future__ import annotations

import math
from typing import Dict, List


class OutOfBlocksError(RuntimeError):
    """Raised when an allocation would exceed the physical block pool."""


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        # LIFO free list: recently freed blocks are reused first.
        self._free: List[int] = list(range(num_blocks - 1, -1, -1))
        self._block_tables: Dict[int, List[int]] = {}
        self._num_tokens: Dict[int, int] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    def blocks_needed(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size)

    def can_allocate(self, num_tokens: int) -> bool:
        return self.blocks_needed(num_tokens) <= self.num_free_blocks

    def block_table(self, seq_id: int) -> List[int]:
        return list(self._block_tables[seq_id])

    def num_tokens(self, seq_id: int) -> int:
        return self._num_tokens[seq_id]

    def has_sequence(self, seq_id: int) -> bool:
        return seq_id in self._block_tables

    def allocate_sequence(self, seq_id: int, num_tokens: int) -> List[int]:
        """Reserve blocks for a whole prompt at once (prefill)."""

        if seq_id in self._block_tables:
            raise ValueError(f"sequence {seq_id} already has blocks allocated")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        needed = self.blocks_needed(num_tokens)
        if needed > self.num_free_blocks:
            raise OutOfBlocksError(
                f"need {needed} blocks for {num_tokens} tokens, only {self.num_free_blocks} free"
            )
        table = [self._free.pop() for _ in range(needed)]
        self._block_tables[seq_id] = table
        self._num_tokens[seq_id] = num_tokens
        return list(table)

    def append_token(self, seq_id: int) -> int:
        """Count one more token, growing the table at block boundaries; returns its slot."""

        if seq_id not in self._block_tables:
            raise KeyError(f"sequence {seq_id} has no allocation")
        position = self._num_tokens[seq_id]
        if position % self.block_size == 0 and position // self.block_size == len(self._block_tables[seq_id]):
            if not self._free:
                raise OutOfBlocksError(f"no free block to grow sequence {seq_id}")
            self._block_tables[seq_id].append(self._free.pop())
        self._num_tokens[seq_id] = position + 1
        return self.slot_for_position(seq_id, position)

    def slot_for_position(self, seq_id: int, position: int) -> int:
        table = self._block_tables[seq_id]
        block_idx, offset = divmod(position, self.block_size)
        return table[block_idx] * self.block_size + offset

    def slots_for_range(self, seq_id: int, start: int, end: int) -> List[int]:
        return [self.slot_for_position(seq_id, pos) for pos in range(start, end)]

    def free_sequence(self, seq_id: int) -> None:
        """Return all blocks to the pool; double-free raises."""

        if seq_id not in self._block_tables:
            raise KeyError(f"sequence {seq_id} has no allocation (double free?)")
        table = self._block_tables.pop(seq_id)
        del self._num_tokens[seq_id]
        for block in table:
            if block in self._free:
                raise RuntimeError(f"block {block} is already free")
            self._free.append(block)
