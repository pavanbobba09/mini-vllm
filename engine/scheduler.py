"""Continuous batching scheduler: admit, decode, and retire sequences every step,
so no request ever waits for an unrelated long sequence to finish."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional

import torch

from engine.block_manager import BlockManager
from engine.config import EngineConfig
from engine.inference import paged_decode_step, paged_prefill
from engine.kv_cache import PagedKVCache
from engine.model import MiniQwenForCausalLM


class RequestState(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


# eq=False: requests compare by identity, so membership checks cannot confuse
# two requests that happen to carry identical prompts.
@dataclass(eq=False)
class Request:
    request_id: int
    prompt_ids: List[int]
    max_new_tokens: int
    eos_token_id: Optional[int] = None
    temperature: float = 0.0  # 0 means greedy
    top_p: float = 1.0
    seed: Optional[int] = None
    generated_ids: List[int] = field(default_factory=list)
    state: RequestState = RequestState.WAITING
    num_preemptions: int = 0

    @property
    def num_context_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.generated_ids)

    @property
    def is_done(self) -> bool:
        if len(self.generated_ids) >= self.max_new_tokens:
            return True
        return bool(
            self.eos_token_id is not None
            and self.generated_ids
            and self.generated_ids[-1] == self.eos_token_id
        )


@dataclass
class StepOutput:
    new_tokens: Dict[int, int]  # request_id -> token emitted this step
    finished: List[Request]


class Scheduler:
    def __init__(self, model: MiniQwenForCausalLM, engine_config: EngineConfig) -> None:
        self.model = model
        self.config = engine_config
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self.kv_cache = PagedKVCache(
            model.config,
            num_blocks=engine_config.num_blocks,
            block_size=engine_config.block_size,
            device=device,
            dtype=dtype,
        )
        self.block_manager = BlockManager(
            num_blocks=engine_config.num_blocks, block_size=engine_config.block_size
        )
        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []  # admission order: index 0 is oldest

    def add_request(self, request: Request) -> None:
        prompt_len = len(request.prompt_ids)
        max_context = self.model.config.max_position_embeddings
        if prompt_len >= max_context:
            raise ValueError(
                f"prompt is {prompt_len} tokens, model context limit is {max_context}"
            )
        # Clamp so the sequence can never outgrow the RoPE table.
        request.max_new_tokens = min(request.max_new_tokens, max_context - prompt_len)
        needed = self.block_manager.blocks_needed(prompt_len)
        if needed > self.config.num_blocks - self.config.watermark_blocks:
            raise ValueError(
                f"prompt needs {needed} blocks, pool has {self.config.num_blocks} "
                f"minus {self.config.watermark_blocks} watermark"
            )
        request.state = RequestState.WAITING
        self.waiting.append(request)

    def _can_admit(self, request: Request) -> bool:
        if len(self.running) >= self.config.max_num_seqs:
            return False
        needed = self.block_manager.blocks_needed(request.num_context_tokens)
        # Watermark keeps headroom so running sequences can still grow a block.
        return needed + self.config.watermark_blocks <= self.block_manager.num_free_blocks

    def _admit(self, request: Request) -> int:
        # Resume replays prompt + generated so far; greedy recompute lands on the
        # same continuation, which is why preemption is recompute-safe.
        context = request.prompt_ids + request.generated_ids
        logits = paged_prefill(
            self.model, self.kv_cache, self.block_manager, request.request_id, context
        )
        request.state = RequestState.RUNNING
        self.running.append(request)
        return self._sample(logits, request)

    def _preempt_youngest(self) -> None:
        # Youngest-first keeps the oldest request progressing: no starvation.
        victim = self.running.pop()
        victim.state = RequestState.WAITING
        victim.num_preemptions += 1
        self.block_manager.free_sequence(victim.request_id)
        self.waiting.appendleft(victim)

    def _needs_new_block(self, seq_id: int) -> bool:
        return self.block_manager.num_tokens(seq_id) % self.config.block_size == 0

    def step(self) -> StepOutput:
        """One engine iteration: admit, batched decode, retire."""

        new_tokens: Dict[int, int] = {}
        finished: List[Request] = []

        # Admit while budget allows; each prefill emits that request's first token.
        admitted_now: List[Request] = []
        while self.waiting and self._can_admit(self.waiting[0]):
            request = self.waiting.popleft()
            token = self._admit(request)
            request.generated_ids.append(token)
            new_tokens[request.request_id] = token
            admitted_now.append(request)

        # Decode sequences admitted in earlier steps.
        decode_batch = [r for r in self.running if r not in admitted_now and not r.is_done]
        if decode_batch:
            self._ensure_decode_capacity(decode_batch)
            decode_batch = [r for r in decode_batch if r.state is RequestState.RUNNING]
        if decode_batch:
            logits = paged_decode_step(
                self.model,
                self.kv_cache,
                self.block_manager,
                [r.request_id for r in decode_batch],
                [r.generated_ids[-1] for r in decode_batch],
            )
            if all(r.temperature <= 0.0 for r in decode_batch):
                # One batched argmax and a single device sync instead of one per row.
                tokens = torch.argmax(logits, dim=-1).tolist()
            else:
                tokens = [self._sample(logits[row], r) for row, r in enumerate(decode_batch)]
            for token, request in zip(tokens, decode_batch):
                request.generated_ids.append(token)
                new_tokens[request.request_id] = token

        # Retire finished sequences now so their blocks free mid-batch.
        still_running: List[Request] = []
        for request in self.running:
            if request.is_done:
                request.state = RequestState.FINISHED
                self.block_manager.free_sequence(request.request_id)
                finished.append(request)
            else:
                still_running.append(request)
        self.running = still_running

        return StepOutput(new_tokens=new_tokens, finished=finished)

    def _ensure_decode_capacity(self, decode_batch: List[Request]) -> None:
        # Capacity is checked BEFORE decoding; a mid-batch OutOfBlocksError would
        # leave half the batch appended and corrupt allocator state.
        while True:
            active = [r for r in decode_batch if r.state is RequestState.RUNNING]
            needed = sum(1 for r in active if self._needs_new_block(r.request_id))
            if needed <= self.block_manager.num_free_blocks or not self.running:
                return
            self._preempt_youngest()

    def abort_request(self, request_id: int) -> bool:
        """Drop a request wherever it lives, freeing its blocks if running.

        The serving-layer half of continuous batching: a disconnected client must
        release its slot immediately, not at max_tokens.
        """

        for i, request in enumerate(self.waiting):
            if request.request_id == request_id:
                del self.waiting[i]
                request.state = RequestState.ABORTED
                return True
        for i, request in enumerate(self.running):
            if request.request_id == request_id:
                self.running.pop(i)
                self.block_manager.free_sequence(request_id)
                request.state = RequestState.ABORTED
                return True
        return False

    @property
    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def _sample(self, logits: torch.Tensor, request: Request) -> int:
        if request.temperature <= 0.0:
            return int(torch.argmax(logits, dim=-1).item())

        probs = torch.softmax(logits.to(torch.float32) / request.temperature, dim=-1)
        generator = self._generator_for(request)
        if request.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # Shift keeps the first token crossing the threshold inside the nucleus.
            cutoff = cumulative - sorted_probs >= request.top_p
            sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum()
            choice = torch.multinomial(sorted_probs, num_samples=1, generator=generator)
            return int(sorted_idx[choice].item())
        return int(torch.multinomial(probs, num_samples=1, generator=generator).item())

    def _generator_for(self, request: Request) -> Optional[torch.Generator]:
        if request.seed is None:
            return None
        # Seed offset by tokens generated so a preempted request keeps a
        # reproducible sample stream after resume.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(request.seed + len(request.generated_ids))
        return generator
