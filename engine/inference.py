"""Inference paths: the M1 full-recompute reference and the M2 paged decode."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import torch

from engine.block_manager import BlockManager
from engine.kv_cache import PagedKVCache, make_meta
from engine.model import MiniQwenForCausalLM
from engine.tokenizer import Tokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class MiniVLLMEngine:
    """Convenience wrapper that keeps model and tokenizer loaded together."""

    def __init__(self, model: MiniQwenForCausalLM, tokenizer: Tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str = DEFAULT_MODEL_ID,
        *,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = "auto",
        cache_dir: str | Path | None = None,
        allow_download: bool = True,
    ) -> "MiniVLLMEngine":
        model = MiniQwenForCausalLM.from_pretrained(
            model_id_or_path,
            device=device,
            dtype=dtype,
            cache_dir=cache_dir,
            allow_download=allow_download,
        )
        tokenizer = Tokenizer.from_pretrained(
            model_id_or_path,
            cache_dir=cache_dir,
            allow_download=allow_download,
        )
        return cls(model=model, tokenizer=tokenizer)

    @torch.no_grad()
    def logits(self, prompt: str, *, add_special_tokens: bool = True) -> torch.Tensor:
        input_ids = self.tokenizer.encode_tensor(
            prompt,
            device=next(self.model.parameters()).device,
            add_special_tokens=add_special_tokens,
        )
        return self.model(input_ids)

    @torch.no_grad()
    def generate_greedy(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        add_special_tokens: bool = True,
    ) -> List[int]:
        input_ids = self.tokenizer.encode_tensor(
            prompt,
            device=next(self.model.parameters()).device,
            add_special_tokens=add_special_tokens,
        )
        output_ids = greedy_decode(
            self.model,
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return output_ids[0].detach().cpu().tolist()


@torch.no_grad()
def greedy_decode(
    model: MiniQwenForCausalLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Greedy decoding baseline that recomputes the full prefix every step."""

    generated = input_ids
    finished = torch.zeros(generated.shape[0], dtype=torch.bool, device=generated.device)
    for _ in range(max_new_tokens):
        logits = model(generated)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        if eos_token_id is not None:
            finished |= next_token.squeeze(-1).eq(eos_token_id)
            if torch.all(finished):
                break
    return generated


# The M1 oracle: later milestones are verified by matching this cache-free path.
reference_generate = greedy_decode


@torch.no_grad()
def paged_prefill(
    model: MiniQwenForCausalLM,
    kv_cache: PagedKVCache,
    block_manager: BlockManager,
    seq_id: int,
    prompt_ids: Sequence[int],
) -> torch.Tensor:
    """Run the whole prompt once, filling its KV blocks; returns last-position logits [vocab]."""

    device = next(model.parameters()).device
    num_tokens = len(prompt_ids)
    block_manager.allocate_sequence(seq_id, num_tokens)
    meta = make_meta(
        slot_mapping=block_manager.slots_for_range(seq_id, 0, num_tokens),
        block_tables=[block_manager.block_table(seq_id)],
        context_lens=[num_tokens],
        is_prefill=True,
        device=device,
    )
    input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    position_ids = torch.arange(num_tokens, device=device).unsqueeze(0)
    logits = model(input_ids, position_ids=position_ids, kv_cache=kv_cache, meta=meta)
    return logits[0, -1, :]


@torch.no_grad()
def paged_decode_step(
    model: MiniQwenForCausalLM,
    kv_cache: PagedKVCache,
    block_manager: BlockManager,
    seq_ids: Sequence[int],
    last_tokens: Sequence[int],
) -> torch.Tensor:
    """One batched decode step over ragged sequences; returns logits [batch, vocab]."""

    device = next(model.parameters()).device
    slot_mapping = [block_manager.append_token(seq_id) for seq_id in seq_ids]
    context_lens = [block_manager.num_tokens(seq_id) for seq_id in seq_ids]
    meta = make_meta(
        slot_mapping=slot_mapping,
        block_tables=[block_manager.block_table(seq_id) for seq_id in seq_ids],
        context_lens=context_lens,
        is_prefill=False,
        device=device,
    )
    input_ids = torch.tensor(list(last_tokens), dtype=torch.long, device=device).unsqueeze(1)
    # The newest token sits at position context_len - 1 (0-indexed RoPE position).
    position_ids = torch.tensor([n - 1 for n in context_lens], dtype=torch.long, device=device).unsqueeze(1)
    logits = model(input_ids, position_ids=position_ids, kv_cache=kv_cache, meta=meta)
    return logits[:, -1, :]


@torch.no_grad()
def paged_generate(
    model: MiniQwenForCausalLM,
    kv_cache: PagedKVCache,
    block_manager: BlockManager,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    seq_id: int = 0,
) -> List[int]:
    """Greedy decode via the paged cache; must match reference_generate token for token."""

    try:
        last_logits = paged_prefill(model, kv_cache, block_manager, seq_id, prompt_ids)
        output_ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            next_token = int(torch.argmax(last_logits, dim=-1).item())
            output_ids.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break
            step_logits = paged_decode_step(model, kv_cache, block_manager, [seq_id], [next_token])
            last_logits = step_logits[0]
        return output_ids
    finally:
        if block_manager.has_sequence(seq_id):
            block_manager.free_sequence(seq_id)


def logits_for_prompt(
    prompt: str,
    *,
    model_id_or_path: str = DEFAULT_MODEL_ID,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = "auto",
    cache_dir: str | Path | None = None,
) -> torch.Tensor:
    """Deliverable helper: load the model and return logits for one prompt."""

    engine = MiniVLLMEngine.from_pretrained(
        model_id_or_path,
        device=device,
        dtype=dtype,
        cache_dir=cache_dir,
    )
    return engine.logits(prompt)
