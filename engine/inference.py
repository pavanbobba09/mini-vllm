"""Small inference helpers for Milestone 1."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch

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
