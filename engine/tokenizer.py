"""Tokenizer wrapper: HF tokenizer is allowed for encode/decode only."""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer

from engine.weights import resolve_model_path


class Tokenizer:
    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        *,
        cache_dir: str | Path | None = None,
        allow_download: bool = True,
    ) -> "Tokenizer":
        model_dir = resolve_model_path(model_id_or_path, cache_dir=cache_dir, allow_download=allow_download)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
        return cls(tokenizer)

    @property
    def eos_token_id(self) -> int | None:
        return self._tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        return self._tokenizer.pad_token_id

    def encode(self, prompt: str, *, add_special_tokens: bool = True) -> List[int]:
        return self._tokenizer.encode(prompt, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: List[int] | torch.Tensor, *, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def encode_tensor(
        self,
        prompt: str,
        *,
        device: torch.device,
        add_special_tokens: bool = True,
    ) -> torch.Tensor:
        ids = self.encode(prompt, add_special_tokens=add_special_tokens)
        return torch.tensor([ids], dtype=torch.long, device=device)
