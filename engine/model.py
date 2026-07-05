"""A minimal Qwen2 causal language model."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn

from engine.config import ModelConfig, resolve_device, resolve_dtype
from engine.kv_cache import PagedBatchMeta, PagedKVCache
from engine.layers import QwenDecoderLayer, QwenRMSNorm
from engine.rope import RotaryEmbedding
from engine.weights import (
    load_safetensors_state_dict,
    remap_hf_state_dict,
    resolve_model_path,
    safetensor_files,
)


class MiniQwenModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [QwenDecoderLayer(config=config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # One rotary table for the whole stack; every layer shares the same angles.
        self.rotary_emb = RotaryEmbedding(
            config.attention_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[PagedKVCache] = None,
        meta: Optional[PagedBatchMeta] = None,
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = make_position_ids(input_ids, attention_mask)

        hidden_states = self.embed_tokens(input_ids)
        rope = self.rotary_emb(position_ids, dtype=hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                rope=rope,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                meta=meta,
            )
        return self.norm(hidden_states)


class MiniQwenForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = MiniQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        *,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = "auto",
        cache_dir: str | Path | None = None,
        allow_download: bool = True,
    ) -> "MiniQwenForCausalLM":
        model_dir = resolve_model_path(model_id_or_path, cache_dir=cache_dir, allow_download=allow_download)
        config = ModelConfig.from_json_file(model_dir / "config.json")
        model = cls(config)

        state_dict = load_safetensors_state_dict(safetensor_files(model_dir))
        state_dict = remap_hf_state_dict(state_dict, config.num_hidden_layers)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if config.tie_word_embeddings and "lm_head.weight" in missing:
            model.lm_head.weight = model.model.embed_tokens.weight
            missing = [key for key in missing if key != "lm_head.weight"]
        if missing or unexpected:
            raise RuntimeError(f"State dict mismatch. Missing={missing}, unexpected={unexpected}")

        runtime_device = resolve_device(device)
        runtime_dtype = resolve_dtype(dtype, runtime_device)
        model.to(device=runtime_device, dtype=runtime_dtype)
        model.eval()
        return model

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[PagedKVCache] = None,
        meta: Optional[PagedBatchMeta] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            meta=meta,
        )
        return self.lm_head(hidden_states)


def make_position_ids(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Create Qwen-compatible absolute positions, respecting padding when present."""

    if attention_mask is None:
        batch_size, seq_len = input_ids.shape
        return torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)
