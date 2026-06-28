"""Configuration handling for the custom Qwen2 forward pass."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass(frozen=True)
class ModelConfig:
    """The subset of HF Qwen2 config fields needed by our modules."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    hidden_act: str
    attention_bias: bool
    mlp_bias: bool
    tie_word_embeddings: bool
    torch_dtype: Optional[str] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    head_dim: Optional[int] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    use_sliding_window: bool = False

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ModelConfig":
        if raw.get("model_type") not in (None, "qwen2"):
            raise ValueError(f"Only Qwen2-style configs are supported, got {raw.get('model_type')!r}")

        rope_scaling = raw.get("rope_scaling")
        if rope_scaling not in (None, {}):
            raise ValueError("RoPE scaling is not implemented in Milestone 1.")

        use_sliding_window = bool(raw.get("use_sliding_window", False))
        if use_sliding_window:
            raise ValueError("Sliding-window attention is not implemented in Milestone 1.")

        num_kv_heads = raw.get("num_key_value_heads", raw["num_attention_heads"])
        return cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(num_kv_heads),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            hidden_act=str(raw.get("hidden_act", "silu")),
            # Qwen2 configs may omit this field, but the architecture default has Q/K/V bias.
            attention_bias=bool(raw.get("attention_bias", True)),
            mlp_bias=bool(raw.get("mlp_bias", False)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            torch_dtype=raw.get("torch_dtype"),
            bos_token_id=raw.get("bos_token_id"),
            eos_token_id=raw.get("eos_token_id"),
            pad_token_id=raw.get("pad_token_id"),
            head_dim=raw.get("head_dim"),
            rope_scaling=rope_scaling,
            use_sliding_window=use_sliding_window,
        )

    @property
    def attention_head_dim(self) -> int:
        # Qwen2.5 stores head_dim explicitly; older configs infer it.
        return int(self.head_dim or (self.hidden_size // self.num_attention_heads))

    @property
    def kv_hidden_size(self) -> int:
        return self.num_key_value_heads * self.attention_head_dim

    @property
    def query_hidden_size(self) -> int:
        return self.num_attention_heads * self.attention_head_dim


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Pick a practical default device for local iteration."""

    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype: str | torch.dtype | None, device: torch.device) -> torch.dtype:
    """Choose a dtype that prioritizes correctness on CPU and speed on accelerators."""

    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype not in (None, "auto"):
        name = dtype.removeprefix("torch.")
        try:
            return getattr(torch, name)
        except AttributeError as exc:
            raise ValueError(f"Unknown torch dtype {dtype!r}") from exc
    if device.type == "cpu":
        return torch.float32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16
