"""Safetensors loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def resolve_model_path(
    model_id_or_path: str,
    *,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
) -> Path:
    """Return a local directory containing config, tokenizer, and safetensors files."""

    path = Path(model_id_or_path).expanduser()
    if path.exists():
        return path
    if not allow_download:
        raise FileNotFoundError(f"{model_id_or_path!r} is not a local path and downloads are disabled")

    downloaded = snapshot_download(
        repo_id=model_id_or_path,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "*.safetensors",
            "*.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "*.model",
        ],
    )
    return Path(downloaded)


def safetensor_files(model_dir: str | Path) -> List[Path]:
    """List safetensors shards in the order described by the HF index if present."""

    root = Path(model_dir)
    index_files = sorted(root.glob("*.safetensors.index.json"))
    if index_files:
        with index_files[0].open("r", encoding="utf-8") as f:
            index = json.load(f)
        names = sorted(set(index["weight_map"].values()))
        return [root / name for name in names]

    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found under {root}")
    return files


def load_safetensors_state_dict(files: Iterable[Path]) -> Dict[str, torch.Tensor]:
    """Load all shards onto CPU before the model is moved to its runtime device."""

    state_dict: Dict[str, torch.Tensor] = {}
    for file in files:
        shard = load_file(str(file), device="cpu")
        overlap = set(state_dict).intersection(shard)
        if overlap:
            raise ValueError(f"Duplicate tensors in safetensors shards: {sorted(overlap)[:5]}")
        state_dict.update(shard)
    return state_dict
