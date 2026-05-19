from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "data_root": "data/ninapro_db5",
        "cache_dir": "data/cache",
        "runs_dir": "runs",
        "checkpoints_dir": "checkpoints",
    },
    "data": {
        "root": "data/ninapro_db5",
        "subjects": list(range(1, 11)),
        "exercises": ["E1", "E2", "E3"],
        "sampling_rate": 200,
        "channels": 16,
        "class_mode": "53_with_rest",
        "train_reps": [1, 3, 4, 6],
        "test_reps": [2, 5],
        "window_ms": 200,
        "stride_ms": 100,
    },
    "preprocess": {
        "demean": True,
        "notch": {"enabled": True, "freq": 50.0, "quality": 30.0},
        "bandpass": {"enabled": False, "low": 20.0, "high": 95.0, "order": 4},
        "zscore": True,
    },
    "model": {
        "embed_dim": 128,
        "depths": [2, 2, 2],
        "num_heads": [4, 4, 8],
        "mlp_ratio": 4.0,
        "dropout": 0.1,
        "patch_size": [1, 4],
        "decoder_depth": 2,
        "projector_dim": 128,
    },
    "mae": {
        "mask_ratio": 0.5,
        "mask_strategies": ["block", "temporal", "sensor", "multi_scale"],
        "decoder_mask_ratio": 0.5,
    },
    "tfc": {
        "temperature": 0.2,
        "lambda_contrastive": 0.5,
        "margin": 1.0,
        "freq_alpha": 0.5,
    },
    "train": {
        "seed": 42,
        "device": "cuda",
        "amp": True,
        "grad_accum_steps": 1,
        "max_batches": None,
        "auto_batch": True,
        "log_every_steps": 0,
        "num_workers": 8,
        "mae": {"epochs": 300, "batch_size": 1024, "lr": 3e-4, "weight_decay": 0.05},
        "tfc": {"epochs": 200, "batch_size": 768, "lr": 1e-4, "weight_decay": 0.05},
        "finetune": {"epochs": 150, "batch_size": 512, "lr": 1e-3, "weight_decay": 0.05, "label_smoothing": 0.1},
    },
    "head": {
        "name": "mlp",
        "dropout": 0.3,
        "bypass": True,
        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
    },
    "wandb": {
        "mode": "disabled",
        "project": "tfmast-ninapro-db5",
        "entity": None,
        "group": None,
        "tags": [],
        "log_model": False,
    },
}


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _set_dotted(target: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = target
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def to_dict(cfg: Any) -> Any:
    if isinstance(cfg, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(cfg).items()}
    if isinstance(cfg, list):
        return [to_dict(v) for v in cfg]
    return cfg


def load_config(
    path: str | Path | None = None,
    *,
    base: Any | None = None,
    overrides: dict[str, Any] | None = None,
) -> SimpleNamespace:
    data = deepcopy(to_dict(base) if base is not None else DEFAULT_CONFIG)
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        _deep_update(data, loaded)
    for key, value in (overrides or {}).items():
        _set_dotted(data, key, value)
    if "root" in data.get("data", {}):
        data["paths"]["data_root"] = data["data"]["root"]
    return _to_namespace(data)


def save_config(cfg: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)
