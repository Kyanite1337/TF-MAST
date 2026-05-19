from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfmast.config import save_config


@dataclass
class TrainResult:
    run_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    best_metrics: dict[str, float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: Any) -> torch.device:
    requested = str(cfg.train.device)
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def make_run_dir(cfg: Any, stage: str, run_name: str | None) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = run_name or stage
    run_dir = Path(cfg.paths.runs_dir) / f"{ts}_{stage}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config_resolved.yaml")
    return run_dir


def append_metrics(run_dir: Path, metrics: dict[str, Any]) -> None:
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False, default=str) + "\n")


def gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    return 0.0


def save_checkpoint(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
