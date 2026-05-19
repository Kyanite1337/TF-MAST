from __future__ import annotations

import os
from typing import Any

from tfmast.config import to_dict


class WandbLogger:
    def __init__(self, cfg: Any, *, run_name: str, stage: str):
        self.mode = str(cfg.wandb.mode)
        self.run = None
        if self.mode == "disabled":
            return
        os.environ["WANDB_MODE"] = self.mode
        try:
            import wandb
        except Exception:
            return
        self._wandb = wandb
        self.run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            tags=list(cfg.wandb.tags) + [stage],
            name=run_name,
            mode=self.mode,
            config=to_dict(cfg),
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
