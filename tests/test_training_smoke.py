from pathlib import Path

import numpy as np
import torch

from tfmast.config import load_config
from tfmast.data.datasets import build_synthetic_loaders
from tfmast.report import write_feedback
from tfmast.training.stage1_mae import train_mae
from tfmast.training.stage2_tfc import train_tfc
from tfmast.training.stage3_finetune import train_finetune
from tfmast.utils.wandb import WandbLogger


def _tiny_cfg(tmp_path: Path):
    return load_config(overrides={
        "paths.runs_dir": str(tmp_path / "runs"),
        "paths.checkpoints_dir": str(tmp_path / "checkpoints"),
        "model.embed_dim": 32,
        "model.depths": [1, 1],
        "model.num_heads": [2, 4],
        "train.mae.epochs": 1,
        "train.tfc.epochs": 1,
        "train.finetune.epochs": 1,
        "train.mae.batch_size": 8,
        "train.tfc.batch_size": 8,
        "train.finetune.batch_size": 8,
        "train.max_batches": 1,
        "wandb.mode": "disabled",
    })


def test_three_stage_training_smoke_and_feedback_files(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    loaders = build_synthetic_loaders(num_train=24, num_test=12, batch_size=8, num_classes=53)

    mae_result = train_mae(cfg, loaders.mae, run_name="mae_smoke")
    tfc_result = train_tfc(cfg, loaders.tfc, init_encoder=mae_result.best_checkpoint, run_name="tfc_smoke")
    ft_result = train_finetune(
        cfg,
        loaders.train,
        loaders.test,
        init_encoder=tfc_result.best_checkpoint,
        head_name="mlp",
        run_name="ft_smoke",
    )

    assert mae_result.best_checkpoint.exists()
    assert tfc_result.best_checkpoint.exists()
    assert ft_result.best_checkpoint.exists()
    assert "macro_f1" in ft_result.best_metrics

    feedback = write_feedback(ft_result.run_dir, latest=True)
    assert feedback.exists()
    assert (Path(cfg.paths.runs_dir) / "latest_feedback.md").exists()
    assert (Path(cfg.paths.runs_dir) / "latest_metrics.json").exists()


def test_wandb_disabled_and_offline_modes_do_not_require_network(tmp_path, monkeypatch):
    cfg = _tiny_cfg(tmp_path)
    logger = WandbLogger(cfg, run_name="disabled", stage="test")
    logger.log({"loss": 1.0}, step=1)
    logger.finish()

    cfg = load_config(base=cfg, overrides={"wandb.mode": "offline"})
    monkeypatch.setenv("WANDB_MODE", "offline")
    logger = WandbLogger(cfg, run_name="offline", stage="test")
    logger.log({"loss": 1.0}, step=1)
    logger.finish()
