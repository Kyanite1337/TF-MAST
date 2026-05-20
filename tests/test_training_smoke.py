from pathlib import Path

import numpy as np
import torch
import json

from tfmast.config import load_config
from tfmast.training.common import append_metrics
from tfmast.data.datasets import build_synthetic_loaders
from tfmast.report import write_feedback
from tfmast.training.common import EarlyStopper
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
        loaders.val,
        loaders.test,
        init_encoder=tfc_result.best_checkpoint,
        head_name="mlp",
        run_name="ft_smoke",
    )

    assert mae_result.best_checkpoint.exists()
    assert tfc_result.best_checkpoint.exists()
    assert ft_result.best_checkpoint.exists()
    assert "macro_f1" in ft_result.best_metrics
    assert "test/macro_f1" in ft_result.best_metrics
    assert (ft_result.run_dir / "test_metrics.json").exists()

    feedback = write_feedback(ft_result.run_dir, latest=True)
    assert feedback.exists()
    assert (Path(cfg.paths.runs_dir) / "latest_feedback.md").exists()
    assert (Path(cfg.paths.runs_dir) / "latest_metrics.json").exists()


def test_feedback_includes_test_metrics_and_matrix_shape_warning(tmp_path):
    run_dir = tmp_path / "runs" / "finetune_run"
    run_dir.mkdir(parents=True)
    append_metrics(run_dir, {
        "epoch": 1,
        "finetune/accuracy": 0.7,
        "finetune/macro_f1": 0.6,
        "confusion_matrix": [[8, 2], [1, 9]],
    })
    (run_dir / "test_metrics.json").write_text(json.dumps({
        "test/accuracy": 0.65,
        "test/macro_f1": 0.55,
        "test/confusion_matrix": [[7, 3], [2, 8]],
        "expected_num_classes": 53,
    }), encoding="utf-8")

    feedback = write_feedback(run_dir, latest=True)
    text = feedback.read_text(encoding="utf-8")
    latest = json.loads((run_dir.parent / "latest_metrics.json").read_text(encoding="utf-8"))

    assert "## Test Metrics" in text
    assert "test/macro_f1" in text
    assert "expected `53 x 53`, got `2 x 2`" in text
    assert latest["test_metrics"]["test/macro_f1"] == 0.55


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


def test_early_stopper_triggers_after_patience_without_improvement():
    stopper = EarlyStopper(mode="min", patience=2, min_delta=0.01)

    assert stopper.update(1.0).improved
    assert not stopper.update(0.995).should_stop
    assert not stopper.update(0.994).should_stop
    decision = stopper.update(0.993)

    assert decision.should_stop
    assert stopper.best == 1.0
