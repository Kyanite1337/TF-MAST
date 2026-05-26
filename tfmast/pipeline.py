from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tfmast.config import load_config
from tfmast.data.datasets import build_loaders, build_synthetic_loaders
from tfmast.data.preprocess import build_preprocessed_dataset
from tfmast.report import write_feedback
from tfmast.train import _normalize_overrides, _parse_override
from tfmast.training.common import TrainResult
from tfmast.training.stage1_mae import train_mae
from tfmast.training.stage2_tfc import train_tfc
from tfmast.training.stage3_finetune import train_finetune


def _loaders_for_pipeline(cfg: Any, *, synthetic: bool, limit_subjects: int | None):
    print("[Pipeline] Preparing data", flush=True)
    if synthetic:
        print("[Pipeline] Using synthetic DB5-like data", flush=True)
        return build_synthetic_loaders(batch_size=8, num_classes=53 if cfg.data.class_mode == "53_with_rest" else 52)
    dataset = build_preprocessed_dataset(cfg, limit_subjects=limit_subjects)
    return build_loaders(dataset, cfg)


def run_pipeline(
    *,
    config_path: str | Path | None = "configs/db5.yaml",
    overrides: dict[str, Any] | None = None,
    synthetic: bool = False,
    limit_subjects: int | None = None,
    experiment: str = "pipeline",
    head: str | None = None,
) -> dict[str, TrainResult]:
    cfg = load_config(config_path, overrides=overrides or {})
    loaders = _loaders_for_pipeline(cfg, synthetic=synthetic, limit_subjects=limit_subjects)

    print("[Pipeline] Starting stage: MAE", flush=True)
    mae = train_mae(cfg, loaders.mae, loaders.mae_val, run_name=f"{experiment}_mae")
    print(f"[Pipeline] MAE best checkpoint: {mae.best_checkpoint}", flush=True)
    print("[Pipeline] Starting stage: TFC", flush=True)
    tfc = train_tfc(cfg, loaders.tfc, loaders.tfc_val, init_encoder=mae.best_checkpoint, run_name=f"{experiment}_tfc")
    print(f"[Pipeline] TFC best checkpoint: {tfc.best_checkpoint}", flush=True)
    print("[Pipeline] Starting stage: fine-tune", flush=True)
    finetune = train_finetune(
        cfg,
        loaders.train,
        loaders.val,
        loaders.test,
        init_encoder=tfc.best_checkpoint,
        head_name=head or cfg.head.name,
        run_name=f"{experiment}_finetune_{head or cfg.head.name}",
    )
    write_feedback(finetune.run_dir, latest=True)
    return {"mae": mae, "tfc": tfc, "finetune": finetune}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run MAE -> TFC -> fine-tune sequentially.")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--config", default="configs/db5.yaml")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--limit-subjects", type=int, default=None)
    parser.add_argument("--experiment", default="pipeline")
    parser.add_argument("--head", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args(argv)
    overrides = _normalize_overrides(_parse_override(args.overrides))
    if args.max_batches is not None:
        overrides["train.max_batches"] = args.max_batches
    result = run_pipeline(
        config_path=args.config,
        overrides=overrides,
        synthetic=args.synthetic,
        limit_subjects=args.limit_subjects,
        experiment=args.experiment,
        head=args.head,
    )
    print("Pipeline complete")
    for stage, stage_result in result.items():
        print(f"{stage}: run_dir={stage_result.run_dir} best={stage_result.best_checkpoint}")


if __name__ == "__main__":
    main()
