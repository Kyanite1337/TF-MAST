from __future__ import annotations

import argparse
from pathlib import Path

from tfmast.config import load_config
from tfmast.data.datasets import build_loaders, build_synthetic_loaders
from tfmast.data.preprocess import build_preprocessed_dataset
from tfmast.report import write_feedback
from tfmast.training.stage1_mae import train_mae
from tfmast.training.stage2_tfc import train_tfc
from tfmast.training.stage3_finetune import train_finetune


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/db5.yaml")
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--limit-subjects", type=int, default=1)
    parser.add_argument("--head", default="mlp")
    parser.add_argument("--wandb-mode", default="disabled")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides={
        "wandb.mode": args.wandb_mode,
        "train.mae.epochs": 1,
        "train.tfc.epochs": 1,
        "train.finetune.epochs": 1,
        "train.mae.batch_size": 8,
        "train.tfc.batch_size": 8,
        "train.finetune.batch_size": 8,
        "train.max_batches": 2,
        "model.embed_dim": 32,
        "model.depths": [1, 1],
        "model.num_heads": [2, 4],
    })
    if args.real_data:
        dataset = build_preprocessed_dataset(cfg, limit_subjects=args.limit_subjects)
        loaders = build_loaders(dataset, cfg)
    else:
        loaders = build_synthetic_loaders(batch_size=8, num_classes=53 if cfg.data.class_mode == "53_with_rest" else 52)
    mae = train_mae(cfg, loaders.mae, run_name="smoke_mae")
    tfc = train_tfc(cfg, loaders.tfc, init_encoder=mae.best_checkpoint, run_name="smoke_tfc")
    ft = train_finetune(cfg, loaders.train, loaders.test, init_encoder=tfc.best_checkpoint, head_name=args.head, run_name=f"smoke_ft_{args.head}")
    feedback = write_feedback(ft.run_dir, latest=True)
    print(f"Smoke run feedback: {feedback}")


if __name__ == "__main__":
    main()
