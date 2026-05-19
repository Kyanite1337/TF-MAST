from __future__ import annotations

import argparse
from pathlib import Path

from tfmast.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/db5.yaml")
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--limit-subjects", type=int, default=1)
    parser.add_argument("--head", default="mlp")
    parser.add_argument("--wandb-mode", default="disabled")
    args = parser.parse_args(argv)
    overrides = {
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
    }
    result = run_pipeline(
        config_path=args.config,
        overrides=overrides,
        synthetic=not args.real_data,
        limit_subjects=args.limit_subjects,
        experiment="smoke",
        head=args.head,
    )
    print(f"Smoke run feedback: {result['finetune'].run_dir / 'feedback.md'}")


if __name__ == "__main__":
    main()
