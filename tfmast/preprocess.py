from __future__ import annotations

import argparse
from pathlib import Path

from tfmast.config import load_config
from tfmast.data.preprocess import build_preprocessed_dataset, save_cache


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/db5.yaml")
    parser.add_argument("--limit-subjects", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    dataset = build_preprocessed_dataset(cfg, limit_subjects=args.limit_subjects)
    output = args.output or Path(cfg.paths.cache_dir) / f"db5_{dataset.preprocessing_hash}.npz"
    save_cache(dataset, output)
    print(f"Saved cache: {output}")
    print(f"Train: {dataset.x_train.shape}, Test: {dataset.x_test.shape}, Classes: {dataset.num_classes}")


if __name__ == "__main__":
    main()
