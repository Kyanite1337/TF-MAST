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


def _parse_override(items: list[str]) -> dict:
    out = {}
    for item in items:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        if raw.lower() in {"true", "false"}:
            value = raw.lower() == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        out[key] = value
    return out


def _is_oom(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower()


def _build_loaders(cfg, args):
    if args.synthetic:
        return None, build_synthetic_loaders(batch_size=8, num_classes=53 if cfg.data.class_mode == "53_with_rest" else 52)
    dataset = build_preprocessed_dataset(cfg, limit_subjects=args.limit_subjects)
    return dataset, build_loaders(dataset, cfg)


def _run_stage(stage, cfg, loaders, overrides):
    if stage == "mae":
        return train_mae(cfg, loaders.mae, run_name=str(overrides.get("experiment", "mae")))
    if stage == "tfc":
        init = overrides.get("init")
        return train_tfc(cfg, loaders.tfc, init_encoder=Path(init) if init else None, run_name=str(overrides.get("experiment", "tfc")))
    if stage == "finetune":
        init = overrides.get("init")
        return train_finetune(cfg, loaders.train, loaders.test, init_encoder=Path(init) if init else None, head_name=str(overrides.get("head", cfg.head.name)), run_name=str(overrides.get("experiment", "finetune")))
    raise ValueError(f"Unknown stage={stage}")


def _batch_key(stage: str) -> str:
    return {"mae": "train.mae.batch_size", "tfc": "train.tfc.batch_size", "finetune": "train.finetune.batch_size"}[stage]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--config", default="configs/db5.yaml")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--limit-subjects", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args(argv)
    overrides = _parse_override(args.overrides)
    if args.max_batches is not None:
        overrides["train.max_batches"] = args.max_batches
    cfg = load_config(args.config, overrides=overrides)
    stage = overrides.get("stage", "mae")
    _, loaders = _build_loaders(cfg, args)
    attempts = 0
    while True:
        try:
            result = _run_stage(stage, cfg, loaders, overrides)
            break
        except RuntimeError as exc:
            if not (bool(cfg.train.auto_batch) and _is_oom(exc) and attempts < 5):
                raise
            attempts += 1
            key = _batch_key(stage)
            current = int(overrides.get(key, getattr(getattr(cfg.train, stage if stage != "finetune" else "finetune"), "batch_size")))
            new_batch = max(1, current // 2)
            if new_batch == current:
                raise
            print(f"[auto-batch] CUDA OOM at batch={current}; retrying with batch={new_batch}")
            overrides[key] = new_batch
            cfg = load_config(args.config, overrides=overrides)
            _, loaders = _build_loaders(cfg, args)
    print(f"Run dir: {result.run_dir}")
    print(f"Best checkpoint: {result.best_checkpoint}")
    print(write_feedback(result.run_dir, latest=True))


if __name__ == "__main__":
    main()
