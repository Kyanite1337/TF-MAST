#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


FIELDS = [
    "experiment",
    "status",
    "started_at",
    "finished_at",
    "duration_sec",
    "git_commit",
    "run_dir",
    "init_checkpoint",
    "pretrain",
    "head",
    "bypass",
    "raw_bypass_type",
    "window_ms",
    "stride_ms",
    "class_mode",
    "train_reps",
    "test_reps",
    "preprocess",
    "embed_dim",
    "depths",
    "num_heads",
    "patch_size",
    "head_dropout",
    "mamba_d_state",
    "mamba_d_conv",
    "mamba_expand",
    "loss",
    "class_weight",
    "label_smoothing",
    "lr",
    "weight_decay",
    "batch_size",
    "max_epochs",
    "best_epoch",
    "seed",
    "amp",
    "num_workers",
    "val_loss",
    "val_acc",
    "val_macro_f1",
    "test_loss",
    "test_acc",
    "test_macro_f1",
    "gap_acc",
    "gap_macro_f1",
    "worst_5_classes_by_f1",
    "best_5_classes_by_f1",
    "rest_class_f1",
    "non_rest_macro_f1",
    "best_checkpoint",
    "last_checkpoint",
    "config_path",
    "metrics_path",
    "test_metrics_path",
    "feedback_path",
    "error_log",
]


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_best_from_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if not rows:
        return {}
    return min(rows, key=lambda row: row.get("val_loss", row.get("finetune/val_loss", float("inf"))))


def _join(value) -> str:
    if isinstance(value, list):
        return "/".join(str(v) for v in value)
    return "" if value is None else str(value)


def _fmt(value) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.10f}".rstrip("0").rstrip(".")


def _preprocess_label(cfg: dict) -> str:
    preprocess = []
    pp = cfg.get("preprocess", {})
    if pp.get("demean"):
        preprocess.append("demean")
    if pp.get("zscore"):
        preprocess.append("zscore")
    if pp.get("notch", {}).get("enabled"):
        preprocess.append(f"notch{pp['notch']['freq']}")
    if pp.get("bandpass", {}).get("enabled"):
        bp = pp["bandpass"]
        preprocess.append(f"bandpass{bp['low']}-{bp['high']}")
    return "+".join(preprocess)


def _per_class_summary(matrix) -> dict[str, str]:
    if not matrix:
        return {"worst": "", "best": "", "rest": "", "non_rest": ""}
    cm = np.asarray(matrix, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        return {"worst": "", "best": "", "rest": "", "non_rest": ""}
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred = cm.sum(axis=0)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)

    def pack(indices) -> str:
        return ";".join(f"{int(i)}:{_fmt(f1[i])}" for i in indices)

    order = np.argsort(f1)
    return {
        "worst": pack(order[:5]),
        "best": pack(order[-5:][::-1]),
        "rest": _fmt(f1[0]) if len(f1) else "",
        "non_rest": _fmt(float(f1[1:].mean())) if len(f1) > 1 else "",
    }


def _metric(best: dict, *names):
    for name in names:
        if name in best:
            return best[name]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--xlsx", required=True, type=Path)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--bypass", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--status", default="done")
    parser.add_argument("--started-at", default="")
    parser.add_argument("--finished-at", default="")
    parser.add_argument("--duration-sec", default="")
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--error-log", default="")
    args = parser.parse_args()

    cfg = yaml.safe_load((args.run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    summary = _read_json(args.run_dir / "summary.json")
    test = _read_json(args.run_dir / "test_metrics.json")
    best = summary.get("best_metrics") or _read_best_from_metrics(args.run_dir / "metrics.jsonl")
    finetune = cfg.get("train", {}).get("finetune", {})

    val_acc = _metric(best, "finetune/accuracy", "accuracy")
    val_macro_f1 = _metric(best, "finetune/macro_f1", "macro_f1")
    test_acc = _metric(test, "test/accuracy")
    test_macro_f1 = _metric(test, "test/macro_f1")
    per_class = _per_class_summary(test.get("test/confusion_matrix"))

    row = {
        "experiment": args.experiment,
        "status": args.status,
        "started_at": args.started_at,
        "finished_at": args.finished_at,
        "duration_sec": args.duration_sec,
        "git_commit": args.git_commit,
        "run_dir": str(args.run_dir),
        "init_checkpoint": args.init_checkpoint,
        "pretrain": args.pretrain,
        "head": args.head,
        "bypass": args.bypass,
        "raw_bypass_type": "token_add",
        "window_ms": cfg["data"]["window_ms"],
        "stride_ms": cfg["data"]["stride_ms"],
        "class_mode": cfg["data"]["class_mode"],
        "train_reps": _join(cfg["data"]["train_reps"]),
        "test_reps": _join(cfg["data"]["test_reps"]),
        "preprocess": _preprocess_label(cfg),
        "embed_dim": cfg["model"]["embed_dim"],
        "depths": _join(cfg["model"]["depths"]),
        "num_heads": _join(cfg["model"]["num_heads"]),
        "patch_size": _join(cfg["model"]["patch_size"]),
        "head_dropout": cfg["head"]["dropout"],
        "mamba_d_state": cfg["head"]["mamba_d_state"],
        "mamba_d_conv": cfg["head"]["mamba_d_conv"],
        "mamba_expand": cfg["head"]["mamba_expand"],
        "loss": finetune.get("loss", "cross_entropy"),
        "class_weight": finetune.get("class_weight", "none"),
        "label_smoothing": finetune.get("label_smoothing"),
        "lr": finetune.get("lr"),
        "weight_decay": finetune.get("weight_decay"),
        "batch_size": finetune.get("batch_size"),
        "max_epochs": finetune.get("epochs"),
        "best_epoch": _metric(best, "epoch"),
        "seed": cfg["train"].get("seed"),
        "amp": cfg["train"].get("amp"),
        "num_workers": cfg["train"].get("num_workers"),
        "val_loss": _fmt(_metric(best, "finetune/val_loss", "val_loss", "loss")),
        "val_acc": _fmt(val_acc),
        "val_macro_f1": _fmt(val_macro_f1),
        "test_loss": _fmt(_metric(test, "test/loss")),
        "test_acc": _fmt(test_acc),
        "test_macro_f1": _fmt(test_macro_f1),
        "gap_acc": _fmt(float(val_acc) - float(test_acc)) if val_acc is not None and test_acc is not None else "",
        "gap_macro_f1": _fmt(float(val_macro_f1) - float(test_macro_f1)) if val_macro_f1 is not None and test_macro_f1 is not None else "",
        "worst_5_classes_by_f1": per_class["worst"],
        "best_5_classes_by_f1": per_class["best"],
        "rest_class_f1": per_class["rest"],
        "non_rest_macro_f1": per_class["non_rest"],
        "best_checkpoint": str(args.run_dir / "best.pt"),
        "last_checkpoint": str(args.run_dir / "last.pt"),
        "config_path": str(args.run_dir / "config_resolved.yaml"),
        "metrics_path": str(args.run_dir / "metrics.jsonl"),
        "test_metrics_path": str(args.run_dir / "test_metrics.json"),
        "feedback_path": str(args.run_dir / "feedback.md"),
        "error_log": args.error_log,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists()
    with args.output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    df = pd.read_csv(args.output)
    with pd.ExcelWriter(args.xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="raw_bypass_ablation", index=False)
        ws = writer.book["raw_bypass_ablation"]
        ws.freeze_panes = "A2"
        for col_idx, name in enumerate(FIELDS, start=1):
            width = min(max(len(name) + 2, 12), 48)
            ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = width


if __name__ == "__main__":
    main()
