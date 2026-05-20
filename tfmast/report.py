from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import torch


def _read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    rows = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _matrix_shape(metrics: dict[str, Any], key: str) -> tuple[int, int] | None:
    matrix = metrics.get(key)
    if not isinstance(matrix, list) or not matrix:
        return None
    first = matrix[0]
    if not isinstance(first, list):
        return None
    return len(matrix), len(first)


def _shape_warning(metrics: dict[str, Any], key: str, expected: int | None) -> str | None:
    shape = _matrix_shape(metrics, key)
    if shape is None or expected is None:
        return None
    if shape != (expected, expected):
        return f"expected `{expected} x {expected}`, got `{shape[0]} x {shape[1]}`"
    return None


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if "confusion_matrix" not in key}


def _run_stage_and_experiment(run_dir: Path) -> tuple[str | None, str | None]:
    parts = run_dir.name.split("_")
    if len(parts) < 3:
        return None, None
    stage = parts[1]
    core = parts[2:]
    if stage in {"mae", "tfc"} and core and core[-1] == stage:
        core = core[:-1]
    elif stage == "finetune" and "finetune" in core:
        core = core[: core.index("finetune")]
    return stage, "_".join(core) if core else None


def _stage_summaries(run_dir: Path) -> dict[str, dict[str, Any]]:
    _, experiment = _run_stage_and_experiment(run_dir)
    if not experiment:
        return {}
    summaries: dict[str, dict[str, Any]] = {}
    candidates = sorted((p for p in run_dir.parent.iterdir() if p.is_dir()), key=lambda p: p.name)
    for candidate in candidates:
        stage, candidate_experiment = _run_stage_and_experiment(candidate)
        if stage not in {"mae", "tfc", "finetune"} or candidate_experiment != experiment:
            continue
        rows = _read_metrics(candidate)
        if not rows:
            continue
        best = {}
        for row in rows:
            if "finetune/macro_f1" in row and row["finetune/macro_f1"] >= best.get("finetune/macro_f1", -1):
                best = row
            elif "val_loss" in row and row["val_loss"] <= best.get("val_loss", float("inf")):
                best = row
        summaries[stage] = {
            "run_dir": str(candidate),
            "best_checkpoint": str(candidate / "best.pt"),
            "best_metrics": _compact_metrics(best),
        }
        test_metrics = _read_json(candidate / "test_metrics.json")
        if test_metrics:
            summaries[stage]["test_metrics"] = _compact_metrics(test_metrics)
    return summaries


def write_feedback(run_dir: str | Path, *, latest: bool = False) -> Path:
    run_dir = Path(run_dir)
    rows = _read_metrics(run_dir)
    last = rows[-10:]
    best = {}
    for row in rows:
        if "finetune/macro_f1" in row and row["finetune/macro_f1"] >= best.get("finetune/macro_f1", -1):
            best = row
        elif "val_loss" in row and "finetune/macro_f1" not in best and row["val_loss"] <= best.get("val_loss", float("inf")):
            best = row
        elif "train_loss" in row and not best:
            best = row
    test_metrics = _read_json(run_dir / "test_metrics.json")
    expected = test_metrics.get("expected_num_classes")
    val_shape_warning = _shape_warning(best, "confusion_matrix", expected)
    test_shape_warning = _shape_warning(test_metrics, "test/confusion_matrix", expected)
    stage_summaries = _stage_summaries(run_dir)
    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
        "best_metrics": best,
        "test_metrics": test_metrics,
        "stage_summaries": stage_summaries,
        "last_metrics": rows[-1] if rows else {},
    }
    feedback = run_dir / "feedback.md"
    lines = [
        f"# TF-MAST Run Feedback",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Python: `{platform.python_version()}`",
        f"- Torch: `{torch.__version__}`",
        f"- CUDA available: `{torch.cuda.is_available()}`",
        f"- Best checkpoint: `{run_dir / 'best.pt'}`",
    ]
    if stage_summaries:
        lines.extend([
            "",
            "## Stage Summary",
            "```json",
            json.dumps(stage_summaries, indent=2, ensure_ascii=False),
            "```",
        ])
    if val_shape_warning or test_shape_warning:
        lines.extend(["", "## Warnings"])
        if val_shape_warning:
            lines.append(f"- Validation confusion matrix shape mismatch: {val_shape_warning}")
        if test_shape_warning:
            lines.append(f"- Test confusion matrix shape mismatch: {test_shape_warning}")
    lines.extend([
        "",
        "## Best Metrics",
        "```json",
        json.dumps(best, indent=2, ensure_ascii=False),
        "```",
    ])
    if test_metrics:
        lines.extend([
            "",
            "## Test Metrics",
            "```json",
            json.dumps(test_metrics, indent=2, ensure_ascii=False),
            "```",
        ])
    lines.extend([
        "",
        "## Last 10 Epochs",
        "```json",
        json.dumps(last, indent=2, ensure_ascii=False),
        "```",
    ])
    feedback.write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if latest:
        runs_root = run_dir.parent
        runs_root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(feedback, runs_root / "latest_feedback.md")
        (runs_root / "latest_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return feedback


def compare_runs(run_dirs: list[Path]) -> str:
    rows = []
    for run in run_dirs:
        metrics = _read_metrics(run)
        last = metrics[-1] if metrics else {}
        rows.append((run.name, last.get("finetune/accuracy", last.get("train_loss")), last.get("finetune/macro_f1", "")))
    lines = ["| run | accuracy/loss | macro_f1 |", "|---|---:|---:|"]
    lines.extend(f"| {name} | {a} | {f1} |" for name, a, f1 in rows)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--compare", nargs="*", type=Path)
    args = parser.parse_args(argv)
    if args.run:
        path = write_feedback(args.run, latest=True)
        print(path)
    if args.compare:
        print(compare_runs(args.compare))


if __name__ == "__main__":
    main()
