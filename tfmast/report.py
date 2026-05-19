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


def write_feedback(run_dir: str | Path, *, latest: bool = False) -> Path:
    run_dir = Path(run_dir)
    rows = _read_metrics(run_dir)
    last = rows[-10:]
    best = {}
    for row in rows:
        if "finetune/macro_f1" in row and row["finetune/macro_f1"] >= best.get("finetune/macro_f1", -1):
            best = row
        elif "train_loss" in row and not best:
            best = row
    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
        "best_metrics": best,
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
        "",
        "## Best Metrics",
        "```json",
        json.dumps(best, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Last 10 Epochs",
        "```json",
        json.dumps(last, indent=2, ensure_ascii=False),
        "```",
    ]
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
