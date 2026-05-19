from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tfmast.config import load_config


def build_suite(name: str = "full") -> list[dict]:
    if name != "full":
        raise ValueError(f"Unknown ablation suite: {name}")
    return [
        {"name": "A0_random_finetune", "stages": ["finetune"], "init": None},
        {"name": "A1_mae_finetune", "stages": ["mae", "finetune"]},
        {"name": "A2_tfc_finetune", "stages": ["tfc", "finetune"]},
        {"name": "A3_mae_tfc_finetune", "stages": ["mae", "tfc", "finetune"]},
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="full")
    parser.add_argument("--config", default="configs/db5.yaml")
    parser.add_argument("--output", type=Path, default=Path("runs/ablation_plan"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    suite = build_suite(args.suite)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "suite.json").write_text(json.dumps(suite, indent=2), encoding="utf-8")
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "stages"])
        writer.writeheader()
        for item in suite:
            writer.writerow({"name": item["name"], "stages": "->".join(item["stages"])})
    (args.output / "summary.md").write_text("\n".join(["# Ablation Suite", "", *[f"- {i['name']}: {' -> '.join(i['stages'])}" for i in suite]]), encoding="utf-8")
    print(f"Wrote ablation suite scaffold to {args.output}")


if __name__ == "__main__":
    main()
