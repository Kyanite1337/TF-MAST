import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_raw_bypass_result_recorder_writes_gap_and_per_class_summary(tmp_path):
    run_dir = tmp_path / "runs" / "20260525_finetune_rawb_test"
    run_dir.mkdir(parents=True)
    config = {
        "data": {
            "window_ms": 300,
            "stride_ms": 100,
            "class_mode": "53_with_rest",
            "train_reps": [1, 3, 4, 6],
            "test_reps": [2, 5],
        },
        "preprocess": {"demean": True, "zscore": True, "notch": {"enabled": True, "freq": 50.0}, "bandpass": {"enabled": False}},
        "model": {"embed_dim": 128, "depths": [2, 2, 2], "num_heads": [4, 4, 8], "patch_size": [1, 4]},
        "head": {"dropout": 0.3, "mamba_d_state": 16, "mamba_d_conv": 4, "mamba_expand": 2},
        "train": {
            "seed": 42,
            "amp": True,
            "num_workers": 0,
            "finetune": {
                "loss": "cross_entropy",
                "class_weight": "none",
                "label_smoothing": 0.1,
                "lr": 0.001,
                "weight_decay": 0.05,
                "batch_size": 512,
                "epochs": 150,
            },
        },
    }
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_metrics": {
                    "epoch": 7,
                    "finetune/accuracy": 0.9,
                    "finetune/macro_f1": 0.8,
                    "finetune/val_loss": 0.4,
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "test_metrics.json").write_text(
        json.dumps(
            {
                "test/loss": 0.6,
                "test/accuracy": 0.7,
                "test/macro_f1": 0.5,
                "test/confusion_matrix": [[8, 2, 0], [1, 5, 4], [0, 6, 4]],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "results.csv"

    subprocess.run(
        [
            sys.executable,
            "runs/append_raw_bypass_ablation_result.py",
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
            "--xlsx",
            str(tmp_path / "results.xlsx"),
            "--experiment",
            "rawb_test",
            "--pretrain",
            "mae_tfc",
            "--head",
            "bimamba",
            "--bypass",
            "true",
            "--init-checkpoint",
            "runs/tfc/best.pt",
            "--status",
            "done",
            "--started-at",
            "2026-05-25T00:00:00+00:00",
            "--finished-at",
            "2026-05-25T00:01:00+00:00",
            "--duration-sec",
            "60",
        ],
        check=True,
    )

    row = next(csv.DictReader(output.open()))
    assert row["gap_acc"] == "0.2"
    assert row["gap_macro_f1"] == "0.3"
    assert row["raw_bypass_type"] == "token_add"
    assert row["worst_5_classes_by_f1"].startswith("1:")
    assert row["rest_class_f1"] == "0.8421052632"
    assert float(row["non_rest_macro_f1"]) < float(row["test_macro_f1"])
