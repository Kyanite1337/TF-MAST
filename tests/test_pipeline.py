from pathlib import Path

from tfmast.pipeline import run_pipeline


def test_pipeline_runs_three_stages_and_writes_latest_feedback(tmp_path):
    result = run_pipeline(
        config_path=None,
        overrides={
            "paths.runs_dir": str(tmp_path / "runs"),
            "paths.checkpoints_dir": str(tmp_path / "checkpoints"),
            "wandb.mode": "disabled",
            "train.mae.epochs": 1,
            "train.tfc.epochs": 1,
            "train.finetune.epochs": 1,
            "train.mae.batch_size": 8,
            "train.tfc.batch_size": 8,
            "train.finetune.batch_size": 8,
            "train.max_batches": 1,
            "model.embed_dim": 32,
            "model.depths": [1, 1],
            "model.num_heads": [2, 4],
        },
        synthetic=True,
        experiment="pipeline_smoke",
        head="mlp",
    )

    assert result["mae"].best_checkpoint.exists()
    assert result["tfc"].best_checkpoint.exists()
    assert result["finetune"].best_checkpoint.exists()
    assert result["tfc"].run_dir.exists()
    assert (Path(tmp_path) / "runs" / "latest_feedback.md").exists()


def test_pipeline_prints_stage_progress_for_long_runs(tmp_path, capsys):
    run_pipeline(
        config_path=None,
        overrides={
            "paths.runs_dir": str(tmp_path / "runs"),
            "wandb.mode": "disabled",
            "train.mae.epochs": 1,
            "train.tfc.epochs": 1,
            "train.finetune.epochs": 1,
            "train.mae.batch_size": 8,
            "train.tfc.batch_size": 8,
            "train.finetune.batch_size": 8,
            "train.max_batches": 1,
            "model.embed_dim": 32,
            "model.depths": [1, 1],
            "model.num_heads": [2, 4],
        },
        synthetic=True,
        experiment="progress_smoke",
        head="mlp",
    )

    output = capsys.readouterr().out
    assert "[Pipeline] Preparing data" in output
    assert "[Pipeline] Starting stage: MAE" in output
    assert "[Pipeline] Starting stage: TFC" in output
    assert "[Pipeline] Starting stage: fine-tune" in output
    assert "step 1/" not in output
    assert "time=" in output
