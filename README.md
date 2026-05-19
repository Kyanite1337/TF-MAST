# TF-MAST for NinaPro DB5

Modular PyTorch project for staged self-supervised sEMG gesture recognition on NinaPro DB5:

1. MAE pretraining with MAST-style mask strategies.
2. TFC pretraining with a retained time encoder and a separate frequency encoder.
3. Supervised fine-tuning with `mlp`, official `mamba-ssm`, or BiMamba classifier heads.

## Data Layout

Place the dataset under the project root:

```text
data/ninapro_db5/
  s1/S1_E1_A1.mat
  s1/S1_E2_A1.mat
  s1/S1_E3_A1.mat
  ...
  s10/S10_E3_A1.mat
```

The default split is train repetitions `1,3,4,6` and test repetitions `2,5`.

## Local Test

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
python -m tfmast.smoke --wandb-mode disabled --head mlp
```

## Server Setup

On the Linux CUDA server:

```bash
bash scripts/install_server.sh
bash scripts/smoke_server.sh
```

`scripts/smoke_server.sh` checks CUDA, runs synthetic three-stage smoke tests, runs Mamba/BiMamba smoke tests when `mamba_ssm` is installed, and then runs a one-subject real-data dry-run if `data/ninapro_db5/s1` exists.

## Common Commands

```bash
python -m tfmast.env_check
python -m tfmast.preprocess --config configs/db5.yaml
python -m tfmast.train --config configs/db5.yaml stage=mae experiment=mae_swin
python -m tfmast.train --config configs/db5.yaml stage=tfc init=runs/<mae_run>/best.pt
python -m tfmast.train --config configs/db5.yaml stage=finetune init=runs/<tfc_run>/best.pt head=mamba
python -m tfmast.ablate --suite full
```

Every training run writes `best.pt`, `last.pt`, `metrics.jsonl`, `config_resolved.yaml`, and `feedback.md`. The latest feedback channel is:

```text
runs/latest_feedback.md
runs/latest_metrics.json
```

Send those files back for analysis and next-step tuning.
