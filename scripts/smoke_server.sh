#!/usr/bin/env bash
set -euo pipefail

python -m tfmast.env_check
python -m tfmast.pipeline --synthetic --experiment smoke_pipeline --head mlp --max-batches 2 wandb.mode=offline train.mae.epochs=1 train.tfc.epochs=1 train.finetune.epochs=1 model.embed_dim=32

if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("mamba_ssm") else 1)
PY
then
  python -m tfmast.pipeline --synthetic --experiment smoke_pipeline_mamba --head mamba --max-batches 2 wandb.mode=offline train.mae.epochs=1 train.tfc.epochs=1 train.finetune.epochs=1 model.embed_dim=32
  python -m tfmast.pipeline --synthetic --experiment smoke_pipeline_bimamba --head bimamba --max-batches 2 wandb.mode=offline train.mae.epochs=1 train.tfc.epochs=1 train.finetune.epochs=1 model.embed_dim=32
else
  echo "mamba_ssm not available; skipped Mamba/BiMamba smoke tests"
fi

if [ -d data/ninapro_db5/s1 ]; then
  python -m tfmast.pipeline --limit-subjects 1 --experiment smoke_real --head mlp --max-batches 2 wandb.mode=offline train.mae.epochs=1 train.tfc.epochs=1 train.finetune.epochs=1 model.embed_dim=32
else
  echo "data/ninapro_db5/s1 not found; skipped real-data dry-run"
fi
