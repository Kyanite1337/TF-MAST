#!/usr/bin/env bash
set -euo pipefail

python -m tfmast.env_check
python -m tfmast.smoke --wandb-mode offline --head mlp

if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("mamba_ssm") else 1)
PY
then
  python -m tfmast.smoke --wandb-mode offline --head mamba
  python -m tfmast.smoke --wandb-mode offline --head bimamba
else
  echo "mamba_ssm not available; skipped Mamba/BiMamba smoke tests"
fi

if [ -d data/ninapro_db5/s1 ]; then
  python -m tfmast.smoke --real-data --limit-subjects 1 --wandb-mode offline --head mlp
else
  echo "data/ninapro_db5/s1 not found; skipped real-data dry-run"
fi
