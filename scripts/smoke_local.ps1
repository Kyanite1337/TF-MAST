$ErrorActionPreference = "Stop"
python -m pytest -q
python -m tfmast.smoke --wandb-mode disabled --head mlp
