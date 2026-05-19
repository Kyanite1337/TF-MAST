#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip

# Pick the CUDA wheel index that matches your server driver/CUDA stack.
# For recent RTX PRO 6000 servers, cu124 is a conservative default.
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
python -m pip install -r requirements-server.txt
python -m pip install -e .

python -m tfmast.env_check
