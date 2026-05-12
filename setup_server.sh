#!/usr/bin/env bash
# One-click conda environment setup for the ArtiResidual training/sim server.
#
# Run this ON THE UBUNTU SERVER (not in the GitHub codespace):
#
#   bash setup_server.sh
#
# What it does:
#   1. Creates a conda env "$ENV_NAME" (default: artiresidual) with Python 3.10.
#   2. Installs PyTorch built against CUDA 12.1.
#   3. Installs this package editable with [train, sim, dev] extras.
#   4. Verifies torch sees the GPU.
#
# Prereqs on the server:
#   - Ubuntu 22.04 (or compatible)
#   - NVIDIA driver supporting CUDA 12.1 (verify: nvidia-smi)
#   - conda or mamba on PATH
#
# Env vars you can override:
#   ENV_NAME            (default: artiresidual)
#   PY_VER              (default: 3.10)
#   CUDA_TAG            (default: cu121)
#   TORCH_VER           (default: 2.4.0)
#   TORCHVISION_VER     (default: 0.19.0)
#
# NOTE: RoboTwin 2.0 and ManiSkill 3 are NOT installed here — they have their
# own non-pip setup steps. See scripts/setup_robotwin.sh once implemented.

set -euo pipefail

ENV_NAME="${ENV_NAME:-artiresidual}"
PY_VER="${PY_VER:-3.10}"
CUDA_TAG="${CUDA_TAG:-cu121}"
TORCH_VER="${TORCH_VER:-2.4.0}"
TORCHVISION_VER="${TORCHVISION_VER:-0.19.0}"

# ---- preflight ----
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found on PATH. Install miniconda first:" >&2
    echo "        https://docs.anaconda.com/miniconda/install/" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[WARN] nvidia-smi not found; CUDA runtime will fail at import time."
fi

# Make `conda activate` work in a non-interactive shell.
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---- env ----
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[INFO] conda env '${ENV_NAME}' already exists; reusing it."
else
    echo "[INFO] creating conda env '${ENV_NAME}' with python ${PY_VER}..."
    conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi
conda activate "${ENV_NAME}"

# ---- pip & torch ----
python -m pip install --upgrade pip wheel

echo "[INFO] installing torch ${TORCH_VER} (${CUDA_TAG})..."
pip install \
    "torch==${TORCH_VER}" \
    "torchvision==${TORCHVISION_VER}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# ---- this package ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[INFO] installing artiresidual (editable) with [train,sim,dev] extras..."
pip install -e "${REPO_ROOT}[train,sim,dev]"

# ---- sanity ----
python - <<'PY'
import torch
print(f"[OK] torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[OK]   device {i}: {torch.cuda.get_device_name(i)}")
PY

cat <<EOF

==============================================================
[OK] conda env '${ENV_NAME}' is ready.

Next steps on this server:
  conda activate ${ENV_NAME}
  wandb login                          # one-time
  # then run whatever Claude printed, e.g.:
  # python -m artiresidual.training.train_refiner --config-name=base

Not installed by this script (do these separately):
  - RoboTwin 2.0:  https://github.com/RoboTwin-Platform/RoboTwin
  - ManiSkill 3:   https://github.com/haosulab/ManiSkill
  - cuRobo:        https://github.com/NVlabs/curobo
==============================================================
EOF
