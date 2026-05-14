#!/usr/bin/env bash
# Smoke test for the Stage-1 IMMArticulationRefiner training loop.
#
# Quick mode (default): 100 steps on CPU, verifies loss decreasing.
# Full  mode (--full):  100 k steps, verifies top1 accuracy ≥ 0.70.
#
# RUN ON THE UBUNTU SERVER ONLY (needs the artiresidual conda env w/ torch).
#
# Usage:
#   bash scripts/smoke_test_train_refiner.sh            # quick
#   bash scripts/smoke_test_train_refiner.sh --full     # full training run
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --- preflight ----------------------------------------------------------------
if ! python -c "import torch" >/dev/null 2>&1; then
    echo "[ERROR] torch not importable. Activate the artiresidual conda env first." >&2
    exit 1
fi

if ! python -c "import omegaconf" >/dev/null 2>&1; then
    echo "[ERROR] omegaconf not importable. Run: pip install omegaconf" >&2
    exit 1
fi

# --- run smoke test -----------------------------------------------------------
FULL_FLAG=""
if [[ "${1:-}" == "--full" ]]; then
    FULL_FLAG="--full"
    echo "[INFO] Running FULL training smoke test (100 k steps). This may take a while."
else
    echo "[INFO] Running quick smoke test (100 steps on CPU)."
fi

python scripts/smoke_test_train_refiner.py $FULL_FLAG
