#!/usr/bin/env bash
# Run the pytest suite on the GPU server.
#
# RUN ON THE UBUNTU SERVER ONLY (needs the artiresidual conda env w/ torch).
# In the GitHub codespace `torch` is not installed; the tests cannot run there.
#
# Usage:
#   bash scripts/run_tests.sh                              # everything
#   bash scripts/run_tests.sh tests/test_state_estimator.py  # one file
#   bash scripts/run_tests.sh tests/test_state_estimator.py -k revolute
#   bash scripts/run_tests.sh -x --tb=short                # stop on first fail
#
# Any args you pass after the script name are forwarded directly to pytest.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --- preflight --------------------------------------------------------------
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] \
   || [[ "${CONDA_DEFAULT_ENV}" != "artiresidual" ]]; then
    echo "[WARN] expected conda env 'artiresidual' to be active."
    echo "       current env: '${CONDA_DEFAULT_ENV:-<none>}'"
    echo "       to activate: conda activate artiresidual"
fi

if ! python -c "import torch" >/dev/null 2>&1; then
    echo "[ERROR] torch not importable. Run setup_server.sh first." >&2
    exit 1
fi

# --- run --------------------------------------------------------------------
exec pytest -v "$@"
