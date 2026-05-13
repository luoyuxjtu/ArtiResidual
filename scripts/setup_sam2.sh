#!/usr/bin/env bash
# Install SAM2 + open3d on the GPU server. RUN ON THE UBUNTU SERVER ONLY.
#
# Prerequisites:
#   - conda env "artiresidual" already created (run setup_server.sh first).
#   - torch with CUDA already installed in that env.
#
# What this script does:
#   1. git-clones facebookresearch/sam2 into $SAM2_DIR (default
#      $HOME/third_party/sam2). Override with `SAM2_DIR=/path bash ...`.
#   2. pip-installs SAM2 in editable mode from that directory.
#   3. Downloads the sam2_hiera_base_plus checkpoint (~80 MB).
#   4. pip-installs open3d (used inside SAM2PartTracker for ICP).
#   5. Imports both to make sure the install works.
#
# After this script: the smoke test is
#   python scripts/smoke_test_part_tracker.py \
#       --sam2-checkpoint $SAM2_DIR/checkpoints/sam2_hiera_base_plus.pt
#
# Env vars you can override:
#   SAM2_DIR             default $HOME/third_party/sam2
#   SAM2_CKPT_VARIANT    default "sam2_hiera_base_plus" (alternatives:
#                        sam2_hiera_tiny / small / large, or any of the
#                        sam2.1_* variants — see SAM2 README).
#   SAM2_CKPT_DATE       default 072824 (original SAM2 release; for
#                        sam2.1_* variants pass 092824).

set -euo pipefail

SAM2_DIR="${SAM2_DIR:-$HOME/third_party/sam2}"
SAM2_CKPT_VARIANT="${SAM2_CKPT_VARIANT:-sam2_hiera_base_plus}"
SAM2_CKPT_DATE="${SAM2_CKPT_DATE:-072824}"
SAM2_CKPT_NAME="${SAM2_CKPT_VARIANT}.pt"
SAM2_CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/${SAM2_CKPT_DATE}/${SAM2_CKPT_NAME}"

# ---- preflight ------------------------------------------------------------
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "${CONDA_DEFAULT_ENV}" != "artiresidual" ]]; then
    echo "[ERROR] activate the artiresidual conda env first:" >&2
    echo "          conda activate artiresidual" >&2
    exit 1
fi
if ! python -c "import torch" >/dev/null 2>&1; then
    echo "[ERROR] torch not importable. Run setup_server.sh first." >&2
    exit 1
fi
if ! command -v git >/dev/null 2>&1; then
    echo "[ERROR] git not found." >&2
    exit 1
fi

# ---- 1. clone SAM2 --------------------------------------------------------
mkdir -p "$(dirname "${SAM2_DIR}")"
if [[ -d "${SAM2_DIR}/.git" ]]; then
    echo "[INFO] SAM2 already cloned at ${SAM2_DIR}; pulling latest"
    git -C "${SAM2_DIR}" pull --ff-only
else
    echo "[INFO] cloning SAM2 to ${SAM2_DIR}"
    git clone https://github.com/facebookresearch/sam2.git "${SAM2_DIR}"
fi

# ---- 2. pip install SAM2 (editable) ---------------------------------------
echo "[INFO] installing SAM2 (pip install -e ${SAM2_DIR})..."
python -m pip install --upgrade pip wheel
# Install SAM2 without letting it upgrade torch/torchvision: SAM2 declares
# torch>=2.5.1 but works fine with 2.4.x+cu121. Installing with --no-deps
# avoids overwriting the cu121 build that matches the server's CUDA 12.x driver.
# We then install SAM2's other deps (iopath, hydra-core, etc.) explicitly.
pip install --no-deps -e "${SAM2_DIR}"
pip install "iopath>=0.1.10" "hydra-core>=1.3.2" "tqdm>=4.66.1"

# ---- 3. download checkpoint -----------------------------------------------
mkdir -p "${SAM2_DIR}/checkpoints"
CKPT_PATH="${SAM2_DIR}/checkpoints/${SAM2_CKPT_NAME}"
if [[ -f "${CKPT_PATH}" ]]; then
    SIZE=$(stat -c %s "${CKPT_PATH}" 2>/dev/null || stat -f %z "${CKPT_PATH}")
    echo "[INFO] checkpoint already exists at ${CKPT_PATH} (${SIZE} bytes)"
else
    echo "[INFO] downloading ${SAM2_CKPT_NAME} from ${SAM2_CKPT_URL} ..."
    if command -v wget >/dev/null 2>&1; then
        wget --show-progress -O "${CKPT_PATH}.tmp" "${SAM2_CKPT_URL}"
    elif command -v curl >/dev/null 2>&1; then
        curl -L --progress-bar -o "${CKPT_PATH}.tmp" "${SAM2_CKPT_URL}"
    else
        echo "[ERROR] need wget or curl on PATH" >&2
        exit 1
    fi
    mv "${CKPT_PATH}.tmp" "${CKPT_PATH}"
fi

# ---- 4. install open3d ----------------------------------------------------
echo "[INFO] installing open3d (for ICP)..."
pip install --upgrade "open3d>=0.18"

# ---- 5. import sanity -----------------------------------------------------
python - <<PY
import importlib, sys

ok = True
for name in ("sam2", "open3d", "PIL", "numpy"):
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", "installed")
        print(f"[OK] {name}: {ver}")
    except Exception as e:
        print(f"[FAIL] cannot import {name}: {e}")
        ok = False
sys.exit(0 if ok else 1)
PY

cat <<EOF

==============================================================
[OK] SAM2 + open3d installed.

Checkpoint:  ${CKPT_PATH}

Smoke test:
  python scripts/smoke_test_part_tracker.py \\
      --sam2-checkpoint ${CKPT_PATH}

Visualization (writes a PNG):
  python scripts/visualize_part_tracker.py \\
      --sam2-checkpoint ${CKPT_PATH} \\
      --out /tmp/part_tracker_demo.png

Set in your shell profile so other scripts find the checkpoint:
  export SAM2_CHECKPOINT=${CKPT_PATH}
==============================================================
EOF
