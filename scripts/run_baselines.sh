#!/usr/bin/env bash
# Run every baseline on the multi-handle benchmark. SERVER ONLY.
#
# Baselines (spec §8):
#   - DP3                 https://github.com/YanjieZe/3D-Diffusion-Policy
#   - ACT                 https://github.com/tonyzhaozh/act
#   - RDT-1B fine-tuned   https://github.com/thu-ml/RoboticsDiffusionTransformer
#   - ArticuBot-bimanual  https://github.com/yufeiwang63/articubot
#   - Buchanan replication
#
# Each baseline produces eval JSON under outputs/baselines/<name>/.
set -euo pipefail
echo "[TODO] wire up after Stage-2 policy training works end-to-end."
exit 0
