#!/usr/bin/env bash
# Smoke test for the full IMMArticulationRefiner forward pipeline:
#
#     init_hypotheses → [step] × 3 → get_f_cond
#
# Simulates 30 control steps; calls step() every N=10 steps (3 refiner updates),
# then calls get_f_cond() to produce the policy conditioning signal.
#
# RUN ON THE UBUNTU SERVER ONLY (needs the artiresidual conda env w/ torch).
#
# Usage:
#   bash scripts/smoke_test_imm_pipeline.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --- preflight ----------------------------------------------------------------
if ! python -c "import torch" >/dev/null 2>&1; then
    echo "[ERROR] torch not importable. Activate the artiresidual conda env first." >&2
    exit 1
fi

# --- inline Python pipeline test ----------------------------------------------
python - <<'PYEOF'
import math
import sys
import torch
from omegaconf import OmegaConf

from artiresidual.refiner.imm_filter import IMMArticulationRefiner

PASS = "✓"
FAIL = "✗"

def check(cond: bool, msg: str) -> None:
    sym = PASS if cond else FAIL
    print(f"  {sym} {msg}")
    if not cond:
        sys.exit(1)

# ── Build model ──────────────────────────────────────────────────────────────
cfg = OmegaConf.create({
    "K": 3, "window_T": 30, "update_interval_N": 10,
    "dim": 256, "n_heads": 4, "n_layers": 4,
    "w_min": 0.05, "lambda_H": 0.01,
    "eta": 0.5, "omega_clip_deg": 30.0, "p_clip_m": 0.05,
})
model = IMMArticulationRefiner(cfg)
model.eval()

B, K, T, N = 2, cfg.K, cfg.window_T, 64
n_control_steps = 30
n_interval = cfg.update_interval_N

print()
print("=" * 64)
print("  IMMArticulationRefiner — full pipeline smoke test")
print("=" * 64)
print(f"  B={B}  K={K}  T={T}  N={N}")
print(f"  Trajectory: {n_control_steps} control steps  |  step() every {n_interval}")
print()

# ── 1. init_hypotheses() ─────────────────────────────────────────────────────
print("[stage 1] init_hypotheses (from mock prior)")
prior_output = {
    "omega":      torch.randn(B, 3),
    "p":          torch.randn(B, 3) * 0.5,
    "joint_type": torch.tensor([0, 1], dtype=torch.long),  # revolute, prismatic
    "confidence": torch.tensor([0.8, 0.7]),
}
hyp = model.init_hypotheses(prior_output)

check(hyp["omega_k"].shape == (B, K, 3),  f"omega_k shape  {tuple(hyp['omega_k'].shape)}")
check(hyp["p_k"].shape     == (B, K, 3),  f"p_k     shape  {tuple(hyp['p_k'].shape)}")
check(hyp["type_k"].shape  == (B, K),     f"type_k  shape  {tuple(hyp['type_k'].shape)}")
check(hyp["w_k"].shape     == (B, K),     f"w_k     shape  {tuple(hyp['w_k'].shape)}")
check(abs(hyp["w_k"][0, 0].item() - 0.6) < 1e-5, "revolute prior → h1 weight = 0.6")
check(abs(hyp["w_k"][1, 2].item() - 0.6) < 1e-5, "prismatic prior → h3 weight = 0.6")

# ── 2. Simulate 30 control steps with step() every 10 ────────────────────────
print(f"\n[stage 2] 30-step trajectory with step() every {n_interval} steps")
n_step_calls = 0
weight_trace = [hyp["w_k"].clone()]

for t in range(n_control_steps):
    if (t + 1) % n_interval == 0:
        # Build dummy evidence window
        window = {
            "delta_flow_k": torch.randn(B, K, T, N, 3),
            "wrench":       torch.randn(B, T, 12),
            "wrench_res_k": torch.randn(B, K, T, 3),
            "action":       torch.randn(B, T, 14),
        }
        with torch.no_grad():
            hyp = model.step(hyp, window)
        n_step_calls += 1
        weight_trace.append(hyp["w_k"].clone())

        # Quick invariants per step
        norms_ok = (hyp["omega_k"].norm(dim=-1) - 1.0).abs().max().item() < 1e-4
        sum_ok   = (hyp["w_k"].sum(dim=-1) - 1.0).abs().max().item() < 1e-4
        floor_ok = hyp["w_k"].min().item() >= cfg.w_min - 1e-5
        print(
            f"   step #{n_step_calls} after control step {t+1:2d}: "
            f"ω unit={'✓' if norms_ok else '✗'}  "
            f"Σw=1 {'✓' if sum_ok else '✗'}  "
            f"floor {'✓' if floor_ok else '✗'}  "
            f"w[sample 0]=[{', '.join(f'{x:.3f}' for x in hyp['w_k'][0].tolist())}]"
        )
        check(norms_ok and sum_ok and floor_ok, f"invariants hold after step #{n_step_calls}")

check(n_step_calls == 3, f"called step() exactly 3 times ({n_step_calls})")

# ── 3. Final get_f_cond() ─────────────────────────────────────────────────────
print("\n[stage 3] get_f_cond — belief-weighted flow + entropy")
theta_t = torch.rand(B) * math.pi
pcd     = torch.randn(B, N, 3)

with torch.no_grad():
    f_cond, entropy = model.get_f_cond(hyp, theta_t, pcd)

check(f_cond.shape  == (B, N, 3), f"f_cond  shape  {tuple(f_cond.shape)}")
check(entropy.shape == (B,),      f"entropy shape  {tuple(entropy.shape)}")

check(
    (not torch.isnan(f_cond).any()) and (not torch.isinf(f_cond).any()),
    "f_cond contains no NaN/Inf",
)
check(
    (not torch.isnan(entropy).any()) and (not torch.isinf(entropy).any()),
    "entropy contains no NaN/Inf",
)
log_K = math.log(K)
check((entropy >= 0).all().item(), "entropy ≥ 0 for all samples")
check((entropy <= log_K + 1e-5).all().item(),
      f"entropy ≤ log(K)={log_K:.4f} for all samples")

# ── Weight evolution summary ──────────────────────────────────────────────────
print("\n[trace] hypothesis weight evolution (sample 0)")
print("   stage            h1      h2      h3      entropy")
print("   ──────────────  ──────  ──────  ──────  ────────")
for i, w in enumerate(weight_trace):
    stage = "init" if i == 0 else f"after step #{i}"
    H = -(w[0].clamp(min=1e-8) * w[0].clamp(min=1e-8).log()).sum().item()
    print(
        f"   {stage:<14}  {w[0,0]:.4f}  {w[0,1]:.4f}  {w[0,2]:.4f}  "
        f"{H:.4f}"
    )

print()
print("=" * 64)
print("  Full pipeline smoke test PASSED")
print("=" * 64)
print()
PYEOF
