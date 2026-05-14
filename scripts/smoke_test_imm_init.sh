#!/usr/bin/env bash
# Smoke test for IMMArticulationRefiner.__init__ + init_hypotheses().
#
# RUN ON THE UBUNTU SERVER ONLY (needs the artiresidual conda env w/ torch).
# In the GitHub codespace `torch` is not installed; this script cannot run there.
#
# Usage:
#   bash scripts/smoke_test_imm_init.sh
#
# Checks:
#   1. Model constructs without error.
#   2. Parameter count in target range 3–5 M.
#   3. Per-submodule parameter breakdown (model summary).
#   4. init_hypotheses() output shapes are correct.
#   5. Hypothesis weights sum to 1 and respect initialization rules.
#   6. Hypothesis axes are unit vectors.
#   7. Hypothesis types are [REVOLUTE, REVOLUTE, PRISMATIC].
#   8. Revolute-prior samples → h1 gets 0.6; prismatic-prior → h3 gets 0.6.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --- preflight ----------------------------------------------------------------
if ! python -c "import torch" >/dev/null 2>&1; then
    echo "[ERROR] torch not importable. Activate the artiresidual conda env first." >&2
    exit 1
fi

# --- run inline Python test ---------------------------------------------------
python - <<'PYEOF'
import sys
import torch
from omegaconf import OmegaConf

from artiresidual.refiner.imm_filter import IMMArticulationRefiner
from artiresidual.refiner.analytical_flow import JOINT_TYPE_REVOLUTE, JOINT_TYPE_PRISMATIC

PASS = "✓"
FAIL = "✗"

def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  {PASS} {msg}")
    else:
        print(f"  {FAIL} {msg}", file=sys.stderr)
        sys.exit(1)

# ── Build model (K=3, default config) ─────────────────────────────────────────
cfg = OmegaConf.create({
    "K": 3, "window_T": 30, "update_interval_N": 10,
    "dim": 256, "n_heads": 4, "n_layers": 4,
    "w_min": 0.05, "lambda_H": 0.01,
    "eta": 0.5, "omega_clip_deg": 30.0, "p_clip_m": 0.05,
})
model = IMMArticulationRefiner(cfg)
model.eval()

# ── 1. Parameter count ─────────────────────────────────────────────────────────
print()
print("=" * 62)
print("  IMMArticulationRefiner — smoke test")
print("=" * 62)

n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
TARGET_LO, TARGET_HI = 3_000_000, 5_000_000

print(f"\n[1] Parameter count")
print(f"    Total      : {n_total:>12,}")
print(f"    Trainable  : {n_train:>12,}")
print(f"    Target     : {TARGET_LO:,} – {TARGET_HI:,}")
check(TARGET_LO <= n_total <= TARGET_HI,
      f"Total params {n_total:,} in [3 M, 5 M]")

# ── 2. Per-submodule summary ───────────────────────────────────────────────────
print(f"\n[2] Model summary (sub-modules)")
print(f"    {'submodule':<28} {'params':>12}")
print(f"    {'-'*28} {'-'*12}")
for name, child in model.named_children():
    n = sum(p.numel() for p in child.parameters())
    print(f"    {name:<28} {n:>12,}")
print(f"    {'TOTAL':<28} {n_total:>12,}")

# ── 3. init_hypotheses shapes ─────────────────────────────────────────────────
print(f"\n[3] init_hypotheses() — shape tests")

B, K = 8, cfg.K  # mix of revolute and prismatic priors
# First 4 samples: revolute prior; last 4: prismatic
prior_output = {
    "omega":      torch.randn(B, 3),
    "p":          torch.randn(B, 3),
    "joint_type": torch.cat([
        torch.zeros(B // 2, dtype=torch.long),   # revolute
        torch.ones (B // 2, dtype=torch.long),   # prismatic
    ]),
    "confidence": torch.rand(B),
}

with torch.no_grad():
    hyp = model.init_hypotheses(prior_output)

expected_shapes = {
    "omega_k": (B, K, 3),
    "p_k":     (B, K, 3),
    "type_k":  (B, K),
    "w_k":     (B, K),
}
for key, exp_shape in expected_shapes.items():
    got = tuple(hyp[key].shape)
    check(got == exp_shape, f"{key}: shape {got} == {exp_shape}")

# ── 4. Weight sanity ───────────────────────────────────────────────────────────
print(f"\n[4] Hypothesis weights")
w = hyp["w_k"]

check(
    torch.allclose(w.sum(dim=-1), torch.ones(B), atol=1e-5),
    "w_k sums to 1.0 per sample",
)
check(
    (w >= 0.0).all().item(),
    "all weights ≥ 0",
)

# Revolute-prior samples (first B//2): h1 should get 0.6
rev_w = w[: B // 2]
check(
    torch.allclose(rev_w[:, 0], torch.full((B // 2,), 0.6), atol=1e-5),
    "revolute-prior → h1 weight = 0.6",
)
check(
    torch.allclose(rev_w[:, 1], torch.full((B // 2,), 0.2), atol=1e-5),
    "revolute-prior → h2 weight = 0.2",
)
check(
    torch.allclose(rev_w[:, 2], torch.full((B // 2,), 0.2), atol=1e-5),
    "revolute-prior → h3 weight = 0.2",
)

# Prismatic-prior samples (last B//2): h3 should get 0.6
pris_w = w[B // 2 :]
check(
    torch.allclose(pris_w[:, 2], torch.full((B // 2,), 0.6), atol=1e-5),
    "prismatic-prior → h3 weight = 0.6",
)
check(
    torch.allclose(pris_w[:, 0], torch.full((B // 2,), 0.2), atol=1e-5),
    "prismatic-prior → h1 weight = 0.2",
)

# ── 5. Axis unit-norm ─────────────────────────────────────────────────────────
print(f"\n[5] Hypothesis axes")
omega_k = hyp["omega_k"]
norms = omega_k.norm(dim=-1)  # [B, K]
check(
    torch.allclose(norms, torch.ones(B, K), atol=1e-5),
    f"all omega_k are unit vectors (max |norm-1| = {(norms - 1).abs().max().item():.2e})",
)

# h1 and h2 axes should be orthogonal to each other
dot_12 = (omega_k[:, 0, :] * omega_k[:, 1, :]).sum(dim=-1)  # [B]
check(
    dot_12.abs().max().item() < 1e-5,
    f"h1 ⊥ h2 for all samples (max |dot| = {dot_12.abs().max().item():.2e})",
)

# h1 and h3 axes should be identical
check(
    torch.allclose(omega_k[:, 0, :], omega_k[:, 2, :], atol=1e-6),
    "h1 and h3 share the same axis direction",
)

# ── 6. Hypothesis types ───────────────────────────────────────────────────────
print(f"\n[6] Hypothesis types")
type_k = hyp["type_k"]
check(
    (type_k[:, 0] == JOINT_TYPE_REVOLUTE).all().item(),
    "h1 type = REVOLUTE (0)",
)
check(
    (type_k[:, 1] == JOINT_TYPE_REVOLUTE).all().item(),
    "h2 type = REVOLUTE (0)",
)
check(
    (type_k[:, 2] == JOINT_TYPE_PRISMATIC).all().item(),
    "h3 type = PRISMATIC (1)",
)

# ── Edge case: omega ≈ z-axis (tests y_hat fallback) ──────────────────────────
print(f"\n[7] Edge case: omega ≈ z-axis (fallback to y_hat)")
prior_z = {
    "omega":      torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
    "p":          torch.zeros(2, 3),
    "joint_type": torch.zeros(2, dtype=torch.long),
    "confidence": torch.ones(2),
}
with torch.no_grad():
    hyp_z = model.init_hypotheses(prior_z)
norms_z = hyp_z["omega_k"].norm(dim=-1)
check(
    torch.allclose(norms_z, torch.ones(2, K), atol=1e-5),
    "omega_k unit even when prior omega = ±z",
)
dot_z = (hyp_z["omega_k"][:, 0, :] * hyp_z["omega_k"][:, 1, :]).sum(dim=-1)
check(
    dot_z.abs().max().item() < 1e-5,
    f"h1 ⊥ h2 when prior omega = ±z  (max |dot| = {dot_z.abs().max().item():.2e})",
)

# ── Done ──────────────────────────────────────────────────────────────────────
print()
print("=" * 62)
print("  All checks PASSED")
print("=" * 62)
print()
PYEOF
