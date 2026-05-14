"""Visualization of IMMArticulationRefiner.step() invariants.

Runs N_STEPS consecutive step() calls with random (untrained) weights and
produces a 4-panel PNG showing that the hard constraints are respected at
every step, regardless of network output.

Panels:
    1. Hypothesis weight evolution     — bar chart per step (stacked)
    2. ω axis correction magnitude     — geodesic angle per (k, step), vs clip bound
    3. p correction magnitude          — ‖Δp‖ per (k, step), vs clip bound
    4. ω unit-norm deviation           — ‖ωₖ‖ − 1 per (k, step), should be < 1e-5

Usage (server only, needs artiresidual conda env + matplotlib):
    python scripts/visualize_imm_step.py [--out PATH] [--steps N] [--seed S]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add repo root to path so the script can run from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artiresidual.refiner.analytical_flow import JOINT_TYPE_PRISMATIC
from artiresidual.refiner.imm_filter import IMMArticulationRefiner

_DEG = math.pi / 180.0


def _make_hypotheses(B: int, K: int) -> dict:
    omega_raw = torch.randn(B, K, 3)
    omega_k = omega_raw / omega_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    type_k = torch.zeros(B, K, dtype=torch.long)
    type_k[:, 2] = JOINT_TYPE_PRISMATIC
    return {
        "omega_k": omega_k,
        "p_k": torch.randn(B, K, 3) * 0.3,
        "type_k": type_k,
        "w_k": torch.full((B, K), 1.0 / K),
    }


def _make_window(B: int, K: int, T: int, N: int) -> dict:
    return {
        "delta_flow_k": torch.randn(B, K, T, N, 3),
        "wrench": torch.randn(B, T, 12),
        "wrench_res_k": torch.randn(B, K, T, 3),
        "action": torch.randn(B, T, 14),
    }


def main(out_path: Path, n_steps: int, seed: int) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
    except ImportError:
        print("[ERROR] matplotlib not installed. pip install matplotlib")
        sys.exit(1)

    torch.manual_seed(seed)

    cfg = OmegaConf.create(
        {
            "K": 3, "window_T": 30, "update_interval_N": 10,
            "dim": 256, "n_heads": 4, "n_layers": 4,
            "w_min": 0.05, "lambda_H": 0.01,
            "eta": 0.5, "omega_clip_deg": 30.0, "p_clip_m": 0.05,
        }
    )
    model = IMMArticulationRefiner(cfg)
    model.eval()
    print(
        f"Model: {sum(p.numel() for p in model.parameters()):,} params  |  "
        f"K={cfg.K}  T={cfg.window_T}  dim={cfg.dim}"
    )

    B, K, T, N = 1, cfg.K, cfg.window_T, 64

    # ── Run n_steps consecutive steps, record stats ───────────────────────────
    hyp = _make_hypotheses(B, K)

    weights_hist   = []  # list of [K] arrays
    angle_hist     = []  # list of [K] arrays (rad)
    dp_hist        = []  # list of [K] arrays (m)
    norm_dev_hist  = []  # list of [K] arrays

    omega_prev = hyp["omega_k"].clone()
    p_prev = hyp["p_k"].clone()

    with torch.no_grad():
        for step_i in range(n_steps):
            win = _make_window(B, K, T, N)
            hyp = model.step(hyp, win)

            # Weights [K]
            weights_hist.append(hyp["w_k"][0].numpy().copy())

            # ω correction: geodesic angle between omega_prev and omega_new
            dot = (omega_prev * hyp["omega_k"]).sum(dim=-1).clamp(-1, 1)  # [B,K]
            angle = torch.acos(dot)[0].numpy().copy()  # [K]
            angle_hist.append(angle)
            omega_prev = hyp["omega_k"].clone()

            # p correction magnitude [K]
            dp = (hyp["p_k"] - p_prev).norm(dim=-1)[0].numpy().copy()
            dp_hist.append(dp)
            p_prev = hyp["p_k"].clone()

            # ω unit-norm deviation [K]
            nd = (hyp["omega_k"].norm(dim=-1) - 1.0).abs()[0].numpy().copy()
            norm_dev_hist.append(nd)

    steps = list(range(1, n_steps + 1))
    weights_arr = np.array(weights_hist)    # [n_steps, K]
    angle_arr   = np.array(angle_hist)      # [n_steps, K]
    dp_arr      = np.array(dp_hist)         # [n_steps, K]
    norm_dev_arr = np.array(norm_dev_hist)  # [n_steps, K]

    max_angle_allowed = cfg.eta * cfg.omega_clip_deg * _DEG  # 0.5 * 30° rad
    max_dp_allowed    = cfg.eta * cfg.p_clip_m               # 0.5 * 5 cm

    hyp_names  = ["h1 rev-vert", "h2 rev-horiz", "h3 prismatic"]
    colors     = ["#4C72B0", "#DD8452", "#55A868"]

    # ── 4-panel figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"IMMArticulationRefiner.step() — {n_steps} steps  "
        f"(random/untrained weights, seed={seed})",
        fontsize=12,
    )

    # Panel 1: Hypothesis weights
    ax = axes[0, 0]
    bottom = np.zeros(n_steps)
    for k in range(K):
        ax.bar(steps, weights_arr[:, k], bottom=bottom,
               label=hyp_names[k], color=colors[k], alpha=0.85)
        bottom += weights_arr[:, k]
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="sum = 1")
    ax.axhline(cfg.w_min, color="red", linewidth=0.8, linestyle=":",
               label=f"w_min = {cfg.w_min}")
    ax.set_xlabel("Step"); ax.set_ylabel("Weight"); ax.set_title("Hypothesis weights")
    ax.set_ylim(0, 1.15); ax.legend(fontsize=8)

    # Panel 2: ω correction angle
    ax = axes[0, 1]
    for k in range(K):
        ax.plot(steps, np.degrees(angle_arr[:, k]),
                marker="o", markersize=3, color=colors[k], label=hyp_names[k])
    ax.axhline(math.degrees(max_angle_allowed), color="red", linestyle="--",
               linewidth=1.2, label=f"η·clip = {math.degrees(max_angle_allowed):.1f}°")
    ax.set_xlabel("Step"); ax.set_ylabel("Angle (°)")
    ax.set_title("ω correction (geodesic angle) — must stay ≤ η·30°")
    ax.legend(fontsize=8)

    # Panel 3: p correction magnitude
    ax = axes[1, 0]
    for k in range(K):
        ax.plot(steps, dp_arr[:, k] * 100,
                marker="s", markersize=3, color=colors[k], label=hyp_names[k])
    ax.axhline(max_dp_allowed * 100, color="red", linestyle="--",
               linewidth=1.2, label=f"η·clip = {max_dp_allowed*100:.1f} cm")
    ax.set_xlabel("Step"); ax.set_ylabel("‖Δp‖ (cm)")
    ax.set_title("p correction magnitude — must stay ≤ η·5 cm")
    ax.legend(fontsize=8)

    # Panel 4: ω unit-norm deviation
    ax = axes[1, 1]
    for k in range(K):
        ax.semilogy(steps, np.maximum(norm_dev_arr[:, k], 1e-12),
                    marker="^", markersize=3, color=colors[k], label=hyp_names[k])
    ax.axhline(1e-5, color="red", linestyle="--", linewidth=1.0, label="tol = 1e-5")
    ax.set_xlabel("Step"); ax.set_ylabel("|‖ωₖ‖ − 1|  (log scale)")
    ax.set_title("ω unit-norm deviation — must stay < 1e-5")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    # Print summary table
    print()
    print("  Step  | max Δω (°) | max Δp (cm) | max |‖ω‖-1|")
    print("  ------+------------+-------------+---------------")
    for i in range(n_steps):
        print(
            f"  {i+1:4d}  | "
            f"{math.degrees(angle_arr[i].max()):10.4f} | "
            f"{dp_arr[i].max()*100:11.4f} | "
            f"{norm_dev_arr[i].max():.2e}"
        )

    # Verify constraints hold
    clip_ok    = (angle_arr <= max_angle_allowed + 1e-5).all()
    dp_ok      = (dp_arr    <= max_dp_allowed    + 1e-6).all()
    norm_ok    = (norm_dev_arr < 1e-4).all()
    weight_ok  = (abs(weights_arr.sum(axis=-1) - 1.0) < 1e-4).all()
    floor_ok   = (weights_arr >= cfg.w_min - 1e-5).all()

    print()
    results = [
        ("ω clip ≤ η·30°", clip_ok),
        ("p clip ≤ η·5cm", dp_ok),
        ("ω unit norm",     norm_ok),
        ("weights sum=1",   weight_ok),
        ("weights ≥ w_min", floor_ok),
    ]
    all_pass = True
    for name, ok in results:
        sym = "✓" if ok else "✗"
        print(f"  {sym}  {name}")
        all_pass = all_pass and ok

    print()
    if all_pass:
        print("All constraints satisfied across all steps.")
    else:
        print("[FAIL] One or more constraints violated — check output above.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent.parent / "visualizations" / "imm_step.png",
        help="Output PNG path (default: visualizations/imm_step.png)",
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of step() calls")
    parser.add_argument("--seed",  type=int, default=42,  help="Random seed")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    main(args.out, args.steps, args.seed)
