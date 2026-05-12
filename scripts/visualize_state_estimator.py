"""Visualize Module 06 JointStateEstimator behavior.

CPU-only. Run on the server (or anywhere with torch + matplotlib):

    pip install matplotlib                 # if not already
    python scripts/visualize_state_estimator.py --out /tmp/state_est.png

Four diagnostic panels (one per row × column of a 2×2 grid):

    [0,0] Revolute sweep
            Plot θ_est vs θ_gt across (-π/2, ~+π). A perfect estimator gives
            the y = x diagonal; any deviation flags a sign / atan2 / wrap bug.

    [0,1] Prismatic sweep
            Same diagnostic for axial displacement across ±0.5 m. Spec says
            error ≤ 1 mm — the annotated max error reports the achieved bound.

    [1,0] K=3 hypotheses on a single observed rotation (30° about +z)
            Bars compare each hypothesis's θ_est. The correct hypothesis hits
            the dashed ground-truth line; the wrong-axis revolute and the
            wrong-type prismatic hypotheses collapse to ~0, which is what the
            IMM refiner relies on as a discriminating signal.

    [1,1] Prismatic noise histogram (σ = 0.5 cm position noise, B = 200)
            θ_est distribution. The annotated sample std should be ≈ σ = 5 mm
            (since projecting an isotropic Gaussian onto a unit axis preserves
            the per-component std). The mean should be close to the ground
            truth displacement (here 0.15 m).

Failures in any panel indicate a real bug — the in-panel annotations report
the achieved bounds and the spec requirements side by side.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from artiresidual.refiner.analytical_flow import (
    JOINT_TYPE_PRISMATIC,
    JOINT_TYPE_REVOLUTE,
)
from artiresidual.refiner.state_estimator import JointStateEstimator

_DEG = math.pi / 180.0


# ---------------------------------------------------------------------------
# Helpers (kept local — duplicate of the test-file helpers, but importing test
# code from a script feels wrong; this file is the visual companion).
# ---------------------------------------------------------------------------


def _axis_angle_to_quat(axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Axis-angle → scalar-first quaternion ``(w, x, y, z)``."""
    axis = axis / torch.linalg.norm(axis).clamp(min=1e-12)
    half = angle_rad / 2.0
    w = math.cos(half)
    s = math.sin(half)
    return torch.tensor(
        [w, s * axis[0].item(), s * axis[1].item(), s * axis[2].item()],
        dtype=torch.float32,
    )


def _identity_quat(B: int) -> torch.Tensor:
    return torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(B, 4).contiguous()


# ---------------------------------------------------------------------------
# Data-gathering routines for each panel.
# ---------------------------------------------------------------------------


def _revolute_sweep(n: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    """Sweep θ across (-π/2, +0.95π) and return (θ_gt, θ_est) as 1-D tensors."""
    omega = torch.tensor([0.0, 0.0, 1.0])
    p = torch.zeros(3)
    initial = torch.cat([torch.zeros(1, 3), _identity_quat(1)], dim=-1)

    # Avoid exactly ±π — the equivalence class q ↔ −q sits on a discontinuity
    # of atan2 (with wrap to ∓π); a real estimator hits it either way.
    theta_gt = torch.linspace(-math.pi / 2.0, math.pi * 0.95, n)
    theta_est = torch.empty_like(theta_gt)

    for i, theta in enumerate(theta_gt):
        q_rot = _axis_angle_to_quat(omega, float(theta)).unsqueeze(0)
        curr = torch.cat([torch.zeros(1, 3), q_rot], dim=-1)
        est = JointStateEstimator(omega, p, JOINT_TYPE_REVOLUTE, initial)
        theta_est[i] = est.estimate(curr).item()
    return theta_gt, theta_est


def _prismatic_sweep(n: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    """Sweep Δ across (-0.5 m, +0.5 m) and return (Δ_gt, Δ_est)."""
    omega = torch.tensor([1.0, 0.0, 0.0])
    p = torch.zeros(3)
    initial = torch.cat([torch.zeros(1, 3), _identity_quat(1)], dim=-1)

    delta_gt = torch.linspace(-0.5, 0.5, n)
    delta_est = torch.empty_like(delta_gt)

    for i, delta in enumerate(delta_gt):
        x_curr = float(delta) * omega.unsqueeze(0)
        curr = torch.cat([x_curr, _identity_quat(1)], dim=-1)
        est = JointStateEstimator(omega, p, JOINT_TYPE_PRISMATIC, initial)
        delta_est[i] = est.estimate(curr).item()
    return delta_gt, delta_est


def _k_hypothesis_demo(
    theta_gt_deg: float = 30.0,
) -> tuple[list[str], list[float], list[str], float]:
    """One observed motion (rotation about +z), three competing hypotheses.

    Returns:
        labels, estimates, units, theta_gt_rad.
    """
    theta_gt = theta_gt_deg * _DEG
    initial = torch.cat([torch.zeros(1, 3), _identity_quat(1)], dim=-1)
    q_rot = _axis_angle_to_quat(
        torch.tensor([0.0, 0.0, 1.0]), theta_gt
    ).unsqueeze(0)
    curr = torch.cat([torch.zeros(1, 3), q_rot], dim=-1)

    hypotheses = [
        ("rev +ẑ\n(correct)", torch.tensor([0.0, 0.0, 1.0]),
         JOINT_TYPE_REVOLUTE, "rad"),
        ("rev +ŷ\n(wrong axis)", torch.tensor([0.0, 1.0, 0.0]),
         JOINT_TYPE_REVOLUTE, "rad"),
        ("prism +x̂\n(wrong type)", torch.tensor([1.0, 0.0, 0.0]),
         JOINT_TYPE_PRISMATIC, "m"),
    ]

    labels: list[str] = []
    estimates: list[float] = []
    units: list[str] = []
    for label, omega, jtype, unit in hypotheses:
        est = JointStateEstimator(omega, torch.zeros(3), jtype, initial)
        estimates.append(est.estimate(curr).item())
        labels.append(label)
        units.append(unit)
    return labels, estimates, units, theta_gt


def _prismatic_noise_demo(
    B: int = 200, sigma_m: float = 0.005, delta_gt: float = 0.15
) -> tuple[torch.Tensor, float, float]:
    """Run B independent noisy prismatic estimates. Returns θ_est tensor [B]."""
    torch.manual_seed(42)
    omega = torch.tensor([1.0, 0.0, 0.0])
    p = torch.zeros(3)

    initial = torch.cat([torch.zeros(B, 3), _identity_quat(B)], dim=-1)
    x_curr_clean = delta_gt * omega.unsqueeze(0).expand(B, 3)
    noise = torch.randn(B, 3) * sigma_m
    curr = torch.cat([x_curr_clean + noise, _identity_quat(B)], dim=-1)

    est = JointStateEstimator(omega, p, JOINT_TYPE_PRISMATIC, initial)
    return est.estimate(curr), sigma_m, delta_gt


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("state_est_demo.png"),
        help="Output image path (PNG).",
    )
    parser.add_argument(
        "--sweep-n", type=int, default=50,
        help="Number of points in each sweep panel.",
    )
    parser.add_argument(
        "--noise-B", type=int, default=200,
        help="Number of noisy samples in the histogram panel.",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            f"matplotlib + numpy required ({e}). Install with: pip install matplotlib"
        ) from None

    rev_gt, rev_est = _revolute_sweep(args.sweep_n)
    pri_gt, pri_est = _prismatic_sweep(args.sweep_n)
    k_labels, k_est, k_units, k_gt_rad = _k_hypothesis_demo()
    noise_est, sigma_m, delta_gt = _prismatic_noise_demo(args.noise_B)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # -- [0, 0] revolute sweep -------------------------------------------------
    ax = axes[0, 0]
    rev_gt_deg = (rev_gt / _DEG).numpy()
    rev_est_deg = (rev_est / _DEG).numpy()
    ax.plot(rev_gt_deg, rev_est_deg, "o", markersize=4, color="#1f77b4",
            label="estimated")
    ax.plot(rev_gt_deg, rev_gt_deg, "k--", linewidth=1.0,
            label="y = x  (perfect)")
    ax.set_xlabel("ground truth θ  [deg]")
    ax.set_ylabel("estimated θ  [deg]")
    ax.set_title("Revolute angle sweep")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    rev_err_deg = np.abs(rev_est_deg - rev_gt_deg)
    ax.text(
        0.97, 0.05,
        f"max |error|: {rev_err_deg.max():.6f}°\nspec bound:  1°",
        transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    # -- [0, 1] prismatic sweep ------------------------------------------------
    ax = axes[0, 1]
    pri_gt_mm = (pri_gt * 1000.0).numpy()
    pri_est_mm = (pri_est * 1000.0).numpy()
    ax.plot(pri_gt_mm, pri_est_mm, "o", markersize=4, color="#ff7f0e",
            label="estimated")
    ax.plot(pri_gt_mm, pri_gt_mm, "k--", linewidth=1.0,
            label="y = x  (perfect)")
    ax.set_xlabel("ground truth Δ  [mm]")
    ax.set_ylabel("estimated Δ  [mm]")
    ax.set_title("Prismatic displacement sweep")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    pri_err_mm = np.abs(pri_est_mm - pri_gt_mm)
    ax.text(
        0.97, 0.05,
        f"max |error|: {pri_err_mm.max():.6f} mm\nspec bound:  1 mm",
        transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    # -- [1, 0] K-hypothesis bars ----------------------------------------------
    ax = axes[1, 0]
    colors = ["#2ca02c", "#d62728", "#9467bd"]
    bars = ax.bar(range(len(k_labels)), k_est, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.6)
    ax.axhline(
        y=k_gt_rad, color="black", linestyle="--", linewidth=1.3,
        label=f"true θ = {k_gt_rad / _DEG:.1f}° about +ẑ  ≈ {k_gt_rad:.4f} rad",
    )
    ax.axhline(y=0.0, color="grey", linewidth=0.6)
    ax.set_xticks(range(len(k_labels)))
    ax.set_xticklabels(k_labels)
    ax.set_ylabel("θ_est   (rad for revolute, m for prismatic)")
    ax.set_title("K = 3 hypotheses on one observed rotation\n"
                 "wrong hypotheses must collapse near 0")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right")
    y_top = max(k_est + [k_gt_rad]) * 1.25 + 0.05
    ax.set_ylim(min(k_est + [0]) - 0.05, y_top)
    for bar, val, unit in zip(bars, k_est, k_units):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + (0.015 if h >= 0 else -0.04),
            f"{val:+.4f} {unit}",
            ha="center", va="bottom" if h >= 0 else "top", fontsize=9,
        )

    # -- [1, 1] noise histogram ------------------------------------------------
    ax = axes[1, 1]
    noise_est_mm = (noise_est * 1000.0).numpy()
    ax.hist(noise_est_mm, bins=25, color="#ff7f0e", alpha=0.75,
            edgecolor="white")
    ax.axvline(
        x=delta_gt * 1000.0, color="black", linestyle="--", linewidth=1.5,
        label=f"true Δ = {delta_gt * 1000.0:.1f} mm",
    )
    ax.set_xlabel("estimated Δ  [mm]")
    ax.set_ylabel("count")
    ax.set_title(
        f"Prismatic estimates under σ = {sigma_m * 100:.1f} cm position noise"
        f"  (B = {args.noise_B})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    mean_mm = float(noise_est.mean()) * 1000.0
    std_mm = float(noise_est.std()) * 1000.0
    max_err_mm = float((noise_est - delta_gt).abs().max()) * 1000.0
    ax.text(
        0.03, 0.95,
        f"sample mean: {mean_mm:.3f} mm\n"
        f"sample std:  {std_mm:.3f} mm\n"
        f"max |error|: {max_err_mm:.3f} mm\n"
        f"theory σ_proj = {sigma_m * 1000:.1f} mm",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    fig.suptitle(
        "Module 06  —  JointStateEstimator diagnostic",
        fontsize=14, y=1.00,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")

    # Stdout summary so failures are visible without opening the PNG.
    print(f"[OK] wrote {args.out.resolve()}")
    print(f"     revolute  max error: {rev_err_deg.max():.6f}°  (spec ≤ 1°)")
    print(f"     prismatic max error: {pri_err_mm.max():.6f} mm (spec ≤ 1 mm)")
    print(f"     K-hyp     θ_est:     "
          + ", ".join(f"{lbl.split(chr(10))[0]}={v:+.4f} {u}"
                      for lbl, v, u in zip(k_labels, k_est, k_units)))
    print(f"     noise     sample std: {std_mm:.3f} mm "
          f"(theory σ_proj = {sigma_m * 1000:.1f} mm)")


if __name__ == "__main__":
    main()
