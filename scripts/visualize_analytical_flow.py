"""Visualize Module 05 analytical flow on the spec-acceptance scenario.

CPU-only. Run on the server (or anywhere with torch + matplotlib):

    pip install matplotlib                # if not already
    python scripts/visualize_analytical_flow.py --out /tmp/flow_demo.png

Two side-by-side 3-D panels are produced:
    1. Revolute door rotated 45° about +z, hinge at origin. Red arrows are the
       analytical flow; they should be tangent to the arc that each panel
       point traces, i.e., perpendicular to the in-plane radius and to ω.
    2. Prismatic drawer with axis at 45° in the xy-plane. Arrows should be
       parallel everywhere and aligned with ω.

This is the visual companion to ``tests/test_analytical_flow.py``: passing
tests + a visual that looks "tangential / parallel" is the sanity bundle
before we trust Module 05 anywhere downstream.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from artiresidual.refiner.analytical_flow import (
    JOINT_TYPE_PRISMATIC,
    JOINT_TYPE_REVOLUTE,
    analytical_flow,
)


def _make_door(theta_rad: float, n_h: int = 6, n_v: int = 5) -> torch.Tensor:
    """Sample a door panel at angle theta about the +z axis through origin.

    The rest panel lies in the x-z plane (y=0) with x ∈ [0.15, 1.0] and
    z ∈ [0, 1.5]. Rotation matrix R_z(theta) maps it to the rendered scene.
    """
    xs = torch.linspace(0.15, 1.0, n_h)
    zs = torch.linspace(0.0, 1.5, n_v)
    grid_x, grid_z = torch.meshgrid(xs, zs, indexing="ij")
    rest = torch.stack(
        [grid_x, torch.zeros_like(grid_x), grid_z], dim=-1
    ).reshape(-1, 3)
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rest @ R.T


def _make_drawer(
    axis_theta_rad: float, slide: float = 0.4, n_h: int = 5, n_v: int = 4
) -> torch.Tensor:
    """Sample a drawer face translated by ``slide`` along an oblique axis."""
    ys_local = torch.linspace(-0.3, 0.3, n_h)
    zs_local = torch.linspace(0.0, 0.6, n_v)
    grid_y, grid_z = torch.meshgrid(ys_local, zs_local, indexing="ij")
    # Local face in the y-z plane, x = 0. We translate along (cos, sin, 0).
    rest = torch.stack(
        [torch.zeros_like(grid_y), grid_y, grid_z], dim=-1
    ).reshape(-1, 3)
    c, s = math.cos(axis_theta_rad), math.sin(axis_theta_rad)
    return rest + slide * torch.tensor([c, s, 0.0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("flow_demo.png"),
        help="Output image path (PNG).",
    )
    parser.add_argument(
        "--theta-deg",
        type=float,
        default=45.0,
        help="Joint configuration for the revolute door (degrees).",
    )
    parser.add_argument(
        "--arrow-length",
        type=float,
        default=0.25,
        help="Arrow length (matplotlib quiver length).",
    )
    args = parser.parse_args()

    # Lazy import so the module imports cheaply if matplotlib is missing.
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            f"matplotlib is required for this script ({e}). "
            f"Install with: pip install matplotlib"
        ) from None

    theta = math.radians(args.theta_deg)

    # --- Revolute door --------------------------------------------------------
    door = _make_door(theta)
    omega_r = torch.tensor([0.0, 0.0, 1.0])
    p_r = torch.tensor([0.0, 0.0, 0.0])
    flow_r = analytical_flow(
        door, omega_r, p_r, JOINT_TYPE_REVOLUTE, normalize_per_part=True
    )

    # --- Prismatic drawer ------------------------------------------------------
    axis_theta = math.radians(45.0)
    drawer = _make_drawer(axis_theta)
    omega_p = torch.tensor([math.cos(axis_theta), math.sin(axis_theta), 0.0])
    p_p = torch.tensor([0.0, 0.0, 0.0])
    flow_p = analytical_flow(
        drawer, omega_p, p_p, JOINT_TYPE_PRISMATIC, normalize_per_part=True
    )

    fig = plt.figure(figsize=(12, 5.5))

    # ``normalize=False`` is critical for revolute: |ω × (x - p)| scales with the
    # in-plane distance to the hinge axis, so points farther out (x ≈ 1.0) draw
    # ~6.7× longer arrows than points near the hinge (x ≈ 0.15). The flow itself
    # was per-part-normalized so the longest arrow has unit data magnitude;
    # ``length`` then sets the on-screen size of that longest arrow.
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(
        door[:, 0].numpy(), door[:, 1].numpy(), door[:, 2].numpy(),
        s=22, c="#1f77b4", depthshade=False,
    )
    ax.quiver(
        door[:, 0].numpy(), door[:, 1].numpy(), door[:, 2].numpy(),
        flow_r[:, 0].numpy(), flow_r[:, 1].numpy(), flow_r[:, 2].numpy(),
        length=args.arrow_length, normalize=False,
        color="#d62728", linewidth=1.2, arrow_length_ratio=0.25,
    )
    ax.plot([0, 0], [0, 0], [-0.15, 1.7], color="black", linewidth=2.5, label="ω (hinge)")
    ax.set_title(f"Revolute door @ {args.theta_deg:.0f}° about +ẑ\n"
                 f"flow = ω × (x - p)   (length ∝ distance to hinge)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(loc="upper left")

    # Prismatic: every flow vector is ω̂ exactly, so the arrows ARE uniform.
    # We still pass normalize=False for consistency with the revolute panel.
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(
        drawer[:, 0].numpy(), drawer[:, 1].numpy(), drawer[:, 2].numpy(),
        s=22, c="#ff7f0e", depthshade=False,
    )
    ax.quiver(
        drawer[:, 0].numpy(), drawer[:, 1].numpy(), drawer[:, 2].numpy(),
        flow_p[:, 0].numpy(), flow_p[:, 1].numpy(), flow_p[:, 2].numpy(),
        length=args.arrow_length, normalize=False,
        color="#d62728", linewidth=1.2, arrow_length_ratio=0.25,
    )
    axis_pts_x = [-0.2 * omega_p[0].item(), 1.2 * omega_p[0].item()]
    axis_pts_y = [-0.2 * omega_p[1].item(), 1.2 * omega_p[1].item()]
    ax.plot(axis_pts_x, axis_pts_y, [0.3, 0.3], color="black", linewidth=2.5, label="ω (slide)")
    ax.set_title("Prismatic drawer (axis at 45° in xy)\n"
                 "flow = ω̂  (uniform)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(loc="upper left")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[OK] wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
