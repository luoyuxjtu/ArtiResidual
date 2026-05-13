"""Visualize SAM2PartTracker (spec §3.0).

RUN ON THE GPU SERVER, not in the codespace.

Produces a 2x3 PNG showing the pipeline end-to-end with a SYNTHETIC scene
("fake door" — a colored rectangle moves across a textured background
between t=0 and t=1 so ICP has structure to lock onto):

    [0,0]   RGB at t=0           with initial part mask overlaid in green
    [0,1]   RGB at t=1           with SAM2-propagated mask overlaid in red
    [0,2]   Mask overlap         green=ref, red=tracked, yellow=both (IoU view)
    [1,0]   Depth at t=0         masked region highlighted
    [1,1]   3-D PCDs             reference (blue) + current (orange) +
                                 ICP-aligned reference (red dashed),
                                 viewed top-down (x-z plane)
    [1,2]   Pose summary         text: bootstrap pose, tracked pose, ICP
                                 translation magnitude, ICP rotation angle.

Usage:
    python scripts/visualize_part_tracker.py \\
        --sam2-checkpoint $HOME/third_party/sam2/checkpoints/sam2_hiera_base_plus.pt \\
        --out /tmp/part_tracker_demo.png
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic scene generator.
# ---------------------------------------------------------------------------


def _make_scene(
    H: int,
    W: int,
    door_tl: tuple[int, int],
    door_size: tuple[int, int],
    bg_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize an RGB image, depth, and bool mask for a "fake door".

    The door is a constant-colored rectangle at depth 0.6 m on top of a
    textured background plane at depth 1.0 m. Background texture is a fixed
    seeded perlin-ish gradient + noise so SAM2 has visual variation to fix
    onto, and ICP has non-uniform depth to fit.
    """
    rng = np.random.default_rng(bg_seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # Background RGB: a smooth gradient + structured texture.
    bg_r = 60.0 + 80.0 * np.sin(xx / 20.0) + 30.0 * np.cos(yy / 17.0)
    bg_g = 100.0 + 40.0 * np.cos(xx / 13.0)
    bg_b = 140.0 + 50.0 * np.sin((xx + yy) / 25.0)
    rgb = np.stack([bg_r, bg_g, bg_b], axis=-1)
    rgb += rng.normal(0.0, 4.0, size=rgb.shape)

    # Background depth: 1m plane with tiny noise.
    depth = np.full((H, W), 1.0, dtype=np.float32)
    depth += rng.normal(0.0, 0.005, size=depth.shape).astype(np.float32)

    # Door overlay.
    y0, x0 = door_tl
    dh, dw = door_size
    y1, x1 = min(y0 + dh, H), min(x0 + dw, W)
    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = True
    # Door color: a strong contrasting yellow.
    rgb[y0:y1, x0:x1, 0] = 230.0 + rng.normal(0, 2.0, (y1 - y0, x1 - x0))
    rgb[y0:y1, x0:x1, 1] = 200.0 + rng.normal(0, 2.0, (y1 - y0, x1 - x0))
    rgb[y0:y1, x0:x1, 2] = 40.0 + rng.normal(0, 2.0, (y1 - y0, x1 - x0))
    # Door is 40 cm in front of the wall.
    depth[y0:y1, x0:x1] = 0.6 + rng.normal(0.0, 0.003, (y1 - y0, x1 - x0)).astype(
        np.float32
    )

    return np.clip(rgb, 0, 255).astype(np.uint8), depth, mask


def _make_intrinsics(H: int, W: int) -> np.ndarray:
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = float(W)
    K[1, 1] = float(W)
    K[0, 2] = W / 2.0
    K[1, 2] = H / 2.0
    return K


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    alpha = 0.45
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def _overlay_mask_difference(
    rgb: np.ndarray, mask_ref: np.ndarray, mask_track: np.ndarray
) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    alpha = 0.50
    both = mask_ref & mask_track
    only_ref = mask_ref & ~mask_track
    only_track = mask_track & ~mask_ref
    out[only_ref] = (1.0 - alpha) * out[only_ref] + alpha * np.array([0, 220, 0])
    out[only_track] = (1.0 - alpha) * out[only_track] + alpha * np.array([220, 0, 0])
    out[both] = (1.0 - alpha) * out[both] + alpha * np.array([220, 220, 0])
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_transform(pcd: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ pcd.T).T + T[:3, 3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=Path(os.environ.get("SAM2_CHECKPOINT", "")) or None,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--shift-px", type=int, default=20, help="Door x-shift between t=0 and t=1.")
    parser.add_argument("--out", type=Path, default=Path("part_tracker_demo.png"))
    args = parser.parse_args()

    if args.sam2_checkpoint is None or not args.sam2_checkpoint.is_file():
        raise SystemExit(
            f"--sam2-checkpoint required (got: {args.sam2_checkpoint}). "
            f"Set SAM2_CHECKPOINT env var or pass --sam2-checkpoint."
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(f"matplotlib required: {e}") from None

    from artiresidual.utils.part_tracker import (
        SAM2PartTracker,
        _back_project_masked,
        _run_icp,
    )

    H, W = args.height, args.width
    K = _make_intrinsics(H, W)
    door_size = (H // 3, W // 3)
    door_tl_0 = (H // 3, W // 3)
    door_tl_1 = (door_tl_0[0], door_tl_0[1] + args.shift_px)

    rgb_0, depth_0, mask_0 = _make_scene(H, W, door_tl_0, door_size, bg_seed=0)
    rgb_1, depth_1, mask_1_gt = _make_scene(H, W, door_tl_1, door_size, bg_seed=0)

    print(f"[INFO] scene H={H} W={W} door shift x={args.shift_px} px")
    print(f"[INFO] loading SAM2 from {args.sam2_checkpoint} on {args.device}")
    tracker = SAM2PartTracker(str(args.sam2_checkpoint), device=args.device)

    # Initialize + bootstrap pose.
    tracker.initialize(rgb_0, [mask_0])
    pose_boot = tracker.estimate(rgb_0, depth_0, K)  # [1, 7]

    # Tracked pose at t=1 — exercises SAM2 + ICP.
    pose_track = tracker.estimate(rgb_1, depth_1, K)  # [1, 7]

    # For the visualization we also need the propagated mask at t=1.
    # Re-run SAM2 propagation directly to get it (estimate() discards it).
    sam2_mask_t1 = tracker._sam2_propagate_to(tracker._frame_count - 1)[0]

    # Compute reference + current PCDs and run ICP a second time for the plot.
    ref_pcd = _back_project_masked(depth_0, K, mask_0)
    cur_pcd = _back_project_masked(depth_1, K, sam2_mask_t1)
    T = _run_icp(ref_pcd, cur_pcd) if cur_pcd.shape[0] > 0 else np.eye(4)
    ref_aligned = _apply_transform(ref_pcd, T)

    # IoU for the plot annotation.
    iou = (sam2_mask_t1 & mask_1_gt).sum() / max((sam2_mask_t1 | mask_1_gt).sum(), 1)
    trans_mm = float(np.linalg.norm(T[:3, 3])) * 1000.0
    # ICP rotation angle from trace.
    cos_theta = (np.trace(T[:3, :3]) - 1.0) / 2.0
    rot_deg = math.degrees(math.acos(max(min(cos_theta, 1.0), -1.0)))

    # --- figure ------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(_overlay_mask(rgb_0, mask_0, (0, 220, 0)))
    axes[0, 0].set_title("t = 0   (initial mask, green)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(_overlay_mask(rgb_1, sam2_mask_t1, (220, 0, 0)))
    axes[0, 1].set_title("t = 1   (SAM2-propagated mask, red)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(_overlay_mask_difference(rgb_1, mask_1_gt, sam2_mask_t1))
    axes[0, 2].set_title(
        f"GT mask (green) vs. SAM2 (red); both = yellow\n"
        f"IoU = {iou:.3f}  (acceptance ≥ 0.85)"
    )
    axes[0, 2].axis("off")

    axes[1, 0].imshow(depth_0, cmap="viridis")
    axes[1, 0].contour(mask_0, levels=[0.5], colors="lime", linewidths=2)
    axes[1, 0].set_title("depth at t = 0  (mask outline in green)")
    axes[1, 0].axis("off")

    ax3 = axes[1, 1]
    if ref_pcd.shape[0] > 0:
        ax3.scatter(ref_pcd[:, 0], ref_pcd[:, 2], s=4, c="#1f77b4", alpha=0.6, label="reference (t=0)")
    if cur_pcd.shape[0] > 0:
        ax3.scatter(cur_pcd[:, 0], cur_pcd[:, 2], s=4, c="#ff7f0e", alpha=0.6, label="current (t=1)")
    if ref_aligned.shape[0] > 0:
        ax3.scatter(
            ref_aligned[:, 0], ref_aligned[:, 2],
            s=4, c="#d62728", alpha=0.4, marker="x",
            label="ICP-aligned reference",
        )
    ax3.set_xlabel("x  [m, camera frame]")
    ax3.set_ylabel("z  [m]")
    ax3.set_title("Point clouds (top-down view)")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()  # z grows away from camera

    txt = (
        f"Bootstrap pose [0]:\n"
        f"  pos  = ({pose_boot[0, 0]:+.3f}, {pose_boot[0, 1]:+.3f}, {pose_boot[0, 2]:+.3f}) m\n"
        f"  quat = ({pose_boot[0, 3]:+.3f}, {pose_boot[0, 4]:+.3f}, "
        f"{pose_boot[0, 5]:+.3f}, {pose_boot[0, 6]:+.3f})\n\n"
        f"Tracked pose [0]:\n"
        f"  pos  = ({pose_track[0, 0]:+.3f}, {pose_track[0, 1]:+.3f}, {pose_track[0, 2]:+.3f}) m\n"
        f"  quat = ({pose_track[0, 3]:+.3f}, {pose_track[0, 4]:+.3f}, "
        f"{pose_track[0, 5]:+.3f}, {pose_track[0, 6]:+.3f})\n\n"
        f"ICP transform t=0 → t=1:\n"
        f"  ‖Δt‖    = {trans_mm:.2f} mm\n"
        f"  rot   = {rot_deg:.3f}°\n\n"
        f"Mask IoU (SAM2 vs GT) = {iou:.3f}\n"
        f"Spec acceptance      ≥ 0.85"
    )
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0, 1.0, txt, transform=axes[1, 2].transAxes,
        family="monospace", fontsize=10, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle("Spec §3.0  —  SAM2PartTracker pipeline (synthetic scene)", fontsize=14)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    tracker.close()
    print(f"[OK] wrote {args.out.resolve()}")
    print(
        f"     IoU={iou:.3f}  trans={trans_mm:.2f} mm  rot={rot_deg:.3f}°  "
        f"(expected: trans ≈ {args.shift_px * 1.0 / W * 0.6 * 1000:.1f} mm under perfect ICP)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
