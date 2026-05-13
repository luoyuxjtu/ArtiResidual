"""End-to-end state-estimator pipeline test on a synthetic revolute door.

RUN ON THE GPU SERVER, not in the codespace.

Pipeline exercised:

    render T frames of a door rotating about +y_cam
        |
        v
    SAM2PartTracker.initialize(rgb[0], [gt_mask_0])
    SAM2PartTracker.estimate(rgb[t], depth[t], K)    → current_part_pose [1, 7]
        |
        v
    JointStateEstimator(omega, p, REVOLUTE, initial_part_pose)
    estimator.estimate(current_part_pose)            → theta_est [1]
        |
        v
    compare theta_est[t] vs theta_gt[t]

Frame convention: everything in the camera frame. Camera is fixed. Hinge axis
omega = +y_cam. Hinge reference point p lives on the left edge of the door.

Writes a 2x3 PNG to outputs/ by default. No GT data download needed; the scene
is procedurally rasterized with numpy.

Usage:
    python scripts/e2e_test_state_estimator.py \\
        --sam2-checkpoint $HOME/third_party/sam2/checkpoints/sam2_hiera_base_plus.pt
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic scene: a revolute door rotating about +y_cam.
# ---------------------------------------------------------------------------


DOOR_W = 0.20  # door width  (x extent of the flat door at theta=0, meters)
DOOR_H = 0.30  # door height (y extent, meters)
HINGE_Z = 0.70  # hinge depth (meters, in front of the camera)
HINGE_X = -DOOR_W / 2.0  # hinge sits at the LEFT edge of the door
HINGE_Y = 0.0  # hinge vertical center
BG_DEPTH = 1.50  # back wall depth


def _door_corners_at_theta(theta_rad: float) -> np.ndarray:
    """Return the 4 door corners in the camera frame at rotation ``theta``.

    Hinge axis is +y_cam through ``(HINGE_X, 0, HINGE_Z)``. Corners at
    ``theta = 0`` lie on the plane z = HINGE_Z with x from HINGE_X to
    HINGE_X + DOOR_W and y in [-DOOR_H/2, +DOOR_H/2].

    Order: top-left, top-right, bottom-right, bottom-left. Returned as [4, 3].
    """
    # Local frame corners (hinge at origin, door extends +x).
    local = np.array(
        [
            [0.0, +DOOR_H / 2.0, 0.0],
            [DOOR_W, +DOOR_H / 2.0, 0.0],
            [DOOR_W, -DOOR_H / 2.0, 0.0],
            [0.0, -DOOR_H / 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    # Rotate about +y by theta, then translate hinge to camera-frame location.
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    R = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )
    rotated = (R @ local.T).T  # [4, 3]
    rotated[:, 0] += HINGE_X
    rotated[:, 1] += HINGE_Y
    rotated[:, 2] += HINGE_Z
    return rotated


def _project(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project camera-frame points [N, 3] to pixel coords [N, 2] (float)."""
    z = np.maximum(points_cam[:, 2], 1e-6)
    u = K[0, 0] * points_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * points_cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1)


def _rasterize_triangle_zbuffer(
    uv: np.ndarray,         # [3, 2] pixel coords
    depth: np.ndarray,      # [3] camera-z per vertex (meters)
    color: tuple[float, float, float],  # RGB 0-255
    rgb_out: np.ndarray,    # [H, W, 3] float32 mutated in place
    depth_out: np.ndarray,  # [H, W] float32 mutated in place
    mask_out: np.ndarray,   # [H, W] bool mutated in place (set True on write)
) -> None:
    """Rasterize a flat-colored triangle with 1/z-interpolated depth + z-buffer.

    The door is a single color, so no UV texturing needed. Depth is
    perspective-correct (interpolate 1/z then invert).
    """
    H, W = depth_out.shape
    u0, v0 = uv[0]
    u1, v1 = uv[1]
    u2, v2 = uv[2]
    min_u = int(max(0, math.floor(min(u0, u1, u2))))
    max_u = int(min(W - 1, math.ceil(max(u0, u1, u2))))
    min_v = int(max(0, math.floor(min(v0, v1, v2))))
    max_v = int(min(H - 1, math.ceil(max(v0, v1, v2))))
    if max_u < min_u or max_v < min_v:
        return

    uu, vv = np.meshgrid(
        np.arange(min_u, max_u + 1, dtype=np.float64),
        np.arange(min_v, max_v + 1, dtype=np.float64),
    )
    # Barycentric via 2D cross products.
    denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
    if abs(denom) < 1e-9:
        return
    w0 = ((v1 - v2) * (uu - u2) + (u2 - u1) * (vv - v2)) / denom
    w1 = ((v2 - v0) * (uu - u2) + (u0 - u2) * (vv - v2)) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    if not inside.any():
        return

    # Perspective-correct z: interpolate 1/z linearly in screen space.
    inv_z = w0 / depth[0] + w1 / depth[1] + w2 / depth[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        z_interp = 1.0 / inv_z
    ys = np.arange(min_v, max_v + 1)
    xs = np.arange(min_u, max_u + 1)
    tile_depth = depth_out[min_v:max_v + 1, min_u:max_u + 1]
    tile_mask = mask_out[min_v:max_v + 1, min_u:max_u + 1]
    tile_rgb = rgb_out[min_v:max_v + 1, min_u:max_u + 1]
    closer = inside & np.isfinite(z_interp) & (z_interp > 0) & (z_interp < tile_depth)
    tile_depth[closer] = z_interp[closer]
    tile_mask[closer] = True
    tile_rgb[closer] = np.asarray(color, dtype=np.float32)


def _make_background(H: int, W: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Textured back wall at depth BG_DEPTH. Returns (rgb [H,W,3] float32, depth [H,W] float32)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bg_r = 60.0 + 80.0 * np.sin(xx / 20.0) + 30.0 * np.cos(yy / 17.0)
    bg_g = 100.0 + 40.0 * np.cos(xx / 13.0)
    bg_b = 140.0 + 50.0 * np.sin((xx + yy) / 25.0)
    rgb = np.stack([bg_r, bg_g, bg_b], axis=-1).astype(np.float32)
    rgb += rng.normal(0.0, 3.0, size=rgb.shape).astype(np.float32)
    depth = np.full((H, W), BG_DEPTH, dtype=np.float32)
    depth += rng.normal(0.0, 0.004, size=depth.shape).astype(np.float32)
    return rgb, depth


def _render_frame(
    theta_rad: float,
    H: int,
    W: int,
    K: np.ndarray,
    seed: int,
    door_depth_noise_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one frame: (rgb uint8 [H,W,3], depth float32 [H,W], mask bool [H,W]).

    ``door_depth_noise_m``: σ of Gaussian noise added to door-pixel depth after
    rasterization. Use ~0.002 m to mimic an Intel RealSense L515 at 0.6-0.7 m
    (manufacturer specs ~1-3 mm RMS at that range).
    """
    rgb_f, depth_f = _make_background(H, W, seed=seed)
    mask = np.zeros((H, W), dtype=bool)

    corners = _door_corners_at_theta(theta_rad)  # [4, 3] camera frame
    uv = _project(corners, K)
    door_color = (230.0, 200.0, 40.0)  # yellow

    # Two triangles: (0,1,2) and (0,2,3).
    for tri in ((0, 1, 2), (0, 2, 3)):
        _rasterize_triangle_zbuffer(
            uv[list(tri)], corners[list(tri), 2], door_color, rgb_f, depth_f, mask,
        )

    if door_depth_noise_m > 0.0 and mask.any():
        rng = np.random.default_rng(seed + 7919)
        n = int(mask.sum())
        depth_f[mask] += rng.normal(0.0, door_depth_noise_m, size=n).astype(np.float32)

    rng = np.random.default_rng(seed + 101)
    rgb_f += rng.normal(0.0, 2.0, size=rgb_f.shape).astype(np.float32)
    rgb = np.clip(rgb_f, 0, 255).astype(np.uint8)
    return rgb, depth_f, mask


def _make_intrinsics(H: int, W: int) -> np.ndarray:
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = float(W)
    K[1, 1] = float(W)
    K[0, 2] = W / 2.0
    K[1, 2] = H / 2.0
    return K


# ---------------------------------------------------------------------------
# Plotting helpers.
# ---------------------------------------------------------------------------


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    c = np.asarray(color, dtype=np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * c
    return np.clip(out, 0, 255).astype(np.uint8)


def _contour_mask(mask: np.ndarray) -> np.ndarray:
    """A thin outline of the mask (boundary pixels) as a bool array."""
    m = mask.astype(bool)
    edge = np.zeros_like(m)
    edge[1:, :] |= m[1:, :] & ~m[:-1, :]
    edge[:-1, :] |= m[:-1, :] & ~m[1:, :]
    edge[:, 1:] |= m[:, 1:] & ~m[:, :-1]
    edge[:, :-1] |= m[:, :-1] & ~m[:, 1:]
    # dilate by 1 for visibility
    e = edge.copy()
    e[1:, :] |= edge[:-1, :]
    e[:-1, :] |= edge[1:, :]
    e[:, 1:] |= edge[:, :-1]
    e[:, :-1] |= edge[:, 1:]
    return e


def _back_project_for_plot(depth: np.ndarray, K: np.ndarray, mask: np.ndarray) -> np.ndarray:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    v_idx, u_idx = np.where(mask)
    z = depth[v_idx, u_idx]
    valid = np.isfinite(z) & (z > 0.05) & (z < 5.0)
    v_idx, u_idx, z = v_idx[valid], u_idx[valid], z[valid].astype(np.float64)
    x = (u_idx.astype(np.float64) - cx) * z / fx
    y = (v_idx.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


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
    parser.add_argument("--theta-max-deg", type=float, default=30.0)
    parser.add_argument("--n-frames", type=int, default=10)
    parser.add_argument(
        "--door-depth-noise-mm",
        type=float,
        default=2.0,
        help="Gaussian sigma added to door-pixel depth (mm). 0 disables. "
             "Default 2 mm mimics Intel RealSense L515 at ~0.6 m.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/e2e_state_estimator_revolute.png"),
    )
    args = parser.parse_args()

    if args.sam2_checkpoint is None or not args.sam2_checkpoint.is_file():
        raise SystemExit(
            f"--sam2-checkpoint required (got: {args.sam2_checkpoint}). "
            f"Set SAM2_CHECKPOINT or pass --sam2-checkpoint."
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(f"matplotlib required: {e}") from None

    import torch
    from artiresidual.refiner.analytical_flow import JOINT_TYPE_REVOLUTE
    from artiresidual.refiner.state_estimator import JointStateEstimator
    from artiresidual.utils.part_tracker import SAM2PartTracker

    H, W = args.height, args.width
    K = _make_intrinsics(H, W)
    theta_max = math.radians(args.theta_max_deg)
    theta_gt = np.linspace(0.0, theta_max, args.n_frames, dtype=np.float64)

    print(f"[INFO] H={H} W={W} T={args.n_frames} theta_max={args.theta_max_deg:.1f}deg "
          f"door_depth_noise={args.door_depth_noise_mm:.1f}mm")
    print("[INFO] rendering synthetic frames...")
    door_noise_m = args.door_depth_noise_mm * 1e-3
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for t, th in enumerate(theta_gt):
        frames.append(_render_frame(th, H, W, K, seed=1000 + t, door_depth_noise_m=door_noise_m))

    # Sanity: the GT mask at t=0 must be non-empty (door visible).
    if not frames[0][2].any():
        raise SystemExit("[ERROR] t=0 door mask is empty; renderer is broken.")
    print(f"[INFO] t=0 mask area = {frames[0][2].sum()} px")

    print(f"[INFO] loading SAM2 from {args.sam2_checkpoint} on {args.device}")
    tracker = SAM2PartTracker(str(args.sam2_checkpoint), device=args.device)

    rgb0, depth0, mask0 = frames[0]
    tracker.initialize(rgb0, [mask0])
    # Bootstrap call: returns (centroid, identity_quat). This IS the
    # initial_part_pose we feed into JointStateEstimator.
    pose0 = tracker.estimate(rgb0, depth0, K)  # [1, 7] numpy
    print(f"[INFO] bootstrap pose = {pose0[0]}")

    omega = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    # p is stored but unused (DECISIONS.md); pass the hinge point for completeness.
    p = torch.tensor([HINGE_X, HINGE_Y, HINGE_Z], dtype=torch.float64)
    initial_pose_t = torch.from_numpy(pose0).to(torch.float64)
    estimator = JointStateEstimator(omega, p, JOINT_TYPE_REVOLUTE, initial_pose_t)

    # Track + estimate through the rest of the sequence. We record SAM2's
    # propagated mask and back-projected PCD at each step for plotting.
    theta_est = np.zeros(args.n_frames, dtype=np.float64)
    theta_est[0] = 0.0  # bootstrap frame, by construction
    iou_series = np.zeros(args.n_frames, dtype=np.float64)
    iou_series[0] = 1.0  # GT mask at t=0, so IoU is by definition 1
    tracked_masks: list[np.ndarray] = [mask0]

    for t in range(1, args.n_frames):
        rgb_t, depth_t, mask_gt_t = frames[t]
        pose_t_np = tracker.estimate(rgb_t, depth_t, K)           # [1, 7]
        pose_t = torch.from_numpy(pose_t_np).to(torch.float64)
        theta_est[t] = float(estimator.estimate(pose_t).item())
        sam2_mask = tracker._sam2_propagate_to(tracker._frame_count - 1)[0]
        tracked_masks.append(sam2_mask)
        inter = (sam2_mask & mask_gt_t).sum()
        union = (sam2_mask | mask_gt_t).sum()
        iou_series[t] = inter / max(union, 1)
        print(
            f"[t={t:02d}] theta_gt={math.degrees(theta_gt[t]):+6.2f}deg  "
            f"theta_est={math.degrees(theta_est[t]):+6.2f}deg  "
            f"err={math.degrees(theta_est[t] - theta_gt[t]):+6.2f}deg  "
            f"IoU={iou_series[t]:.3f}"
        )

    err_deg = np.degrees(theta_est - theta_gt)
    abs_err = np.abs(err_deg)
    print("\n=== summary ===")
    print(f"  max |err|  = {abs_err.max():.3f} deg")
    print(f"  mean |err| = {abs_err.mean():.3f} deg")
    print(f"  median IoU = {float(np.median(iou_series)):.3f}")

    # --- plot --------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    t_mid = args.n_frames // 2
    t_end = args.n_frames - 1

    # Row 0: three RGB frames with overlays.
    for col, t in enumerate((0, t_mid, t_end)):
        rgb, _, mask_gt = frames[t]
        if t == 0:
            overlay = _overlay_mask(rgb, mask_gt, (0, 220, 0), alpha=0.45)
            title = f"t = 0   initial mask (green)\ntheta_gt = 0.00°"
        else:
            overlay = _overlay_mask(rgb, tracked_masks[t], (220, 0, 0), alpha=0.45)
            overlay = _overlay_mask(overlay, _contour_mask(mask_gt), (0, 220, 0), alpha=0.85)
            title = (
                f"t = {t}   SAM2 mask (red) + GT contour (green)\n"
                f"theta_gt = {math.degrees(theta_gt[t]):.2f}°   IoU = {iou_series[t]:.3f}"
            )
        axes[0, col].imshow(overlay)
        axes[0, col].set_title(title, fontsize=10)
        axes[0, col].axis("off")

    # Row 1 col 0: PCDs top-down (x-z plane).
    ax_pcd = axes[1, 0]
    ref_pcd = _back_project_for_plot(frames[0][1], K, frames[0][2])
    end_pcd = _back_project_for_plot(frames[t_end][1], K, tracked_masks[t_end])
    if ref_pcd.shape[0] > 0:
        ax_pcd.scatter(ref_pcd[:, 0], ref_pcd[:, 2], s=4, c="#1f77b4", alpha=0.6, label=f"t=0 (ref, n={ref_pcd.shape[0]})")
    if end_pcd.shape[0] > 0:
        ax_pcd.scatter(end_pcd[:, 0], end_pcd[:, 2], s=4, c="#ff7f0e", alpha=0.6, label=f"t={t_end} (tracked, n={end_pcd.shape[0]})")
    # Draw hinge.
    ax_pcd.scatter([HINGE_X], [HINGE_Z], marker="x", s=80, c="k", label="hinge")
    ax_pcd.set_xlabel("x  [m, camera frame]")
    ax_pcd.set_ylabel("z  [m]")
    ax_pcd.set_title("Point clouds (top-down x-z)")
    ax_pcd.legend(loc="upper right", fontsize=8)
    ax_pcd.grid(True, alpha=0.3)
    ax_pcd.invert_yaxis()

    # Row 1 col 1: theta(t) GT vs estimate.
    ax_th = axes[1, 1]
    ts = np.arange(args.n_frames)
    ax_th.plot(ts, np.degrees(theta_gt), "k-o", lw=2, ms=5, label="GT")
    ax_th.plot(ts, np.degrees(theta_est), "--s", lw=2, ms=5, color="#d62728", label="estimated")
    ax_th.set_xlabel("frame t")
    ax_th.set_ylabel("theta  [deg]")
    ax_th.set_title("Joint angle: GT vs end-to-end estimate")
    ax_th.grid(True, alpha=0.3)
    ax_th.legend(loc="upper left", fontsize=9)

    # Row 1 col 2: |error|(t) and summary text.
    ax_er = axes[1, 2]
    ax_er.plot(ts, abs_err, "-o", lw=2, ms=5, color="#9467bd", label="|err|")
    ax_er.axhline(abs_err.mean(), ls=":", color="gray", label=f"mean = {abs_err.mean():.2f}°")
    ax_er.axhline(abs_err.max(), ls="--", color="red", label=f"max  = {abs_err.max():.2f}°")
    ax_er.set_xlabel("frame t")
    ax_er.set_ylabel("|theta_est - theta_gt|  [deg]")
    ax_er.set_title("End-to-end angle error")
    ax_er.grid(True, alpha=0.3)
    ax_er.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        f"End-to-end state-estimator test  —  revolute door, "
        f"theta_max = {args.theta_max_deg:.0f}°, T = {args.n_frames} frames, "
        f"door depth noise = {args.door_depth_noise_mm:.1f} mm",
        fontsize=12,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    tracker.close()
    print(f"\n[OK] wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
