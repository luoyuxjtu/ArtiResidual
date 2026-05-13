"""Smoke test for SAM2PartTracker (spec §3.0).

RUN ON THE GPU SERVER, not in the codespace.

What this verifies:
    1. SAM2PartTracker constructs (SAM2 weights load successfully).
    2. initialize(rgb_0, [mask_0]) runs without crashing.
    3. estimate(rgb_0, depth_0, K) returns a numpy array of shape [P, 7]
       — the *bootstrap* call (first estimate) sets the reference PCDs
       and returns (centroid, identity quat) per part. No SAM2 propagation
       is exercised on this call.
    4. estimate(rgb_1, depth_1, K) returns shape [P, 7] — this exercises
       SAM2 mask propagation + ICP. Pass is shape-correct output.

What this does NOT verify:
    - Pose ACCURACY against ground truth (no GT available with random data).
    - SAM2 mask QUALITY (random pixels are not photographs).
    - Real-robot integration — that's a separate test (spec §3.0 acceptance
      test on 10 cabinet trajectories, ≤ 5mm / 3° vs. mocap).

Usage:
    python scripts/smoke_test_part_tracker.py \\
        --sam2-checkpoint $HOME/third_party/sam2/checkpoints/sam2_hiera_base_plus.pt

Exit codes:
    0   all 4 steps passed
    1   any step crashed (stack trace printed)
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np


def _make_random_rgb(H: int, W: int, seed: int) -> np.ndarray:
    """Plausible-looking RGB: structured gradient + per-pixel noise so SAM2
    has SOMETHING to lock on to (vs. iid noise which can produce zero-area
    masks)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base_r = (255.0 * xx / max(W - 1, 1))
    base_g = (255.0 * yy / max(H - 1, 1))
    base_b = (255.0 * (xx + yy) / max(W + H - 2, 1))
    img = np.stack([base_r, base_g, base_b], axis=-1)
    img += rng.normal(0.0, 5.0, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _make_random_depth(H: int, W: int, seed: int) -> np.ndarray:
    """Plausible depth in meters: ~1m plane with mild noise + a closer
    rectangular blob (so masked back-projection finds points at distinct depths)."""
    rng = np.random.default_rng(seed)
    depth = np.full((H, W), 1.0, dtype=np.float32) + rng.normal(0.0, 0.01, (H, W)).astype(np.float32)
    # Foreground "part" blob — same region as the mask we'll use.
    y0, y1 = H // 3, 2 * H // 3
    x0, x1 = W // 3, 2 * W // 3
    depth[y0:y1, x0:x1] = 0.6 + rng.normal(0.0, 0.005, (y1 - y0, x1 - x0)).astype(np.float32)
    return depth


def _make_center_box_mask(H: int, W: int) -> np.ndarray:
    """Bool mask covering the central 1/3 × 1/3 box."""
    mask = np.zeros((H, W), dtype=bool)
    y0, y1 = H // 3, 2 * H // 3
    x0, x1 = W // 3, 2 * W // 3
    mask[y0:y1, x0:x1] = True
    return mask


def _make_intrinsics(H: int, W: int) -> np.ndarray:
    """Simple pinhole intrinsics with fx=fy=W and principal point at image center."""
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = float(W)
    K[1, 1] = float(W)
    K[0, 2] = W / 2.0
    K[1, 2] = H / 2.0
    return K


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=Path(os.environ.get("SAM2_CHECKPOINT", "")) or None,
        help="Path to the SAM2 .pt checkpoint (default: $SAM2_CHECKPOINT).",
    )
    parser.add_argument("--device", default="cuda", help="torch device (cuda / cpu).")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.sam2_checkpoint is None or not args.sam2_checkpoint.is_file():
        print(
            "[ERROR] --sam2-checkpoint required (or set SAM2_CHECKPOINT env var). "
            f"Got: {args.sam2_checkpoint}",
            file=sys.stderr,
        )
        return 1

    H, W = args.height, args.width
    P = 1  # spec says smoke test only needs P=1
    rgb_0 = _make_random_rgb(H, W, args.seed)
    rgb_1 = _make_random_rgb(H, W, args.seed + 1)
    depth_0 = _make_random_depth(H, W, args.seed)
    depth_1 = _make_random_depth(H, W, args.seed + 1)
    K = _make_intrinsics(H, W)
    mask_0 = _make_center_box_mask(H, W)

    print(f"[INFO] H={H} W={W} P={P} device={args.device}")
    print(f"[INFO] checkpoint: {args.sam2_checkpoint}")

    # ---- step 1: construct ------------------------------------------------
    print("[STEP 1/4] constructing SAM2PartTracker...")
    try:
        from artiresidual.utils.part_tracker import SAM2PartTracker

        tracker = SAM2PartTracker(
            sam2_checkpoint=str(args.sam2_checkpoint),
            device=args.device,
        )
    except Exception:
        traceback.print_exc()
        print("[FAIL 1/4] construction failed.")
        return 1
    print("[PASS 1/4] construction OK")

    # ---- step 2: initialize -----------------------------------------------
    print("[STEP 2/4] running initialize()...")
    try:
        tracker.initialize(rgb_0, [mask_0])
    except Exception:
        traceback.print_exc()
        print("[FAIL 2/4] initialize() crashed.")
        return 1
    print("[PASS 2/4] initialize() OK")

    # ---- step 3: bootstrap estimate ---------------------------------------
    print("[STEP 3/4] running estimate() — bootstrap call (no SAM2 propagation)...")
    try:
        poses_boot = tracker.estimate(rgb_0, depth_0, K)
    except Exception:
        traceback.print_exc()
        print("[FAIL 3/4] bootstrap estimate() crashed.")
        return 1
    if not isinstance(poses_boot, np.ndarray):
        print(f"[FAIL 3/4] expected np.ndarray, got {type(poses_boot)}")
        return 1
    if poses_boot.shape != (P, 7):
        print(f"[FAIL 3/4] expected shape ({P}, 7), got {poses_boot.shape}")
        return 1
    quat_norm = np.linalg.norm(poses_boot[0, 3:])
    if not np.isclose(quat_norm, 1.0, atol=1e-5):
        print(f"[FAIL 3/4] bootstrap quaternion not unit norm: {quat_norm}")
        return 1
    print(f"[PASS 3/4] shape={poses_boot.shape}  pose[0]={poses_boot[0]}")

    # ---- step 4: tracked estimate (exercises SAM2 + ICP) ------------------
    print("[STEP 4/4] running estimate() — tracked call (SAM2 + ICP)...")
    try:
        poses_track = tracker.estimate(rgb_1, depth_1, K)
    except Exception:
        traceback.print_exc()
        print("[FAIL 4/4] tracked estimate() crashed.")
        return 1
    if poses_track.shape != (P, 7):
        print(f"[FAIL 4/4] expected shape ({P}, 7), got {poses_track.shape}")
        return 1
    quat_norm = np.linalg.norm(poses_track[0, 3:])
    if not np.isclose(quat_norm, 1.0, atol=1e-4):
        print(f"[FAIL 4/4] tracked quaternion not unit norm: {quat_norm}")
        return 1
    print(f"[PASS 4/4] shape={poses_track.shape}  pose[0]={poses_track[0]}")

    tracker.close()
    print()
    print("==============================================================")
    print("[OK] all 4 smoke-test steps passed.")
    print("==============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
