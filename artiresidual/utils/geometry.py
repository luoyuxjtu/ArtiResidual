"""Geometry primitives used across modules.

Quaternion / rotation matrix conversions, tangent-space exp/log maps for S²,
SE(3) interpolation, point-cloud transformations. Pure functions, no learning.

Where the IMM refiner (Module 04) needs the tangent-space exp map for omega
updates (spec §4.4), prefer adding it here rather than inline in the filter.
"""
from __future__ import annotations

import torch
from torch import Tensor

__all__: list[str] = ["transform_poses"]


# ---------------------------------------------------------------------------
# Internal quaternion helpers (scalar-first convention: w, x, y, z).
# ---------------------------------------------------------------------------


def _quat_mul(q1: Tensor, q2: Tensor) -> Tensor:
    """Quaternion product q1 ⊗ q2, scalar-first (w, x, y, z).

    Represents the rotation obtained by first applying q2, then q1,
    i.e. the combined rotation matrix is R(q1) @ R(q2).

    Args:
        q1: [..., 4] quaternion.
        q2: [..., 4] quaternion.

    Returns:
        [..., 4] product quaternion (unit norm if both inputs are unit norm).
    """
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _rot_to_quat(R: Tensor) -> Tensor:
    """Convert rotation matrix to unit quaternion (scalar-first: w, x, y, z).

    Uses Shepperd's method: selects the numerically largest component as
    the pivot to avoid catastrophic cancellation near singular branches.

    Args:
        R: [..., 3, 3] rotation matrix.

    Returns:
        [..., 4] unit quaternion.
    """
    *batch, _, _ = R.shape
    R_flat = R.reshape(-1, 3, 3)  # [M, 3, 3]
    M = R_flat.shape[0]

    # Pre-compute all four "s" denominators up front to avoid repeated sqrt.
    # Branch w: pivot = trace + 1 = 4 w²
    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]  # [M]
    s_w = torch.sqrt((trace + 1.0).clamp(min=0.0)) * 2.0  # s_w = 4w
    q_w = torch.stack(
        [
            0.25 * s_w,
            (R_flat[:, 2, 1] - R_flat[:, 1, 2]) / s_w.clamp(min=1e-10),
            (R_flat[:, 0, 2] - R_flat[:, 2, 0]) / s_w.clamp(min=1e-10),
            (R_flat[:, 1, 0] - R_flat[:, 0, 1]) / s_w.clamp(min=1e-10),
        ],
        dim=-1,
    )  # [M, 4]

    # Branch x: pivot = 1 + R00 - R11 - R22 = 4 x²
    s_x = torch.sqrt((1.0 + R_flat[:, 0, 0] - R_flat[:, 1, 1] - R_flat[:, 2, 2]).clamp(min=0.0)) * 2.0
    q_x = torch.stack(
        [
            (R_flat[:, 2, 1] - R_flat[:, 1, 2]) / s_x.clamp(min=1e-10),
            0.25 * s_x,
            (R_flat[:, 0, 1] + R_flat[:, 1, 0]) / s_x.clamp(min=1e-10),
            (R_flat[:, 0, 2] + R_flat[:, 2, 0]) / s_x.clamp(min=1e-10),
        ],
        dim=-1,
    )  # [M, 4]

    # Branch y: pivot = 1 + R11 - R00 - R22 = 4 y²
    s_y = torch.sqrt((1.0 + R_flat[:, 1, 1] - R_flat[:, 0, 0] - R_flat[:, 2, 2]).clamp(min=0.0)) * 2.0
    q_y = torch.stack(
        [
            (R_flat[:, 0, 2] - R_flat[:, 2, 0]) / s_y.clamp(min=1e-10),
            (R_flat[:, 0, 1] + R_flat[:, 1, 0]) / s_y.clamp(min=1e-10),
            0.25 * s_y,
            (R_flat[:, 1, 2] + R_flat[:, 2, 1]) / s_y.clamp(min=1e-10),
        ],
        dim=-1,
    )  # [M, 4]

    # Branch z: pivot = 1 + R22 - R00 - R11 = 4 z²
    s_z = torch.sqrt((1.0 + R_flat[:, 2, 2] - R_flat[:, 0, 0] - R_flat[:, 1, 1]).clamp(min=0.0)) * 2.0
    q_z = torch.stack(
        [
            (R_flat[:, 1, 0] - R_flat[:, 0, 1]) / s_z.clamp(min=1e-10),
            (R_flat[:, 0, 2] + R_flat[:, 2, 0]) / s_z.clamp(min=1e-10),
            (R_flat[:, 1, 2] + R_flat[:, 2, 1]) / s_z.clamp(min=1e-10),
            0.25 * s_z,
        ],
        dim=-1,
    )  # [M, 4]

    # Select the branch whose pivot is largest (most numerically stable).
    cond_w = trace > 0  # [M]
    cond_x = (R_flat[:, 0, 0] > R_flat[:, 1, 1]) & (R_flat[:, 0, 0] > R_flat[:, 2, 2]) & ~cond_w
    cond_y = (R_flat[:, 1, 1] > R_flat[:, 2, 2]) & ~cond_w & ~cond_x
    # cond_z = ~cond_w & ~cond_x & ~cond_y (implicit default)

    q = q_z.clone()
    q = torch.where(cond_y.unsqueeze(-1), q_y, q)
    q = torch.where(cond_x.unsqueeze(-1), q_x, q)
    q = torch.where(cond_w.unsqueeze(-1), q_w, q)

    # Renormalize to guard against accumulated floating-point drift.
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    return q.reshape(*batch, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transform_poses(poses: Tensor, cam_to_world: Tensor) -> Tensor:
    """Apply a rigid cam→world transform to a batch of SE(3) poses.

    `SAM2PartTracker.estimate()` returns poses in the camera frame (see
    DECISIONS.md 2026-05-14). Call this function to convert them to world
    frame before feeding into Module 04 `step()` or Module 06.

    Args:
        poses: [..., 7] poses in camera frame.
            Layout: ``[x, y, z, w, qx, qy, qz]`` — position first, then
            scalar-first quaternion ``(w, x, y, z)``.
        cam_to_world: [4, 4] homogeneous rigid transform, camera→world.
            Must be a proper rotation + translation (no scaling/shear).

    Returns:
        poses_world: [..., 7] poses in world frame, same layout as input.

    Example::

        cam_to_world = torch.eye(4)          # identity: camera IS world
        poses_world = transform_poses(poses, cam_to_world)
        assert torch.allclose(poses_world, poses)
    """
    R_cw = cam_to_world[:3, :3]  # [3, 3]
    t_cw = cam_to_world[:3, 3]   # [3]

    pos = poses[..., :3]   # [..., 3]  x, y, z in camera frame
    quat = poses[..., 3:]  # [..., 4]  (w, x, y, z) in camera frame

    # Transform position: p_world = R_cw @ p_cam + t_cw
    # pos @ R_cw.T is equivalent to (R_cw @ pos.T).T for batched inputs.
    pos_world = pos @ R_cw.T + t_cw  # [..., 3]

    # Transform rotation: q_world = q_cw ⊗ q_cam
    # _rot_to_quat expects [..., 3, 3]; pass a [1, 3, 3] and squeeze back.
    q_cw = _rot_to_quat(R_cw.unsqueeze(0)).squeeze(0)  # [4]
    quat_world = _quat_mul(q_cw.expand_as(quat), quat)  # [..., 4]

    return torch.cat([pos_world, quat_world], dim=-1)  # [..., 7]
