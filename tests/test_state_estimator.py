"""Unit tests for artiresidual.refiner.state_estimator (Module 06).

Spec §3 Module 06 acceptance test:
    "On synthetic trajectories with known theta_t: error ≤ 1° (revolute) or
     ≤ 1 mm (prismatic). Robust to ±0.5 cm Gaussian noise on part pose."

Test plan (10 functions, ~27 effective tests with pytest.parametrize):

    1. test_revolute_estimate_recovers_45deg_about_z_axis
         Headline acceptance test: door rotated 45° about +z, error ≤ 1°.
    2. test_revolute_estimate_recovers_various_angles
         Parametrized over 10 angles in (-π/2, ~+170°); all within 1°.
    3. test_prismatic_estimate_recovers_displacement
         Parametrized over 7 displacements; all within 1 mm.
    4. test_revolute_robust_to_05cm_position_noise
         σ=0.5cm position noise; revolute uses only orientation → unaffected.
    5. test_prismatic_robust_to_05cm_position_noise
         σ=0.5cm position noise → error stays within 3σ ≈ 1.5cm theoretical.
    6. test_estimate_handles_various_batch_sizes
         Parametrized B ∈ {1, 16, 32}; output shape + correctness.
    7. test_K_hypotheses_tracked_simultaneously
         K=3 estimators (different ω, p, type) on one observed pose.
    8. test_revolute_with_offset_hinge_p_is_irrelevant_to_angle
         Same ω, different p → same θ. p is stored but not used.
    9. test_fixed_joint_estimate_is_always_zero
         JOINT_TYPE_FIXED always returns 0 regardless of input.
   10. test_negative_angles_and_displacements_have_correct_sign
         Revolute -π/4 and prismatic -0.1m return negative θ.

All tests are CPU-only and require only torch.
"""
from __future__ import annotations

import math

import pytest
import torch

from artiresidual.refiner.analytical_flow import (
    JOINT_TYPE_FIXED,
    JOINT_TYPE_PRISMATIC,
    JOINT_TYPE_REVOLUTE,
)
from artiresidual.refiner.state_estimator import JointStateEstimator


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

_DEG = math.pi / 180.0


def _axis_angle_to_quat(axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Convert axis-angle to (w, x, y, z) scalar-first quaternion.

    The axis is normalized internally. Returned dtype is float32 to match the
    project default.
    """
    axis = axis / torch.linalg.norm(axis).clamp(min=1e-12)
    half = angle_rad / 2.0
    w = math.cos(half)
    s = math.sin(half)
    return torch.tensor(
        [w, s * axis[0].item(), s * axis[1].item(), s * axis[2].item()],
        dtype=torch.float32,
    )


def _identity_quat(B: int) -> torch.Tensor:
    """Return [B, 4] identity quaternions ``(1, 0, 0, 0)`` (scalar-first)."""
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    return q.expand(B, 4).contiguous()


def _make_pose(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Concatenate position [B, 3] and quaternion [B, 4] → pose [B, 7]."""
    return torch.cat([x, q], dim=-1)


# ---------------------------------------------------------------------------
# 1. Headline acceptance: revolute @ 45° about +z.
# ---------------------------------------------------------------------------


def test_revolute_estimate_recovers_45deg_about_z_axis() -> None:
    """Door rotated 45° about +z, hinge at origin. Spec: error ≤ 1°."""
    theta_gt = math.pi / 4
    omega = torch.tensor([0.0, 0.0, 1.0])
    p = torch.tensor([0.0, 0.0, 0.0])

    x_init = torch.tensor([[0.5, 0.2, 0.7]])
    q_init = _identity_quat(B=1)
    initial = _make_pose(x_init, q_init)

    q_rot = _axis_angle_to_quat(omega, theta_gt).unsqueeze(0)
    # Position can be anywhere — revolute estimator ignores it.
    curr = _make_pose(torch.tensor([[1.3, -0.4, 0.7]]), q_rot)

    est = JointStateEstimator(omega, p, JOINT_TYPE_REVOLUTE, initial)
    theta_est = est.estimate(curr)

    assert theta_est.shape == (1,)
    assert abs(theta_est.item() - theta_gt) < 1.0 * _DEG


# ---------------------------------------------------------------------------
# 2. Revolute at various angles.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "theta_deg",
    [-90.0, -45.0, -10.0, 0.0, 10.0, 30.0, 60.0, 90.0, 135.0, 170.0],
)
def test_revolute_estimate_recovers_various_angles(theta_deg: float) -> None:
    """Spec acceptance at multiple angles. Tolerance: 1°."""
    theta_gt = theta_deg * _DEG
    omega = torch.tensor([0.0, 0.0, 1.0])
    p = torch.zeros(3)

    initial = _make_pose(torch.zeros(1, 3), _identity_quat(B=1))
    q_rot = _axis_angle_to_quat(omega, theta_gt).unsqueeze(0)
    curr = _make_pose(torch.zeros(1, 3), q_rot)

    est = JointStateEstimator(omega, p, JOINT_TYPE_REVOLUTE, initial)
    theta_est = est.estimate(curr)
    assert abs(theta_est.item() - theta_gt) < 1.0 * _DEG, (
        f"angle {theta_deg}°: got {theta_est.item() / _DEG:.4f}°"
    )


# ---------------------------------------------------------------------------
# 3. Prismatic at various displacements.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "delta_m", [-0.5, -0.05, 0.0, 0.001, 0.01, 0.1, 0.3]
)
def test_prismatic_estimate_recovers_displacement(delta_m: float) -> None:
    """Spec acceptance: error ≤ 1 mm at multiple displacements."""
    omega = torch.tensor([1.0, 0.0, 0.0])
    p = torch.zeros(3)

    x_init = torch.tensor([[0.2, 0.5, 0.3]])
    q_init = _identity_quat(B=1)
    initial = _make_pose(x_init, q_init)

    x_curr = x_init + delta_m * omega.unsqueeze(0)
    curr = _make_pose(x_curr, q_init)

    est = JointStateEstimator(omega, p, JOINT_TYPE_PRISMATIC, initial)
    theta_est = est.estimate(curr)
    assert abs(theta_est.item() - delta_m) < 1e-3, (
        f"delta {delta_m} m: got {theta_est.item():.6f} m"
    )


# ---------------------------------------------------------------------------
# 4. Revolute robust to 0.5 cm position noise.
# ---------------------------------------------------------------------------


def test_revolute_robust_to_05cm_position_noise() -> None:
    """Spec: robust to ±0.5cm Gaussian position noise.

    Implementation uses only orientation, so position noise must NOT shift the
    angle estimate. Verify per-sample error stays ≤ 1° across a batch of 16.
    """
    torch.manual_seed(42)

    omega = torch.tensor([0.0, 0.0, 1.0])
    p = torch.zeros(3)
    theta_gt = 30.0 * _DEG

    B = 16
    x_init = torch.randn(B, 3)
    q_init = _identity_quat(B)
    initial = _make_pose(x_init, q_init)

    q_rot = _axis_angle_to_quat(omega, theta_gt).expand(B, 4).contiguous()
    noise = torch.randn(B, 3) * 0.005  # σ = 0.5 cm
    x_curr_noisy = x_init + noise
    curr = _make_pose(x_curr_noisy, q_rot)

    est = JointStateEstimator(omega, p, JOINT_TYPE_REVOLUTE, initial)
    theta_est = est.estimate(curr)
    err_deg = torch.abs(theta_est - theta_gt) / _DEG
    assert torch.all(err_deg < 1.0), f"max error {err_deg.max().item():.4f}°"


# ---------------------------------------------------------------------------
# 5. Prismatic robust to 0.5 cm position noise.
# ---------------------------------------------------------------------------


def test_prismatic_robust_to_05cm_position_noise() -> None:
    """Spec: robust to ±0.5cm Gaussian position noise.

    For prismatic, the projection of an isotropic σ=5mm noise onto the unit
    axis has σ=5mm. Theory: max over 32 samples ≤ ~3σ = 1.5cm at the 99.7%
    level; mean absolute error ≈ σ·sqrt(2/π) ≈ 4mm.
    """
    torch.manual_seed(42)

    omega = torch.tensor([1.0, 0.0, 0.0])
    p = torch.zeros(3)
    delta_gt = 0.15  # 15 cm displacement (well above the noise floor)

    B = 32
    x_init = torch.zeros(B, 3)
    q_init = _identity_quat(B)
    initial = _make_pose(x_init, q_init)

    x_curr_clean = delta_gt * omega.unsqueeze(0).expand(B, 3)
    noise = torch.randn(B, 3) * 0.005
    x_curr = x_curr_clean + noise
    curr = _make_pose(x_curr, q_init)

    est = JointStateEstimator(omega, p, JOINT_TYPE_PRISMATIC, initial)
    theta_est = est.estimate(curr)
    errors = torch.abs(theta_est - delta_gt)

    # 3σ_proj upper bound for any single sample.
    assert errors.max().item() < 0.015, (
        f"max error {errors.max().item() * 100:.3f} cm; expected < 1.5 cm"
    )
    # Theoretical MAE ≈ 0.4 cm; allow up to 0.8 cm to cover small-N variance.
    assert errors.mean().item() < 0.008, (
        f"mean error {errors.mean().item() * 100:.3f} cm; expected < 0.8 cm"
    )


# ---------------------------------------------------------------------------
# 6. Batch sizes 1, 16, 32.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B", [1, 16, 32])
def test_estimate_handles_various_batch_sizes(B: int) -> None:
    """Verify output shape and correctness across batch sizes."""
    omega = torch.tensor([0.0, 0.0, 1.0])
    p = torch.zeros(3)
    theta_gt = 20.0 * _DEG

    initial = _make_pose(torch.zeros(B, 3), _identity_quat(B))
    q_rot = _axis_angle_to_quat(omega, theta_gt).expand(B, 4).contiguous()
    curr = _make_pose(torch.zeros(B, 3), q_rot)

    est = JointStateEstimator(omega, p, JOINT_TYPE_REVOLUTE, initial)
    theta_est = est.estimate(curr)

    assert theta_est.shape == (B,)
    err_deg = torch.abs(theta_est - theta_gt) / _DEG
    assert torch.all(err_deg < 1.0), f"max error {err_deg.max().item():.4f}°"


# ---------------------------------------------------------------------------
# 7. K hypotheses tracked simultaneously.
# ---------------------------------------------------------------------------


def test_K_hypotheses_tracked_simultaneously() -> None:
    """K=3 hypotheses share one observed pose. True motion: 30° about +z.

    Per spec §3 Module 04 implementation notes, the IMM refiner instantiates
    K estimators with the same ``initial_part_pose`` but distinct
    ``(omega, p, joint_type)``. Each returns its own ``theta_t``; the refiner
    combines them. This test verifies the multi-instance pattern works and
    that wrong-hypothesis estimates collapse correctly.
    """
    B = 4
    K = 3
    theta_gt = 30.0 * _DEG

    omegas = [
        torch.tensor([0.0, 0.0, 1.0]),  # k=0: correct axis (+z, revolute)
        torch.tensor([0.0, 1.0, 0.0]),  # k=1: wrong axis (+y, revolute)
        torch.tensor([1.0, 0.0, 0.0]),  # k=2: prismatic +x
    ]
    types = [JOINT_TYPE_REVOLUTE, JOINT_TYPE_REVOLUTE, JOINT_TYPE_PRISMATIC]

    initial = _make_pose(torch.zeros(B, 3), _identity_quat(B))
    q_rot_z = _axis_angle_to_quat(omegas[0], theta_gt).expand(B, 4).contiguous()
    curr = _make_pose(torch.zeros(B, 3), q_rot_z)

    estimators = [
        JointStateEstimator(omegas[k], torch.zeros(3), types[k], initial)
        for k in range(K)
    ]
    theta_t_k = torch.stack([e.estimate(curr) for e in estimators], dim=-1)
    assert theta_t_k.shape == (B, K)

    # k=0 (correct hypothesis): θ ≈ 30°.
    assert torch.all(torch.abs(theta_t_k[:, 0] - theta_gt) / _DEG < 1.0)
    # k=1 (orthogonal axis): no twist component about +y, so θ ≈ 0.
    assert torch.all(torch.abs(theta_t_k[:, 1]) / _DEG < 1.0)
    # k=2 (prismatic, but no translation): θ ≈ 0 exactly.
    assert torch.all(torch.abs(theta_t_k[:, 2]) < 1e-6)


# ---------------------------------------------------------------------------
# 8. p irrelevance: same ω, different p → same θ.
# ---------------------------------------------------------------------------


def test_revolute_with_offset_hinge_p_is_irrelevant_to_angle() -> None:
    """Revolute angle depends only on orientation, NOT on p. Two estimators
    with the same ω but different p must agree on θ."""
    omega = torch.tensor([0.0, 0.0, 1.0])
    theta_gt = 60.0 * _DEG

    initial = _make_pose(torch.zeros(1, 3), _identity_quat(B=1))
    q_rot = _axis_angle_to_quat(omega, theta_gt).unsqueeze(0)
    curr = _make_pose(torch.zeros(1, 3), q_rot)

    p1 = torch.zeros(3)
    p2 = torch.tensor([1.0, 2.0, -3.0])
    e1 = JointStateEstimator(omega, p1, JOINT_TYPE_REVOLUTE, initial)
    e2 = JointStateEstimator(omega, p2, JOINT_TYPE_REVOLUTE, initial)

    theta1 = e1.estimate(curr)
    theta2 = e2.estimate(curr)
    assert torch.allclose(theta1, theta2, atol=1e-7)
    assert abs(theta1.item() - theta_gt) < 1.0 * _DEG


# ---------------------------------------------------------------------------
# 9. Fixed joint → 0 regardless of input.
# ---------------------------------------------------------------------------


def test_fixed_joint_estimate_is_always_zero() -> None:
    """``JOINT_TYPE_FIXED`` returns zero θ no matter what pose change is fed."""
    omega = torch.tensor([1.0, 0.0, 0.0])
    p = torch.zeros(3)

    B = 4
    initial = _make_pose(torch.zeros(B, 3), _identity_quat(B))

    # Adversarial: arbitrary translation AND rotation.
    x_curr = torch.randn(B, 3)
    q_curr = (
        _axis_angle_to_quat(torch.tensor([0.3, 0.7, 0.5]), 1.234)
        .expand(B, 4)
        .contiguous()
    )
    curr = _make_pose(x_curr, q_curr)

    est = JointStateEstimator(omega, p, JOINT_TYPE_FIXED, initial)
    theta_est = est.estimate(curr)
    assert torch.allclose(theta_est, torch.zeros(B))


# ---------------------------------------------------------------------------
# 10. Sign correctness.
# ---------------------------------------------------------------------------


def test_negative_angles_and_displacements_have_correct_sign() -> None:
    """Sign of θ must match the sign of the true motion (revolute & prismatic)."""
    # Revolute: -45° about +z.
    omega = torch.tensor([0.0, 0.0, 1.0])
    theta_gt = -math.pi / 4

    initial = _make_pose(torch.zeros(1, 3), _identity_quat(B=1))
    q_rot = _axis_angle_to_quat(omega, theta_gt).unsqueeze(0)
    curr = _make_pose(torch.zeros(1, 3), q_rot)

    e_rev = JointStateEstimator(omega, torch.zeros(3), JOINT_TYPE_REVOLUTE, initial)
    theta = e_rev.estimate(curr)
    assert theta.item() < 0.0
    assert abs(theta.item() - theta_gt) < 1.0 * _DEG

    # Prismatic: -0.1 m along +x.
    omega_p = torch.tensor([1.0, 0.0, 0.0])
    delta_gt = -0.1
    x_curr = delta_gt * omega_p.unsqueeze(0)
    curr_p = _make_pose(x_curr, _identity_quat(B=1))

    e_pri = JointStateEstimator(omega_p, torch.zeros(3), JOINT_TYPE_PRISMATIC, initial)
    delta = e_pri.estimate(curr_p)
    assert delta.item() < 0.0
    assert abs(delta.item() - delta_gt) < 1e-3
