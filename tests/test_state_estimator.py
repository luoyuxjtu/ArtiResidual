"""Unit tests for artiresidual.refiner.state_estimator (Module 06).

Module 06 is a pure-geometry module — no ML, no perception. It converts a
6-DoF ``current_part_pose`` (produced upstream either by the simulator's GT
joint state when ``perception.use_gt_part_pose: True``, or by
``SAM2PartTracker.estimate()`` from spec §3.0 when ``False``) into a scalar
``theta_t``. These tests verify the conversion across the angle / displacement
ranges the spec acceptance test requires, plus robustness to position noise
that simulates ICP output.

Test plan (9 functions, 16 effective tests after pytest.parametrize fan-out):

    1. test_revolute_at_required_angles                    parametrized × 4
         spec acceptance: 0°, 45°, 90°, 180° — error ≤ 1° each.
    2. test_prismatic_at_required_displacements            parametrized × 3
         spec acceptance: 0 cm, 5 cm, 20 cm — error ≤ 1 mm each.
    3. test_revolute_robust_to_05cm_icp_noise              1
         Position noise (σ=0.5 cm, simulating ICP output) must NOT shift
         the orientation-based revolute estimate.
    4. test_prismatic_robust_to_05cm_icp_noise             1
         Projection of σ=0.5 cm isotropic noise onto ω̂ has σ=5 mm. Verify
         max error < 3σ ≈ 1.5 cm and mean error well under 8 mm.
    5. test_estimate_handles_various_batch_sizes           parametrized × 3
         B ∈ {1, 16, 32}: shape + correctness.
    6. test_K3_hypotheses_tracked_simultaneously           1
         The IMM K=3 instance pattern: correct hypothesis recovers θ;
         wrong-axis and wrong-type hypotheses collapse near 0.
    7. test_revolute_with_offset_hinge_p_is_irrelevant     1
         Two estimators sharing ω but differing in p must agree on θ
         (the spec deliberately says angle depends on orientation only).
    8. test_fixed_joint_estimate_is_always_zero            1
         JOINT_TYPE_FIXED returns 0 regardless of observed motion.
    9. test_negative_signs_are_recovered_correctly         1
         -45° revolute and -10 cm prismatic both return negative θ.

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
    """Convert axis-angle to scalar-first quaternion ``(w, x, y, z)``."""
    axis = axis / torch.linalg.norm(axis).clamp(min=1e-12)
    half = angle_rad / 2.0
    w = math.cos(half)
    s = math.sin(half)
    return torch.tensor(
        [w, s * axis[0].item(), s * axis[1].item(), s * axis[2].item()],
        dtype=torch.float32,
    )


def _identity_quat(B: int) -> torch.Tensor:
    """Return ``[B, 4]`` identity quaternions ``(1, 0, 0, 0)``."""
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    return q.expand(B, 4).contiguous()


def _make_pose(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Concatenate position ``[B, 3]`` and quaternion ``[B, 4]`` → ``[B, 7]``."""
    return torch.cat([x, q], dim=-1)


# ---------------------------------------------------------------------------
# 1. Pure revolute at the four spec-required angles.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta_deg", [0.0, 45.0, 90.0, 180.0])
def test_revolute_at_required_angles(theta_deg: float) -> None:
    """Spec acceptance: revolute angle error ≤ 1° at {0°, 45°, 90°, 180°}.

    180° is the edge case for the twist-swing decomposition (``w = 0``);
    keeping it in the matrix protects against numerical regressions when the
    representative-quaternion sign flip is touched.
    """
    theta_gt = theta_deg * _DEG
    omega = torch.tensor([0.0, 0.0, 1.0])
    p = torch.zeros(3)

    initial = _make_pose(torch.zeros(1, 3), _identity_quat(B=1))
    q_rot = _axis_angle_to_quat(omega, theta_gt).unsqueeze(0)
    curr = _make_pose(torch.zeros(1, 3), q_rot)

    est = JointStateEstimator(omega, p, JOINT_TYPE_REVOLUTE, initial)
    theta_est = est.estimate(curr)
    err_deg = abs(theta_est.item() - theta_gt) / _DEG
    assert err_deg < 1.0, (
        f"angle {theta_deg}°: got {theta_est.item() / _DEG:.4f}°  err={err_deg:.4f}°"
    )


# ---------------------------------------------------------------------------
# 2. Pure prismatic at the three spec-required displacements.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delta_m", [0.0, 0.05, 0.20])
def test_prismatic_at_required_displacements(delta_m: float) -> None:
    """Spec acceptance: prismatic error ≤ 1 mm at {0 cm, 5 cm, 20 cm}."""
    omega = torch.tensor([1.0, 0.0, 0.0])
    p = torch.zeros(3)

    x_init = torch.tensor([[0.2, 0.5, 0.3]])
    q_init = _identity_quat(B=1)
    initial = _make_pose(x_init, q_init)

    x_curr = x_init + delta_m * omega.unsqueeze(0)
    curr = _make_pose(x_curr, q_init)

    est = JointStateEstimator(omega, p, JOINT_TYPE_PRISMATIC, initial)
    theta_est = est.estimate(curr)
    err_m = abs(theta_est.item() - delta_m)
    assert err_m < 1e-3, (
        f"delta {delta_m * 100:.1f} cm: got {theta_est.item() * 100:.4f} cm "
        f"err={err_m * 1000:.4f} mm"
    )


# ---------------------------------------------------------------------------
# 3. Revolute robust to ±0.5 cm ICP noise on position.
# ---------------------------------------------------------------------------


def test_revolute_robust_to_05cm_icp_noise() -> None:
    """Spec acceptance: robust to ±0.5 cm Gaussian noise on ``current_part_pose``
    (simulating ICP output noise — Section 3.0's ICP step has ~5 mm std).

    Implementation uses ONLY orientation, so position noise must NOT shift
    the angle estimate. Verify per-sample error stays ≤ 1° across a batch.
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
    icp_noise = torch.randn(B, 3) * 0.005   # σ = 0.5 cm
    x_curr_noisy = x_init + icp_noise
    curr = _make_pose(x_curr_noisy, q_rot)

    est = JointStateEstimator(omega, p, JOINT_TYPE_REVOLUTE, initial)
    theta_est = est.estimate(curr)
    err_deg = torch.abs(theta_est - theta_gt) / _DEG
    assert torch.all(err_deg < 1.0), f"max error {err_deg.max().item():.4f}°"


# ---------------------------------------------------------------------------
# 4. Prismatic robust to ±0.5 cm ICP noise on position.
# ---------------------------------------------------------------------------


def test_prismatic_robust_to_05cm_icp_noise() -> None:
    """Spec acceptance: robust to ±0.5 cm Gaussian noise on ``current_part_pose``
    (simulating ICP output noise — Section 3.0's ICP step).

    Theory: an isotropic σ=5 mm noise projected onto the unit axis has σ=5 mm.
    Max error over 32 samples should sit under ~3σ = 1.5 cm; mean absolute
    error ≈ σ·√(2/π) ≈ 4 mm.
    """
    torch.manual_seed(42)

    omega = torch.tensor([1.0, 0.0, 0.0])
    p = torch.zeros(3)
    delta_gt = 0.15  # 15 cm — well above the noise floor

    B = 32
    x_init = torch.zeros(B, 3)
    q_init = _identity_quat(B)
    initial = _make_pose(x_init, q_init)

    x_curr_clean = delta_gt * omega.unsqueeze(0).expand(B, 3)
    icp_noise = torch.randn(B, 3) * 0.005   # σ = 0.5 cm
    curr = _make_pose(x_curr_clean + icp_noise, q_init)

    est = JointStateEstimator(omega, p, JOINT_TYPE_PRISMATIC, initial)
    theta_est = est.estimate(curr)
    errors = torch.abs(theta_est - delta_gt)

    assert errors.max().item() < 0.015, (
        f"max error {errors.max().item() * 100:.3f} cm; expected < 1.5 cm (3σ)"
    )
    assert errors.mean().item() < 0.008, (
        f"mean error {errors.mean().item() * 100:.3f} cm; expected < 0.8 cm"
    )


# ---------------------------------------------------------------------------
# 5. Batch sizes 1, 16, 32.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B", [1, 16, 32])
def test_estimate_handles_various_batch_sizes(B: int) -> None:
    """Output shape ``(B,)`` and per-sample correctness for B ∈ {1, 16, 32}."""
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
# 6. K=3 hypotheses tracked simultaneously (the IMM pattern).
# ---------------------------------------------------------------------------


def test_K3_hypotheses_tracked_simultaneously() -> None:
    """IMM canonical pattern: K=3 hypotheses share one observed pose.

    True motion: 30° about +ẑ. Hypotheses:
        k=0  revolute about +ẑ   (correct)         → θ ≈ +30°
        k=1  revolute about +ŷ   (orthogonal axis) → θ ≈ 0    (no twist)
        k=2  prismatic along +x̂  (wrong type)      → θ = 0    (no translation)

    This is exactly what Module 04 (IMM refiner) relies on as the
    discriminating evidence — wrong hypotheses produce trivial θ, so the
    residual flow they imply diverges from observation and their weight
    decays.
    """
    B = 4
    K = 3
    theta_gt = 30.0 * _DEG

    omegas = [
        torch.tensor([0.0, 0.0, 1.0]),   # k=0
        torch.tensor([0.0, 1.0, 0.0]),   # k=1
        torch.tensor([1.0, 0.0, 0.0]),   # k=2
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

    assert torch.all(torch.abs(theta_t_k[:, 0] - theta_gt) / _DEG < 1.0)
    assert torch.all(torch.abs(theta_t_k[:, 1]) / _DEG < 1.0)
    assert torch.all(torch.abs(theta_t_k[:, 2]) < 1e-6)


# ---------------------------------------------------------------------------
# 7. p irrelevance — two estimators with same ω but different p agree.
# ---------------------------------------------------------------------------


def test_revolute_with_offset_hinge_p_is_irrelevant_to_angle() -> None:
    """The revolute angle is a property of the *rotation*, not of which point
    on the axis we pick. Two estimators sharing ω but differing in p must
    return identical θ."""
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
# 8. JOINT_TYPE_FIXED always returns zero.
# ---------------------------------------------------------------------------


def test_fixed_joint_estimate_is_always_zero() -> None:
    """JOINT_TYPE_FIXED returns 0 regardless of observed motion."""
    omega = torch.tensor([1.0, 0.0, 0.0])
    p = torch.zeros(3)

    B = 4
    initial = _make_pose(torch.zeros(B, 3), _identity_quat(B))

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
# 9. Sign correctness for negative motion.
# ---------------------------------------------------------------------------


def test_negative_signs_are_recovered_correctly() -> None:
    """Sign of θ must follow the sign of the true motion (both joint types)."""
    # Revolute: -45° about +ẑ.
    omega = torch.tensor([0.0, 0.0, 1.0])
    theta_gt = -math.pi / 4

    initial = _make_pose(torch.zeros(1, 3), _identity_quat(B=1))
    q_rot = _axis_angle_to_quat(omega, theta_gt).unsqueeze(0)
    curr = _make_pose(torch.zeros(1, 3), q_rot)

    e_rev = JointStateEstimator(omega, torch.zeros(3), JOINT_TYPE_REVOLUTE, initial)
    theta = e_rev.estimate(curr)
    assert theta.item() < 0.0
    assert abs(theta.item() - theta_gt) < 1.0 * _DEG

    # Prismatic: -10 cm along +x̂.
    omega_p = torch.tensor([1.0, 0.0, 0.0])
    delta_gt = -0.10
    x_curr = delta_gt * omega_p.unsqueeze(0)
    curr_p = _make_pose(x_curr, _identity_quat(B=1))

    e_pri = JointStateEstimator(omega_p, torch.zeros(3), JOINT_TYPE_PRISMATIC, initial)
    delta = e_pri.estimate(curr_p)
    assert delta.item() < 0.0
    assert abs(delta.item() - delta_gt) < 1e-3
