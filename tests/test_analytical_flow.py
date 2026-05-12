"""Unit tests for artiresidual.refiner.analytical_flow (Module 05).

Spec §3 Module 05 acceptance test:
    "For known synthetic configurations (door at 45° about z-axis), computed
     flow matches analytical expectation to floating-point precision."

The five tests below realize that acceptance test plus the {hard, soft, diff}
variant guarantees:

    1. test_revolute_door_at_45deg_matches_omega_cross_x_minus_p
         The headline acceptance test. Door points sampled after a 45°
         rotation about z; verify f = ω × (x - p) at every point to FP precision.
    2. test_revolute_with_offset_hinge_uses_x_minus_p_not_x
         Same 45° geometry but with the hinge offset from origin; catches the
         bug where someone forgets to subtract p.
    3. test_prismatic_drawer_at_45deg_axis_is_uniform
         Prismatic-joint analogue of the acceptance test: drawer face translated
         along an oblique axis; flow must be uniformly equal to ω̂ everywhere.
    4. test_soft_variant_collapses_to_hard_when_logits_are_one_hot
         The soft type-mixing variant must reduce to the hard revolute output
         in the limit of a one-hot logit.
    5. test_diff_variant_propagates_gradient_through_omega_and_p
         The differentiable variant must produce non-zero gradients on both ω
         and p (needed for the refiner's Δμ head and joint fine-tune).

All tests run on CPU and require only torch.
"""
from __future__ import annotations

import math

import torch

from artiresidual.refiner.analytical_flow import (
    JOINT_TYPE_PRISMATIC,
    JOINT_TYPE_REVOLUTE,
    analytical_flow,
    analytical_flow_diff,
    analytical_flow_soft,
)


def test_revolute_door_at_45deg_matches_omega_cross_x_minus_p() -> None:
    """Acceptance test — door rotated 45° about +z, flow = ω × (x - p)."""
    theta = math.pi / 4
    c, s = math.cos(theta), math.sin(theta)
    # Door panel: x ∈ {0.3, 1.0} from the hinge, z ∈ {0.0, 1.5}, y = 0 initially.
    # After 45° rotation about z, the (x, 0, z) panel maps to (x·c, x·s, z).
    coords = torch.tensor(
        [
            [0.3 * c, 0.3 * s, 0.0],
            [1.0 * c, 1.0 * s, 0.0],
            [0.3 * c, 0.3 * s, 1.5],
            [1.0 * c, 1.0 * s, 1.5],
        ]
    )
    omega = torch.tensor([0.0, 0.0, 1.0])
    p = torch.tensor([0.0, 0.0, 0.0])

    flow = analytical_flow(
        coords, omega, p, JOINT_TYPE_REVOLUTE, normalize_per_part=False
    )

    # ω × (x - p) with ω = +ẑ, p = 0:  (-y, x, 0).
    expected = torch.tensor(
        [
            [-0.3 * s, 0.3 * c, 0.0],
            [-1.0 * s, 1.0 * c, 0.0],
            [-0.3 * s, 0.3 * c, 0.0],
            [-1.0 * s, 1.0 * c, 0.0],
        ]
    )
    assert flow.shape == (4, 3)
    assert torch.allclose(flow, expected, atol=1e-7)

    # Tangential property: every flow vector is perpendicular to (x - p).
    rel = coords - p.unsqueeze(0)
    dots = (flow * rel).sum(dim=-1)
    assert torch.allclose(dots, torch.zeros(4), atol=1e-7)


def test_revolute_with_offset_hinge_uses_x_minus_p_not_x() -> None:
    """Hinge offset from origin — verifies (x - p) is used, not raw x."""
    theta = math.pi / 4
    c, s = math.cos(theta), math.sin(theta)
    p = torch.tensor([0.5, -0.2, 0.0])
    rel_x = 1.0
    # Point at relative position (rel_x, 0, 0) rotated by 45° around p.
    coords = torch.tensor(
        [[p[0].item() + rel_x * c, p[1].item() + rel_x * s, 0.4]]
    )
    omega = torch.tensor([0.0, 0.0, 1.0])

    flow = analytical_flow(
        coords, omega, p, JOINT_TYPE_REVOLUTE, normalize_per_part=False
    )

    # Expected: ω × (x - p) = +ẑ × (c, s, 0.4) = (-s, c, 0).
    expected = torch.tensor([[-s, c, 0.0]])
    assert torch.allclose(flow, expected, atol=1e-7)

    # Sanity: using p = 0 would give a DIFFERENT (wrong) answer.
    flow_wrong = analytical_flow(
        coords,
        omega,
        torch.zeros(3),
        JOINT_TYPE_REVOLUTE,
        normalize_per_part=False,
    )
    assert not torch.allclose(flow_wrong, expected, atol=1e-3)


def test_prismatic_drawer_at_45deg_axis_is_uniform() -> None:
    """Prismatic counterpart — drawer slid along an oblique axis."""
    theta = math.pi / 4
    omega = torch.tensor([math.cos(theta), math.sin(theta), 0.0])
    p = torch.tensor([0.0, 0.0, 0.0])
    # Drawer face sampled at arbitrary positions (face has been translated by
    # 0.4 along ω from its rest pose, but flow must not depend on that).
    delta = 0.4
    base = torch.tensor(
        [
            [delta * math.cos(theta), delta * math.sin(theta), 0.0],
            [delta * math.cos(theta) - 0.1, delta * math.sin(theta) + 0.05, 0.3],
            [delta * math.cos(theta) + 0.07, delta * math.sin(theta) - 0.02, 0.55],
        ]
    )

    flow = analytical_flow(
        base, omega, p, JOINT_TYPE_PRISMATIC, normalize_per_part=False
    )
    expected = omega.unsqueeze(0).expand_as(flow)
    assert torch.allclose(flow, expected, atol=1e-7)


def test_soft_variant_collapses_to_hard_when_logits_are_one_hot() -> None:
    """``analytical_flow_soft`` must equal the hard revolute output when the
    revolute logit dominates (the IMM design assumes this limit holds).

    We compare with ``normalize_per_part=False`` so that the hard variant's
    max-norm and the diff variant's logsumexp-smoothed-max-norm don't disagree
    by a few percent — we want to isolate the soft-mixing behavior here, not
    the normalization smoothing.
    """
    theta = math.pi / 4
    c, s = math.cos(theta), math.sin(theta)
    coords = torch.tensor([[c, s, 0.0], [0.5 * c, 0.5 * s, 1.0]])
    omega = torch.tensor([0.0, 0.0, 1.0])
    p = torch.tensor([0.0, 0.0, 0.0])

    hard = analytical_flow(
        coords, omega, p, JOINT_TYPE_REVOLUTE, normalize_per_part=False
    )
    # Very large logit on revolute (index 0); negligible on prismatic / fixed.
    logits = torch.tensor([50.0, -50.0, -50.0])
    soft = analytical_flow_soft(
        coords, omega, p, logits, normalize_per_part=False, temperature=1.0
    )
    assert torch.allclose(soft, hard, atol=1e-6)


def test_diff_variant_propagates_gradient_through_omega_and_p() -> None:
    """``analytical_flow_diff`` must accumulate gradients on ω and p.

    The refiner's residual head (spec §3 Module 04) emits Δμ = (Δω, Δp) in
    tangent space; downstream losses (NLL on hypothesis weight, MSE on flow,
    diffusion loss in stage 3) backprop through the corrected (ω', p') and
    therefore through this function. Both gradients must be non-zero for the
    revolute case where flow depends on both."""
    theta = math.pi / 4
    c, s = math.cos(theta), math.sin(theta)
    coords = torch.tensor([[c, s, 0.0], [0.7 * c, 0.7 * s, 0.4]])
    omega = torch.tensor([0.0, 0.0, 1.0], requires_grad=True)
    p = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)

    flow = analytical_flow_diff(
        coords, omega, p, JOINT_TYPE_REVOLUTE, normalize_per_part=False
    )
    # Arbitrary scalar that exercises every flow component.
    flow.pow(2).sum().backward()

    assert omega.grad is not None, "omega.grad should be populated"
    assert p.grad is not None, "p.grad should be populated"
    assert omega.grad.abs().sum().item() > 0.0
    assert p.grad.abs().sum().item() > 0.0
