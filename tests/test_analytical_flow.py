"""Unit tests for artiresidual.refiner.analytical_flow (Module 05).

These are CPU-only sanity checks. The full acceptance test (spec §3 Module 05)
runs against synthetic door-at-45°-about-z configurations and must match the
analytical expectation to floating-point precision.
"""
from __future__ import annotations

import torch

from artiresidual.refiner.analytical_flow import (
    JOINT_TYPE_FIXED,
    JOINT_TYPE_PRISMATIC,
    JOINT_TYPE_REVOLUTE,
    analytical_flow,
    analytical_flow_batched,
    belief_weighted_flow,
)


def test_revolute_around_z_axis_at_origin_yields_tangential_flow() -> None:
    # Point on +x axis, revolute joint = +z through origin → flow direction ∝ +y.
    coords = torch.tensor([[1.0, 0.0, 0.0]])
    omega = torch.tensor([0.0, 0.0, 1.0])
    p = torch.tensor([0.0, 0.0, 0.0])
    flow = analytical_flow(
        coords, omega, p, JOINT_TYPE_REVOLUTE, normalize_per_part=True
    )
    assert flow.shape == (1, 3)
    assert torch.allclose(flow[0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


def test_prismatic_returns_axis_direction_everywhere() -> None:
    coords = torch.randn(8, 3)
    omega = torch.tensor([0.0, 1.0, 0.0])
    p = torch.tensor([0.0, 0.0, 0.0])
    flow = analytical_flow(
        coords, omega, p, JOINT_TYPE_PRISMATIC, normalize_per_part=False
    )
    assert flow.shape == (8, 3)
    expected = omega.unsqueeze(0).expand(8, 3)
    assert torch.allclose(flow, expected)


def test_fixed_joint_is_zero_flow() -> None:
    coords = torch.randn(4, 3)
    omega = torch.tensor([1.0, 0.0, 0.0])
    p = torch.tensor([0.0, 0.0, 0.0])
    flow = analytical_flow(coords, omega, p, JOINT_TYPE_FIXED)
    assert torch.allclose(flow, torch.zeros_like(coords))


def test_batched_matches_single_for_each_hypothesis() -> None:
    coords = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    omega = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    p = torch.zeros(2, 3)
    joint_type = torch.tensor([JOINT_TYPE_REVOLUTE, JOINT_TYPE_PRISMATIC])
    batched = analytical_flow_batched(coords, omega, p, joint_type)
    assert batched.shape == (2, 2, 3)

    single_0 = analytical_flow(coords, omega[0], p[0], JOINT_TYPE_REVOLUTE)
    single_1 = analytical_flow(coords, omega[1], p[1], JOINT_TYPE_PRISMATIC)
    assert torch.allclose(batched[0], single_0)
    assert torch.allclose(batched[1], single_1)


def test_belief_weighted_flow_mixes_hypotheses_by_weight() -> None:
    coords = torch.tensor([[1.0, 0.0, 0.0]])
    omega = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    p = torch.zeros(2, 3)
    joint_type = torch.tensor([JOINT_TYPE_PRISMATIC, JOINT_TYPE_PRISMATIC])

    weights = torch.tensor([1.0, 0.0])
    f_cond = belief_weighted_flow(
        coords, omega, p, joint_type, weights, normalize_per_part=False
    )
    assert torch.allclose(f_cond[0], torch.tensor([0.0, 0.0, 1.0]))

    weights = torch.tensor([0.0, 1.0])
    f_cond = belief_weighted_flow(
        coords, omega, p, joint_type, weights, normalize_per_part=False
    )
    assert torch.allclose(f_cond[0], torch.tensor([1.0, 0.0, 0.0]))

    weights = torch.tensor([0.5, 0.5])
    f_cond = belief_weighted_flow(
        coords, omega, p, joint_type, weights, normalize_per_part=False
    )
    assert torch.allclose(f_cond[0], torch.tensor([0.5, 0.0, 0.5]))
