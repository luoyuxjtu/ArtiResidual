"""Pure analytical articulation flow computation.

This module computes per-point velocity flow induced by single-DoF articulation,
given joint axis omega, axis reference point p, joint type, and joint configuration theta.

The math (matching FlowBot3D convention):

    revolute joint:    f(x; omega, p, theta) = omega x (x - p)
    prismatic joint:   f(x; omega, p, theta) = omega
    fixed:             f(x; omega, p, theta) = 0

Output is normalized per-part to unit max magnitude (max over points), so that the
flow magnitude reflects "direction at unit joint motion" not "actual motion magnitude".

This file contains NO learnable parameters. It is used:
    1. During training data generation, to compute ground-truth flow from sim URDF.
    2. During run-time self-correction loop, every N steps, to compute f_analytical
       per hypothesis from (ω_k, p_k, type_k, θ_t_estimated).
    3. During training, to provide analytical signal that learned flow is regressed against.

For *differentiable* versions (gradient through omega/p) and *soft type* versions
(gradient through type logits), see `affordance_utils.py`.

References:
    - Eisner et al., FlowBot3D, RSS 2022.
    - Wang et al., ArticuBot, RSS 2025.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

# Joint type codes - keep these consistent across the entire project.
JOINT_TYPE_REVOLUTE = 0
JOINT_TYPE_PRISMATIC = 1
JOINT_TYPE_FIXED = 2

JointType = Literal[0, 1, 2]


def normalize_axis(omega: Tensor, eps: float = 1e-8) -> Tensor:
    """Project omega to the unit sphere.

    Args:
        omega: [..., 3] axis direction (not necessarily unit norm).
        eps: floor on norm to avoid div-by-zero.

    Returns:
        omega_unit: [..., 3] unit vector.
    """
    norm = torch.linalg.norm(omega, dim=-1, keepdim=True).clamp(min=eps)
    return omega / norm


def analytical_flow(
    coords_xyz: Tensor,
    omega: Tensor,
    p: Tensor,
    joint_type: int,
    *,
    normalize_per_part: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Compute per-point analytical articulation flow for a single rigid part.

    Args:
        coords_xyz: [N, 3] point positions on the part, in world frame.
        omega: [3] joint axis direction (will be normalized internally).
        p: [3] reference point on the joint axis, in world frame.
        joint_type: one of JOINT_TYPE_REVOLUTE / JOINT_TYPE_PRISMATIC / JOINT_TYPE_FIXED.
        normalize_per_part: if True, divide flow by max per-point norm so output has
            unit max magnitude. This matches FlowBot3D convention.
        eps: floor on max-norm during normalization.

    Returns:
        flow: [N, 3] per-point velocity flow under unit joint motion.
            For revolute: cross-product field around axis (omega normalized to unit).
            For prismatic: constant field equal to omega (replicated to N points).
            For fixed: zero field.
    """
    if coords_xyz.dim() != 2 or coords_xyz.shape[-1] != 3:
        raise ValueError(f"coords_xyz must be [N, 3]; got {tuple(coords_xyz.shape)}")
    if omega.shape != (3,):
        raise ValueError(f"omega must be [3]; got {tuple(omega.shape)}")
    if p.shape != (3,):
        raise ValueError(f"p must be [3]; got {tuple(p.shape)}")

    omega_unit = normalize_axis(omega, eps=eps)  # [3]

    if joint_type == JOINT_TYPE_REVOLUTE:
        # f(x) = omega x (x - p)
        rel = coords_xyz - p.unsqueeze(0)  # [N, 3]
        flow = torch.linalg.cross(
            omega_unit.unsqueeze(0).expand_as(rel), rel, dim=-1
        )  # [N, 3]
    elif joint_type == JOINT_TYPE_PRISMATIC:
        # f(x) = omega (constant per point)
        flow = omega_unit.unsqueeze(0).expand(coords_xyz.shape[0], 3)  # [N, 3]
    elif joint_type == JOINT_TYPE_FIXED:
        flow = torch.zeros_like(coords_xyz)
    else:
        raise ValueError(
            f"Unknown joint_type {joint_type}; expected one of "
            f"{{REVOLUTE={JOINT_TYPE_REVOLUTE}, PRISMATIC={JOINT_TYPE_PRISMATIC}, "
            f"FIXED={JOINT_TYPE_FIXED}}}"
        )

    if normalize_per_part:
        max_norm = torch.linalg.norm(flow, dim=-1).max().clamp(min=eps)
        flow = flow / max_norm

    return flow


def analytical_flow_batched(
    coords_xyz: Tensor,
    omega: Tensor,
    p: Tensor,
    joint_type: Tensor,
    *,
    normalize_per_part: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Batched version of analytical_flow over K hypotheses.

    Used by the self-correction loop to compute f_analytical for all K hypotheses
    in one shot, e.g. K=3 hypotheses for the multi-hypothesis IMM filter.

    Args:
        coords_xyz: [N, 3] points (shared across hypotheses).
        omega: [K, 3] axis direction per hypothesis.
        p: [K, 3] reference point per hypothesis.
        joint_type: [K] joint type per hypothesis (long tensor, not int).
        normalize_per_part: see `analytical_flow`.
        eps: see `analytical_flow`.

    Returns:
        flows: [K, N, 3] flow per hypothesis per point.
    """
    if omega.dim() != 2 or omega.shape[-1] != 3:
        raise ValueError(f"omega must be [K, 3]; got {tuple(omega.shape)}")
    if p.shape != omega.shape:
        raise ValueError(f"p must match omega shape [K, 3]; got {tuple(p.shape)}")
    if joint_type.shape != omega.shape[:1]:
        raise ValueError(
            f"joint_type must be [K]; got {tuple(joint_type.shape)}, K={omega.shape[0]}"
        )

    K = omega.shape[0]
    flows = []
    for k in range(K):
        f_k = analytical_flow(
            coords_xyz,
            omega[k],
            p[k],
            int(joint_type[k].item()),
            normalize_per_part=normalize_per_part,
            eps=eps,
        )
        flows.append(f_k)
    return torch.stack(flows, dim=0)  # [K, N, 3]


def belief_weighted_flow(
    coords_xyz: Tensor,
    omega: Tensor,
    p: Tensor,
    joint_type: Tensor,
    weights: Tensor,
    *,
    normalize_per_part: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Compute belief-weighted analytical flow over K hypotheses.

    f_cond(x) = Σ_k w_k * f_analytical(x; omega_k, p_k, type_k)

    This is the *primary* output that gets injected into the DiT policy as
    conditioning (`f_cond` in the architecture diagram).

    Args:
        coords_xyz: [N, 3] points.
        omega: [K, 3] axis direction per hypothesis.
        p: [K, 3] axis reference point per hypothesis.
        joint_type: [K] joint type per hypothesis.
        weights: [K] hypothesis weights, must sum to 1 (will be renormalized).
        normalize_per_part: applied per-hypothesis BEFORE mixing. The mixed result
            is NOT re-normalized; this preserves the meaningful interpretation that
            a low-weight hypothesis contributes proportionally less.
        eps: see `analytical_flow`.

    Returns:
        f_cond: [N, 3] belief-weighted flow.
    """
    if weights.shape != omega.shape[:1]:
        raise ValueError(
            f"weights must be [K]; got {tuple(weights.shape)}, K={omega.shape[0]}"
        )
    weights = weights / weights.sum().clamp(min=eps)  # renormalize defensively

    flows = analytical_flow_batched(
        coords_xyz,
        omega,
        p,
        joint_type,
        normalize_per_part=normalize_per_part,
        eps=eps,
    )  # [K, N, 3]
    f_cond = (weights.view(-1, 1, 1) * flows).sum(dim=0)  # [N, 3]
    return f_cond


def constraint_directions(
    x_ee: Tensor,
    omega: Tensor,
    p: Tensor,
    joint_type: int,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Compute the predicted constraint subspace directions at the end-effector.

    Used by `wrench_features.py` to project observed wrench against the expected
    constraint geometry under each hypothesis.

    Geometry:
        revolute joint:   the constraint *moment* direction at x_ee around the joint
                         is (x_ee - p) x omega, normalized.
        prismatic joint:  the constraint *force* direction is the 2-D plane perpendicular
                         to omega; we return omega itself, callers should project the
                         observed force onto its orthogonal complement.
        fixed:           returns zeros.

    Args:
        x_ee: [3] end-effector position.
        omega: [3] axis direction (will be normalized).
        p: [3] axis reference point.
        joint_type: see JOINT_TYPE_*.
        eps: floor on norms.

    Returns:
        direction: [3] unit vector indicating the *expected motion direction* under
            this hypothesis. For revolute, this is the tangential motion direction
            of the end-effector. For prismatic, this is omega. For fixed, zeros.
    """
    omega_unit = normalize_axis(omega, eps=eps)
    if joint_type == JOINT_TYPE_REVOLUTE:
        rel = x_ee - p
        tangent = torch.linalg.cross(omega_unit, rel, dim=-1)
        norm = torch.linalg.norm(tangent).clamp(min=eps)
        return tangent / norm
    elif joint_type == JOINT_TYPE_PRISMATIC:
        return omega_unit
    elif joint_type == JOINT_TYPE_FIXED:
        return torch.zeros_like(omega)
    else:
        raise ValueError(f"Unknown joint_type {joint_type}")


# ---------------------------------------------------------------------------
# Differentiable variant (gradient through omega and p).
# ---------------------------------------------------------------------------


def analytical_flow_diff(
    coords_xyz: Tensor,
    omega: Tensor,
    p: Tensor,
    joint_type: int,
    *,
    normalize_per_part: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Differentiable analytical flow.

    Identical math to :func:`analytical_flow`, but designed so that gradients
    flow through ``omega`` and ``p``. Used in:
        - Refiner's Δμ residual head (spec §4.4): refiner outputs a tangent-space
          correction Δω, Δp; the corrected (ω', p') is fed through this and the
          loss (NLL of true hypothesis or MSE on flow) backprops through to Δμ.
        - Stage-3 joint fine-tuning: gradients from the policy's diffusion loss
          can flow through ``f_cond`` back to the refiner's hypothesis state.
          (Spec usually DETACHES this in practice; the diff variant is here for
          ablations that want gradient flow.)

    The per-part normalization uses a smooth ``logsumexp`` max so that the
    gradient is well-defined even when one point dominates the max norm.

    Args:
        coords_xyz: [N, 3] points (no gradient; observations).
        omega: [3] axis direction. Gradient flows through this.
        p: [3] axis reference point. Gradient flows through this.
        joint_type: discrete int (no gradient — for soft type, use
            :func:`analytical_flow_soft`).
        normalize_per_part: see :func:`analytical_flow`.
        eps: see :func:`analytical_flow`.

    Returns:
        flow: [N, 3] flow with gradient w.r.t. ``omega``, ``p``.
    """
    omega_unit = normalize_axis(omega, eps=eps)

    if joint_type == JOINT_TYPE_REVOLUTE:
        rel = coords_xyz - p.unsqueeze(0)  # [N, 3]
        flow = torch.linalg.cross(
            omega_unit.unsqueeze(0).expand_as(rel), rel, dim=-1
        )
    elif joint_type == JOINT_TYPE_PRISMATIC:
        flow = omega_unit.unsqueeze(0).expand(coords_xyz.shape[0], 3)
    elif joint_type == JOINT_TYPE_FIXED:
        flow = torch.zeros_like(coords_xyz)
    else:
        raise ValueError(f"Unknown joint_type {joint_type}")

    if normalize_per_part:
        norms = torch.linalg.norm(flow, dim=-1)  # [N]
        # Soft-max via logsumexp: smooth approximation of max(norms) with a
        # well-defined gradient even when a single point dominates.
        soft_max = torch.logsumexp(norms * 10.0, dim=0) / 10.0
        flow = flow / soft_max.clamp(min=eps)

    return flow


# ---------------------------------------------------------------------------
# Soft joint-type mixing (gradient through type logits).
# ---------------------------------------------------------------------------


def analytical_flow_soft(
    coords_xyz: Tensor,
    omega: Tensor,
    p: Tensor,
    joint_type_logits: Tensor,
    *,
    normalize_per_part: bool = True,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Soft-type mixed analytical flow for early-stage / ablation training.

    Computes a softmax-weighted mixture of revolute/prismatic/fixed flows so
    that gradient flows through ``joint_type_logits``. Useful in early
    pre-training where the discrete type decision should be smoothed.

    NOTE: In the v3 ArtiResidual design (spec §3 Module 04), the IMM refiner
    handles type discreteness via parallel hypotheses — each hypothesis has a
    fixed type and weights vary across hypotheses. So this soft variant is
    mostly kept for:
        - Ablations comparing IMM-style vs. soft-mix.
        - Pre-training the perception net (Module 02) where soft type can help.
        - Debugging.

    Args:
        coords_xyz: [N, 3].
        omega: [3].
        p: [3].
        joint_type_logits: [3] logits in order [revolute, prismatic, fixed].
        normalize_per_part: see :func:`analytical_flow`.
        temperature: softmax temperature; lower → harder selection.
        eps: see :func:`analytical_flow`.

    Returns:
        flow: [N, 3] soft-mixed flow, differentiable in all of
            ``(omega, p, joint_type_logits)``.
    """
    if joint_type_logits.shape != (3,):
        raise ValueError(
            f"joint_type_logits must be [3]; got {tuple(joint_type_logits.shape)}"
        )

    weights = F.softmax(joint_type_logits / temperature, dim=-1)  # [3]

    flow_rev = analytical_flow_diff(
        coords_xyz, omega, p, JOINT_TYPE_REVOLUTE,
        normalize_per_part=normalize_per_part, eps=eps,
    )
    flow_pri = analytical_flow_diff(
        coords_xyz, omega, p, JOINT_TYPE_PRISMATIC,
        normalize_per_part=normalize_per_part, eps=eps,
    )
    flow_fix = torch.zeros_like(coords_xyz)

    return weights[0] * flow_rev + weights[1] * flow_pri + weights[2] * flow_fix


# ---------------------------------------------------------------------------
# Naming convention alias.
# ---------------------------------------------------------------------------
# The "hard" variant is the canonical `analytical_flow` above; this alias makes
# the {hard, soft, diff} family explicit in call sites and matches the spec
# wording (§3 Module 05: "Three variants in prior code: hard, soft, diff").
analytical_flow_hard = analytical_flow
