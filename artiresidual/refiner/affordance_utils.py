"""Differentiable variants and training utilities for analytical articulation flow.

This module wraps `analytical_flow.py` with versions that:
    1. Allow gradients through (omega, p) — used in joint training & in the
       refiner's residual head where Δμ updates ω/p in tangent space.
    2. Allow soft mixing across joint types via softmax over logits — used in
       early-stage training where the type discrete decision is not yet locked in.
    3. Compute consistency losses between learned per-point flow and analytical flow,
       used to train the perception network and to regularize the refiner.
    4. Compute residual flow Δ_flow = f_predicted - f_cond, the central evidence
       signal of the self-correction loop.

It also provides a few small numerical utilities used across training:
    - tangent-space exponential map for ω updates
    - cosine-similarity loss (with sign disambiguation)
    - entropy of hypothesis weights (used as policy meta-conditioning + regularizer)

This file CONTAINS NO LEARNABLE PARAMETERS. All operations are pure functions of
their inputs, with autograd-traceable gradients.

Conventions:
    - All tensors expected on the same device.
    - All "[..., 3]" tensors use xyz ordering.
    - Batch dimensions are explicit; we never silently squeeze.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .analytical_flow import (
    JOINT_TYPE_FIXED,
    JOINT_TYPE_PRISMATIC,
    JOINT_TYPE_REVOLUTE,
    normalize_axis,
)


# ---------------------------------------------------------------------------
# Differentiable analytical flow (gradient through omega and p).
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

    Identical math to `analytical_flow.analytical_flow`, but designed so that
    gradients flow through `omega` and `p`. Used in:
        - Refiner's Δμ residual head: refiner outputs a tangent-space correction
          Δω, Δp; the corrected (ω', p') is fed through this function and the loss
          (NLL of true hypothesis or MSE on flow) backprops through to Δμ.
        - Stage-3 joint fine-tuning: gradients from the policy's diffusion loss
          flow through f_cond back to the refiner's hypothesis state. (Note: see
          §4.1 of the tech spec for stop-gradient policy; this gradient is usually
          DETACHED in practice. The differentiable version is here in case you
          want to enable it in ablations.)

    Args:
        coords_xyz: [N, 3] points (gradient does NOT flow back; coords are observations).
        omega: [3] axis direction. Gradient flows through this.
        p: [3] axis reference point. Gradient flows through this.
        joint_type: discrete int (no gradient — for soft type, use `analytical_flow_soft`).
        normalize_per_part: see analytical_flow.
        eps: see analytical_flow.

    Returns:
        flow: [N, 3] flow with gradient w.r.t. omega, p.
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
        # Use a soft-max-norm via logsumexp for differentiability.
        # When evaluated at training time on revolute parts where one point is far
        # from p, hard-max gives non-differentiable kinks; soft-max-norm avoids this.
        norms = torch.linalg.norm(flow, dim=-1)  # [N]
        # Soft max via log-sum-exp; smooth approximation of max.
        soft_max = torch.logsumexp(norms * 10.0, dim=0) / 10.0  # smoothed max
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
    """Soft-type mixed analytical flow for early-stage training.

    Computes a softmax-weighted mixture of revolute/prismatic/fixed flows, allowing
    gradient to flow through `joint_type_logits`. Useful in early training stages
    where the discrete type decision should be smoothed.

    NOTE: For the v3 ArtiResidual design, the IMM refiner handles type discreteness
    via parallel hypotheses (each hypothesis has a fixed type, weights vary across
    hypotheses). So `analytical_flow_soft` is mainly a *legacy* utility kept for:
        - Ablations comparing IMM vs soft-mix
        - Early prototyping / debugging
        - Pre-training perception net where soft type may help

    Args:
        coords_xyz: [N, 3].
        omega: [3].
        p: [3].
        joint_type_logits: [3] logits for [revolute, prismatic, fixed].
        normalize_per_part: see analytical_flow.
        temperature: softmax temperature; lower = harder.
        eps: see analytical_flow.

    Returns:
        flow: [N, 3] soft-mixed flow with gradient w.r.t. all of (omega, p, logits).
    """
    if joint_type_logits.shape != (3,):
        raise ValueError(
            f"joint_type_logits must be [3]; got {tuple(joint_type_logits.shape)}"
        )

    weights = F.softmax(joint_type_logits / temperature, dim=-1)  # [3]

    # Compute three flows independently. Each is differentiable in (omega, p).
    flow_rev = analytical_flow_diff(
        coords_xyz, omega, p, JOINT_TYPE_REVOLUTE,
        normalize_per_part=normalize_per_part, eps=eps,
    )
    flow_pri = analytical_flow_diff(
        coords_xyz, omega, p, JOINT_TYPE_PRISMATIC,
        normalize_per_part=normalize_per_part, eps=eps,
    )
    flow_fix = torch.zeros_like(coords_xyz)

    # Soft mix.
    flow = (
        weights[0] * flow_rev
        + weights[1] * flow_pri
        + weights[2] * flow_fix
    )
    return flow


# ---------------------------------------------------------------------------
# Tangent-space updates for ω (used by refiner Δμ).
# ---------------------------------------------------------------------------


def exp_map_sphere(
    omega: Tensor, delta_omega: Tensor, *, eps: float = 1e-8
) -> Tensor:
    """Exponential map on the unit sphere S² for axis updates.

    Given current axis omega ∈ S² and a tangent vector delta_omega ∈ T_omega(S²),
    returns the rotated axis omega' ∈ S² along the geodesic.

    This is used by the refiner's residual head to apply corrections Δω with the
    correct manifold structure (vs. naive omega + delta which would leave S²).

    Args:
        omega: [3] unit vector. (Will be normalized defensively.)
        delta_omega: [3] tangent vector (need not be exactly orthogonal to omega;
            the component parallel to omega is automatically projected out).
        eps: floor for stability.

    Returns:
        omega_new: [3] unit vector.
    """
    omega_unit = normalize_axis(omega, eps=eps)

    # Project delta_omega onto the tangent plane at omega.
    parallel = (delta_omega * omega_unit).sum(dim=-1, keepdim=True) * omega_unit
    tangent = delta_omega - parallel
    angle = torch.linalg.norm(tangent).clamp(min=eps)

    if angle < eps:
        return omega_unit  # zero update

    direction = tangent / angle
    omega_new = torch.cos(angle) * omega_unit + torch.sin(angle) * direction
    # Defensive renormalize against accumulating numerical drift.
    return normalize_axis(omega_new, eps=eps)


def clip_axis_correction(
    delta_omega: Tensor,
    max_angle_rad: float,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Clip a tangent-space axis correction to a max angular magnitude.

    Used to enforce the "small-medium perturbation" regime: refiner corrections
    are clipped to ≤30° per update step, so K=3 hypotheses cover the gross
    discrete decisions while the residual head only does fine-tuning within each.

    Args:
        delta_omega: [3] tangent vector (assumed already projected to tangent plane,
            but we don't enforce that here).
        max_angle_rad: maximum allowed angular magnitude in radians.
        eps: floor.

    Returns:
        clipped: [3] tangent vector with norm ≤ max_angle_rad.
    """
    norm = torch.linalg.norm(delta_omega).clamp(min=eps)
    if norm <= max_angle_rad:
        return delta_omega
    return delta_omega * (max_angle_rad / norm)


def clip_position_correction(
    delta_p: Tensor,
    max_dist_m: float,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Clip a position correction to a max distance in meters.

    Used for refiner Δp; defaults in the spec are max 5 cm per update step.
    """
    norm = torch.linalg.norm(delta_p).clamp(min=eps)
    if norm <= max_dist_m:
        return delta_p
    return delta_p * (max_dist_m / norm)


# ---------------------------------------------------------------------------
# Consistency losses (compare learned flow with analytical flow).
# ---------------------------------------------------------------------------


def cosine_similarity_loss(
    flow_pred: Tensor,
    flow_target: Tensor,
    *,
    eps: float = 1e-8,
    sign_invariant: bool = False,
) -> Tensor:
    """Cosine similarity loss between predicted and target flows.

    Args:
        flow_pred: [..., 3] predicted flow.
        flow_target: [..., 3] target flow (e.g., analytical GT or another hypothesis).
        eps: floor for numerical stability.
        sign_invariant: if True, treats opposite directions as equivalent
            (loss = 1 - |cos|). Useful when the task is symmetric under articulation
            direction reversal (e.g., for revolute joints when sense of rotation is
            ambiguous from a single observation). Default False.

    Returns:
        loss: scalar mean loss in [0, 1] (or [0, 1] for sign_invariant).
    """
    if flow_pred.shape != flow_target.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(flow_pred.shape)} vs target "
            f"{tuple(flow_target.shape)}"
        )
    cos = F.cosine_similarity(flow_pred, flow_target, dim=-1, eps=eps)
    if sign_invariant:
        return (1.0 - cos.abs()).mean()
    return (1.0 - cos).mean()


def consistency_loss(
    flow_pred: Tensor,
    flow_analytical: Tensor,
    *,
    cosine_weight: float = 1.0,
    mse_weight: float = 0.1,
    sign_invariant: bool = False,
    eps: float = 1e-8,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combined consistency loss: cosine + scaled MSE.

    The dominant signal is cosine (direction); MSE is a small magnitude term
    that helps when both flows have meaningful magnitudes (e.g., normalized to
    unit max).

    Args:
        flow_pred: [..., 3] predicted flow.
        flow_analytical: [..., 3] analytical (ground truth or hypothesis) flow.
        cosine_weight: weight on cosine term.
        mse_weight: weight on MSE term.
        sign_invariant: see `cosine_similarity_loss`.
        eps: stability.

    Returns:
        loss: scalar combined loss.
        metrics: dict with individual components for logging.
    """
    loss_cos = cosine_similarity_loss(
        flow_pred, flow_analytical, eps=eps, sign_invariant=sign_invariant
    )
    loss_mse = F.mse_loss(flow_pred, flow_analytical)
    total = cosine_weight * loss_cos + mse_weight * loss_mse
    return total, {
        "loss_cosine": loss_cos.detach(),
        "loss_mse": loss_mse.detach(),
        "loss_total": total.detach(),
    }


# ---------------------------------------------------------------------------
# Residual flow computation (the central self-correction signal).
# ---------------------------------------------------------------------------


def residual_flow(
    flow_predicted: Tensor,
    flow_belief: Tensor,
) -> Tensor:
    """Compute the residual flow Δ_flow = f_predicted - f_belief.

    This is the *central evidence signal* of the self-correction loop. It is:
        - Computed every control step.
        - Aggregated over a 30-step window before being fed to the refiner network.
        - Used (mean-pooled) as a per-step refinement scalar feature.

    Args:
        flow_predicted: [..., N, 3] perception network output.
        flow_belief: [..., N, 3] belief-weighted f_cond from `belief_weighted_flow`,
            or analytical flow from a single hypothesis.

    Returns:
        delta: [..., N, 3] residual flow.
    """
    if flow_predicted.shape != flow_belief.shape:
        raise ValueError(
            f"shape mismatch: predicted {tuple(flow_predicted.shape)} vs "
            f"belief {tuple(flow_belief.shape)}"
        )
    return flow_predicted - flow_belief


def residual_flow_summary(
    delta: Tensor, *, eps: float = 1e-8
) -> dict[str, Tensor]:
    """Compute scalar summary statistics of a residual flow tensor.

    Used for logging during the self-correction loop (e.g., to W&B every step) and
    for the refiner's auxiliary scalar features.

    Args:
        delta: [..., N, 3] residual flow tensor (last two dims are points × xyz).
        eps: floor.

    Returns:
        Dict with:
            mean_norm: scalar, mean L2 norm of per-point residual.
            max_norm:  scalar, max  L2 norm (where the disagreement is worst).
            mean_dot_with_predicted: scalar in [-1, 1], indicates whether residual
                is "aligned" (predicted flow is bigger than belief) or "opposed".
                (Reserved for future use; computed only if a valid magnitude exists.)
    """
    norms = torch.linalg.norm(delta, dim=-1)  # [..., N]
    return {
        "mean_norm": norms.mean(),
        "max_norm": norms.amax(dim=-1).mean(),
    }


# ---------------------------------------------------------------------------
# Hypothesis-set utilities (entropy, normalization).
# ---------------------------------------------------------------------------


def hypothesis_entropy(weights: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Compute the Shannon entropy of a categorical hypothesis distribution.

    Used as:
        1. Conditioning signal injected into DiT policy (the "entropy token" in
           the architecture diagram, telling the policy how (un)certain the
           current belief is).
        2. Regularization term in refiner training to prevent collapse to a
           one-hot distribution prematurely.

    Args:
        weights: [..., K] hypothesis weights (assumed to sum to ~1 along last dim).
        eps: floor inside log to avoid log(0).

    Returns:
        entropy: [...] scalar entropy in nats. For K=3 the maximum is log(3) ≈ 1.0986.
    """
    w = weights.clamp(min=eps)
    w = w / w.sum(dim=-1, keepdim=True)
    return -(w * torch.log(w)).sum(dim=-1)


def renormalize_with_floor(
    weights: Tensor, *, w_min: float = 0.05, eps: float = 1e-8
) -> Tensor:
    """Apply minimum-weight floor and renormalize, preserving the floor.

    Critical for preventing permanent hypothesis collapse: if a hypothesis weight
    drops to ~0, it can never recover via Bayesian update. Flooring at w_min keeps
    every hypothesis "alive at low priority" so it can come back if evidence later
    supports it.

    Algorithm (preserves floor exactly):
        1. Reserve K * w_min mass for the floor (must satisfy K * w_min ≤ 1).
        2. Distribute the remaining mass (1 - K * w_min) proportionally to the
           input weights.
        3. Result: w_k = w_min + (1 - K*w_min) * input_k / sum(input).

    A naive "clamp then renormalize" does NOT preserve the floor — clamping
    increases sum, then dividing by sum can push small entries below w_min again.
    This implementation is correct.

    Args:
        weights: [..., K] non-negative weights (need not sum to 1).
        w_min: minimum allowable weight per hypothesis (default 0.05).
            Must satisfy K * w_min ≤ 1 along the last dim.
        eps: floor for division stability.

    Returns:
        renormalized: [..., K] with each entry ≥ w_min and sum = 1.

    Raises:
        ValueError: if K * w_min > 1 (impossible to satisfy floor).
    """
    K = weights.shape[-1]
    if K * w_min > 1.0 + eps:
        raise ValueError(
            f"K * w_min = {K * w_min} > 1; cannot floor every hypothesis at {w_min} "
            f"with K={K} hypotheses."
        )

    w = weights.clamp(min=0.0)  # ensure non-negative
    total = w.sum(dim=-1, keepdim=True).clamp(min=eps)
    proportions = w / total  # [..., K], sums to 1

    # Reserve K * w_min for the floor; distribute (1 - K*w_min) by proportions.
    excess_mass = 1.0 - K * w_min
    return w_min + excess_mass * proportions


# ---------------------------------------------------------------------------
# Convenience: GT flow batch generator (for training data prep).
# ---------------------------------------------------------------------------


def gt_flow_from_articulation(
    coords_xyz_per_part: Sequence[Tensor],
    omega_per_part: Sequence[Tensor],
    p_per_part: Sequence[Tensor],
    joint_type_per_part: Sequence[int],
    *,
    normalize_per_part: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Compute ground-truth per-point flow for an articulated object with multiple parts.

    Used during training data generation: given a URDF with K articulated parts,
    each with its own (ω, p, type) and the points belonging to that part, produce
    a single concatenated [N_total, 3] flow tensor that can be regressed by the
    perception network.

    For points belonging to a part, the flow is the analytical flow of that joint.
    Points are assumed to be already split by part membership.

    Args:
        coords_xyz_per_part: list of [N_i, 3] tensors, one per part.
        omega_per_part: list of [3] axis tensors per part.
        p_per_part: list of [3] reference point tensors per part.
        joint_type_per_part: list of int joint types per part.
        normalize_per_part: see analytical_flow. Applied independently per part.
        eps: see analytical_flow.

    Returns:
        flow: [sum(N_i), 3] concatenated flow tensor in part order.
    """
    n_parts = len(coords_xyz_per_part)
    if not (
        len(omega_per_part) == n_parts
        and len(p_per_part) == n_parts
        and len(joint_type_per_part) == n_parts
    ):
        raise ValueError("All per-part lists must have the same length")

    # Use the non-differentiable version for GT generation (no grad needed).
    from .analytical_flow import analytical_flow

    flows = []
    for i in range(n_parts):
        f_i = analytical_flow(
            coords_xyz_per_part[i],
            omega_per_part[i],
            p_per_part[i],
            joint_type_per_part[i],
            normalize_per_part=normalize_per_part,
            eps=eps,
        )
        flows.append(f_i)
    return torch.cat(flows, dim=0)
