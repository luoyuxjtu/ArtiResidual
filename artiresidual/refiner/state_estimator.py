"""Module 06 — Joint State Estimator.

See artiresidual_tech_spec.md §3 Module 06 for the authoritative API.

At every control step, recover the scalar joint configuration ``theta_t`` from
the articulated part's current 6-DoF pose relative to its reference pose at
t=0:

    revolute   theta_t  =  signed twist angle of (q_curr · q_init⁻¹) about ω̂
                          (extracted via the twist-swing decomposition).
    prismatic  theta_t  =  (x_curr - x_init) · ω̂

For multi-hypothesis (IMM) usage, instantiate K independent estimators that
share the same ``initial_part_pose`` but carry their own (omega, p,
joint_type). See ``tests/test_state_estimator.py::test_K_hypotheses_*`` for
the canonical pattern. The IMM refiner (Module 04) creates K=3 instances at
task start.

Pure geometry, no learnable parameters. PyTorch ops are used because (a) the
refiner's Δμ residual head needs gradients to flow back through ``theta_t``
into hypothesis (ω, p) in stage-3 joint fine-tune; (b) it keeps device
handling uniform with the rest of the project.

Quaternion convention (THIS MODULE): scalar-first ``(w, x, y, z)``.
"""
from __future__ import annotations

import torch
from torch import Tensor

from artiresidual.refiner.analytical_flow import (
    JOINT_TYPE_FIXED,
    JOINT_TYPE_PRISMATIC,
    JOINT_TYPE_REVOLUTE,
    normalize_axis,
)

__all__ = ["JointStateEstimator"]


# ---------------------------------------------------------------------------
# Quaternion primitives (scalar-first (w, x, y, z) convention).
# ---------------------------------------------------------------------------


def _quat_normalize(q: Tensor, eps: float = 1e-8) -> Tensor:
    """Defensively renormalize a quaternion tensor to unit length.

    Args:
        q: [..., 4] quaternion.
        eps: floor on norm to prevent divide-by-zero on pathological input.

    Returns:
        q_unit: [..., 4] unit quaternion.
    """
    norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=eps)
    return q / norm


def _quat_conjugate(q: Tensor) -> Tensor:
    """Conjugate of a scalar-first quaternion: ``(w, -x, -y, -z)``.

    For unit quaternions, the conjugate equals the inverse.

    Args:
        q: [..., 4] in (w, x, y, z).

    Returns:
        q_conj: [..., 4].
    """
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Hamilton product ``q1 ⊗ q2`` for scalar-first quaternions.

    Args:
        q1: [..., 4] first operand.
        q2: [..., 4] second operand.

    Returns:
        product: [..., 4] q1 ⊗ q2.
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


def _signed_twist_angle(q: Tensor, axis: Tensor) -> Tensor:
    """Signed twist angle of unit quaternion ``q`` about ``axis``.

    Twist-swing decomposition: any rotation ``q`` factors as ``q_twist ⊗
    q_swing`` where ``q_twist`` rotates about ``axis`` and ``q_swing`` is
    perpendicular. The twist angle is recovered as

        theta  =  2 · atan2(v · axis,  w)

    after picking the representative with ``w ≥ 0`` (each quaternion class has
    two members, ``q`` and ``-q``; their twist angles differ by 2π, so we
    canonicalize). ``atan2`` keeps the answer well-behaved across the full
    range including near-π rotations and zero twist.

    Args:
        q: [..., 4] unit quaternion (w, x, y, z).
        axis: [..., 3] unit axis vector. Broadcastable against ``q``'s batch.

    Returns:
        theta: [...] signed angle in radians, in ``(-π, π]``.
    """
    # Pick representative with w >= 0; sign(0) → 0, which we map to +1 so that
    # an exact 180° rotation (w = 0) keeps its v component intact.
    w_sign = torch.sign(q[..., :1])
    w_sign = torch.where(w_sign == 0, torch.ones_like(w_sign), w_sign)
    q = q * w_sign

    w = q[..., 0]                       # [...]
    v = q[..., 1:]                       # [..., 3]
    v_dot_axis = (v * axis).sum(dim=-1)  # [...]
    return 2.0 * torch.atan2(v_dot_axis, w)


# ---------------------------------------------------------------------------
# Public API (spec §3 Module 06).
# ---------------------------------------------------------------------------


class JointStateEstimator:
    """Recover the scalar joint configuration ``theta_t`` from part pose.

    See artiresidual_tech_spec.md §3 Module 06.

    Each instance represents ONE articulation hypothesis (one ``(omega, p,
    joint_type)`` triple) anchored to ONE reference pose at t=0. The estimator
    is stateless past construction — every ``estimate`` call is a pure function
    of the current observation and the cached reference state.

    Multi-hypothesis (K) usage: instantiate K estimators that share the same
    ``initial_part_pose`` but have their own ``(omega, p, joint_type)``. The
    IMM refiner (Module 04) does this at task start with K=3. There is no
    state to update between calls.

    Not an ``nn.Module``: no learnable parameters. ``omega`` is stored as a
    plain tensor on the device the caller passed it on, with the caller's
    dtype, and gradients flow through it (used by the refiner's Δμ residual
    head and by stage-3 joint fine-tune).

    Output units:
        - revolute  → radians, in ``(-π, π]``.
        - prismatic → same length unit as ``initial_part_pose[..., :3]``
                      (meters in this project).
        - fixed     → zeros.
    """

    def __init__(
        self,
        omega: Tensor,
        p: Tensor,
        joint_type: int,
        initial_part_pose: Tensor,
        *,
        eps: float = 1e-8,
    ) -> None:
        """Cache the reference state at t=0.

        Args:
            omega: [3] or [B, 3] joint axis direction. Need NOT be unit norm —
                the constructor normalizes defensively. Gradients flow back
                through it.
            p: [3] or [B, 3] reference point on the joint axis. STORED but not
                used by either revolute or prismatic estimation — the math is
                independent of which point on the axis you pick. Kept in the
                signature for API completeness with the spec, and so that
                future position-augmented variants can share the constructor.
            joint_type: discrete int — one of ``JOINT_TYPE_REVOLUTE`` (0),
                ``JOINT_TYPE_PRISMATIC`` (1), ``JOINT_TYPE_FIXED`` (2). For
                soft-type mixing, use ``analytical_flow_soft`` from Module 05.
            initial_part_pose: [B, 7] = 3-pos + 4-quat ``(w, x, y, z)`` of the
                articulated part at t=0, in world frame.
            eps: numerical floor for normalization.

        Raises:
            ValueError: on unrecognized ``joint_type`` or wrong tensor shapes.
        """
        if joint_type not in (
            JOINT_TYPE_REVOLUTE,
            JOINT_TYPE_PRISMATIC,
            JOINT_TYPE_FIXED,
        ):
            raise ValueError(
                f"Unknown joint_type {joint_type}; expected one of "
                f"{{REVOLUTE={JOINT_TYPE_REVOLUTE}, "
                f"PRISMATIC={JOINT_TYPE_PRISMATIC}, "
                f"FIXED={JOINT_TYPE_FIXED}}}"
            )
        if initial_part_pose.dim() != 2 or initial_part_pose.shape[-1] != 7:
            raise ValueError(
                f"initial_part_pose must be [B, 7]; got "
                f"{tuple(initial_part_pose.shape)}"
            )

        self.joint_type: int = joint_type
        self.eps: float = eps
        self.omega: Tensor = normalize_axis(omega, eps=eps)              # [3] or [B, 3]
        self.p: Tensor = p                                                # stored, unused
        self.x_init: Tensor = initial_part_pose[..., :3]                 # [B, 3]
        self.q_init: Tensor = _quat_normalize(
            initial_part_pose[..., 3:], eps=eps
        )                                                                  # [B, 4]
        self._B: int = initial_part_pose.shape[0]

    def estimate(self, current_part_pose: Tensor) -> Tensor:
        """Estimate ``theta_t`` for the current observation.

        Args:
            current_part_pose: [B, 7] = 3-pos + 4-quat ``(w, x, y, z)`` at the
                current control step. Batch size must match the one passed at
                construction.

        Returns:
            theta_t: [B] joint configuration scalar (radians for revolute,
                meters for prismatic, zeros for fixed).

        Raises:
            ValueError: on shape mismatch.
        """
        if current_part_pose.dim() != 2 or current_part_pose.shape[-1] != 7:
            raise ValueError(
                f"current_part_pose must be [B, 7]; got "
                f"{tuple(current_part_pose.shape)}"
            )
        if current_part_pose.shape[0] != self._B:
            raise ValueError(
                f"batch mismatch: current_part_pose has B="
                f"{current_part_pose.shape[0]}, but initial_part_pose had B="
                f"{self._B}"
            )

        if self.joint_type == JOINT_TYPE_FIXED:
            return torch.zeros(
                self._B,
                device=current_part_pose.device,
                dtype=current_part_pose.dtype,
            )

        if self.joint_type == JOINT_TYPE_PRISMATIC:
            x_curr = current_part_pose[..., :3]               # [B, 3]
            delta = x_curr - self.x_init                       # [B, 3]
            return (delta * self.omega).sum(dim=-1)            # [B]

        # JOINT_TYPE_REVOLUTE: signed twist angle of the relative rotation.
        q_curr = _quat_normalize(
            current_part_pose[..., 3:], eps=self.eps
        )                                                       # [B, 4]
        q_init_inv = _quat_conjugate(self.q_init)              # [B, 4]
        q_rel = _quat_multiply(q_curr, q_init_inv)             # [B, 4]
        return _signed_twist_angle(q_rel, self.omega)          # [B]
