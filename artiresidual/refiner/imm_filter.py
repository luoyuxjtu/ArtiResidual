"""Module 04 ★ — IMM Articulation Refiner (CORE MODULE).

See artiresidual_tech_spec.md §3 Module 04 for the authoritative API and
acceptance test. This is the paper's central technical contribution.

What it does:
    Maintains K=3 parallel hypotheses about (omega, p, joint_type) for an
    articulated part. Every N=10 control steps, it refines the hypotheses
    using a window of recent evidence — predicted vs. analytical flow residual
    plus wrench feedback — and produces:
        f_cond   = Σ_k w_k · f_ana(omega_k, p_k, type_k, theta_t)
        H(w)     entropy over hypothesis weights
    Both are consumed by the DiT policy (Module 08) as conditioning so that
    the policy is uncertainty-aware.

Update rule (every N control steps, see spec §4.3):
    ℓ_k       = NeuralLogLikelihood(window, hypothesis_k)
    w_k_new   ∝ w_k · exp(ℓ_k)
    Δμ_k      = (Δω_k, Δp_k)  in tangent space, applied via exp-map with
                clip 30° (ω) and 5 cm (p), learning rate η=0.5.

Loss terms during Stage-1 training (refiner-only, policy frozen):
    L_NLL          = -log(w_k*)              ground-truth hypothesis selection
    L_mu_residual  = Σ_k w_k · (1 - cos(ω_k, ω*)) + λ_p · ||p_k - p*||²
    L_H            = -λ_H · H(w)             entropy regularizer (λ_H = 0.01)

DO NOT in v1: hypothesis spawning / pruning. K is fixed at 3 throughout.

References:
    - Differentiable filters: https://github.com/akloss/differentiable_filters
    - IMM classical:          Blom & Bar-Shalom 1988
"""
from __future__ import annotations

import math
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from artiresidual.refiner.affordance_utils import (
    hypothesis_entropy,
    renormalize_with_floor,
)
from artiresidual.refiner.analytical_flow import (
    JOINT_TYPE_FIXED,
    JOINT_TYPE_PRISMATIC,
    JOINT_TYPE_REVOLUTE,
)

__all__ = ["IMMArticulationRefiner"]


# ---------------------------------------------------------------------------
# Private sub-modules
# ---------------------------------------------------------------------------


class _PointNetMini(nn.Module):
    """3-layer per-point MLP + global mean-pool (no max-pool, no local grouping).

    Spec §3 Module 04: PointNet-mini (3 layers, 64→128→256) → mean-pool → dim.
    Shared weights across K hypotheses and across T window steps; callers are
    responsible for batching along those dimensions before calling forward.

    Args:
        in_dim: input feature dimension per point (default 3 for xyz flow).
        out_dim: global feature dimension after pooling (default 256).
    """

    def __init__(self, in_dim: int = 3, out_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [..., N, in_dim] per-point features.

        Returns:
            [..., out_dim] global feature via mean-pool over the N dimension.
        """
        feat = self.mlp(x)     # [..., N, out_dim]
        return feat.mean(dim=-2)  # [..., out_dim]


# ---------------------------------------------------------------------------
# Batched geometry helpers (module-private, used only by step())
# ---------------------------------------------------------------------------


def _batched_clip_norm(v: Tensor, max_norm: float, eps: float = 1e-8) -> Tensor:
    """Clip each vector in a batch so its L² norm ≤ max_norm.

    Works on any leading batch shape [..., 3].
    Vectors already within the bound are returned unchanged (no scaling).

    Args:
        v: [..., 3] batch of vectors.
        max_norm: maximum allowed norm.
        eps: numerical floor on computed norm.

    Returns:
        [..., 3] clipped batch, with ‖v[...]‖ ≤ max_norm.
    """
    norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)  # [..., 1]
    scale = (max_norm / norm).clamp(max=1.0)            # [..., 1] ≤ 1
    return v * scale


def _batched_exp_map_sphere(
    omega: Tensor, delta: Tensor, eps: float = 1e-8
) -> Tensor:
    """Batched exponential map on S² (spec §4.4 formula, vectorized).

    Given current axes ``omega`` ∈ S² and tangent corrections ``delta``,
    returns the rotated axes ``omega_new`` ∈ S².  The formula is:

        δ_tan = delta − (delta · ω̂) ω̂          # project onto tangent plane
        ω_new = ω̂ cos(‖δ_tan‖) + (δ_tan/‖δ_tan‖) sin(‖δ_tan‖)

    ``delta`` need not be pre-projected onto the tangent plane; the parallel
    component is removed here.  The output is renormalized defensively.

    Args:
        omega: [..., 3] current unit axes on S² (will be renormalized).
        delta: [..., 3] tangent-space corrections (already clipped to cone).
        eps: numerical floor.

    Returns:
        [..., 3] updated unit axes on S².
    """
    omega_unit = omega / omega.norm(dim=-1, keepdim=True).clamp(min=eps)

    # Project delta onto the tangent plane at omega (remove parallel component).
    dot = (delta * omega_unit).sum(dim=-1, keepdim=True)  # [..., 1]
    tangent = delta - dot * omega_unit                     # [..., 3]  ⊥ omega

    angle = tangent.norm(dim=-1, keepdim=True).clamp(min=eps)  # [..., 1]
    direction = tangent / angle                                 # [..., 3] unit

    # Rodrigues: rotate omega toward direction by angle radians.
    omega_new = torch.cos(angle) * omega_unit + torch.sin(angle) * direction
    return omega_new / omega_new.norm(dim=-1, keepdim=True).clamp(min=eps)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class IMMArticulationRefiner(nn.Module):
    """IMM Articulation Refiner — paper's central technical contribution.

    Maintains K parallel hypotheses about (omega, p, joint_type) and refines
    them every N control steps. See module docstring for full description.

    Architecture (spec §3 Module 04):
        1. Per-step feature encoding:
             flow_encoder  : [B*K*T, N, 3] delta_flow → [B*K*T, dim]
             wrench_mlp    : [B*K*T, 15]   (wrench 12 + wrench_res 3) → [B*K*T, 64]
             action_mlp    : [B*T,   14]   action → [B*T, 64]   (shared across K)
             step_proj     : [B*K*T, dim+128] concat → [B*K*T, dim]
        2. Per-hypothesis transformer encoder (shared weights across K):
             hyp_encoder   : [B*K, T, dim] → [B*K, T, dim] → mean-pool → [B*K, dim]
        3. Cross-hypothesis attention (K tokens see each other):
             cross_hyp_attn: [B, K, dim] → [B, K, dim]
        4. Output heads per hypothesis:
             ll_head       : [B, K, dim] → [B, K]   log-likelihood score ℓ_k
             residual_head : [B, K, dim] → [B, K, 6] tangent-space Δω(3) + Δp(3)

    Args:
        config: Hydra DictConfig (or any object with attribute access) with
            fields: K, window_T, update_interval_N, dim, n_heads, n_layers,
            w_min, lambda_H, eta, omega_clip_deg, p_clip_m.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()

        # --- Scalar hyperparameters -------------------------------------------
        self.K: int = int(config.K)                           # number of hypotheses
        self.window_T: int = int(config.window_T)             # evidence window length
        self.N_update: int = int(config.update_interval_N)   # call step() every N steps
        self.w_min: float = float(config.w_min)              # weight floor (0.05)
        self.eta: float = float(config.eta)                  # residual learning rate (0.5)
        self.omega_clip_rad: float = math.radians(float(config.omega_clip_deg))  # 30° → rad
        self.p_clip_m: float = float(config.p_clip_m)        # 5 cm
        self.lambda_H: float = float(config.lambda_H)        # entropy regularization weight

        dim: int = int(config.dim)            # 256 — transformer model width
        n_heads: int = int(config.n_heads)   # 4
        n_layers: int = int(config.n_layers) # 4
        self.dim: int = dim                   # stored for use in step()

        # --- Per-step flow encoder (PointNet-mini, shared across K and T) ----
        # Encodes the per-point flow residual Δ_flow_k = f_pred - f_ana(hyp_k)
        # into a global 256-dim token representing "how well does hypothesis k
        # predict the observed flow at this step".
        # Call-time shape: [B*K*T, N, 3] → [B*K*T, dim]
        self.flow_encoder = _PointNetMini(in_dim=3, out_dim=dim)

        # --- Per-step wrench encoder ------------------------------------------
        # Encodes (wrench, per-hypothesis wrench residual) into a 64-dim token.
        # Wrench residual = observed wrench projected against expected constraint
        # direction for this hypothesis (constraint_directions() in Module 05).
        # Call-time shape: [B*K*T, 15] → [B*K*T, 64]
        self.wrench_mlp = nn.Sequential(
            nn.Linear(12 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # --- Per-step action encoder ------------------------------------------
        # Encodes the bimanual joint command (14-dim) into 64 features.
        # Shared across K hypotheses; broadcasted at concat time.
        # Call-time shape: [B*T, 14] → [B*T, 64]
        self.action_mlp = nn.Sequential(
            nn.Linear(14, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # --- Step-feature projection: (dim + 64 + 64) → dim ------------------
        # After concatenating flow(dim=256) + wrench(64) + action(64) = 384,
        # project down to dim so the per-hypothesis transformer sees dim-wide tokens.
        # Call-time shape: [B*K*T, 384] → [B*K*T, dim]
        self.step_proj = nn.Linear(dim + 64 + 64, dim)

        # --- Per-hypothesis transformer encoder (shared weights across K) -----
        # Processes the T-step window for one hypothesis; shared weights mean
        # all K hypotheses go through identical learned dynamics.
        # Call-time shape (batch_first=True): [B*K, T, dim] → [B*K, T, dim]
        # then mean-pooled over T → [B*K, dim] hypothesis summary token.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability (Wang et al. 2022).
        )
        self.hyp_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # --- Cross-hypothesis attention (1 layer, IMM mode-mixing) ------------
        # Lets K hypothesis tokens attend to each other so the refiner can
        # avoid collapsing two hypotheses onto identical corrections when evidence
        # is ambiguous.
        # Call-time shape: [B, K, dim] → [B, K, dim]
        cross_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.cross_hyp_attn = nn.TransformerEncoder(cross_layer, num_layers=1)

        # --- Output heads (applied per hypothesis after cross-attention) ------
        # Log-likelihood score: drives the IMM weight update (spec §4.3).
        # [B, K, dim] → [B, K, 1] → squeeze → [B, K]
        self.ll_head = nn.Linear(dim, 1)

        # Tangent-space residual: Δω ∈ ℝ³ (tangent of S² at ωₖ) and Δp ∈ ℝ³.
        # Applied via exp_map_sphere (ω) and clipped translation (p) with η=0.5.
        # [B, K, dim] → [B, K, 6]
        self.residual_head = nn.Linear(dim, 6)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize output heads so the refiner starts near identity.

        Near-zero initialization means: at the start of training,
        - All log-likelihoods ≈ 0  → equal weights among hypotheses (max entropy).
        - All residuals ≈ 0        → no correction applied (identity update).
        This prevents early destructive hypothesis collapse before the encoder
        has learned anything useful.
        """
        nn.init.zeros_(self.ll_head.bias)
        nn.init.normal_(self.ll_head.weight, std=0.01)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.normal_(self.residual_head.weight, std=0.01)

    # -------------------------------------------------------------------------
    # Public API (stubs — implemented in subsequent sessions)
    # -------------------------------------------------------------------------

    def init_hypotheses(self, prior_output: dict) -> dict:
        """Initialize K=3 hypotheses from Module 01 prior estimator output.

        Fixed hypothesis structure (spec §3 Module 04, v1 — DO NOT change order):
            h1 (index 0): revolute,  axis  = omega from prior  (nominally vertical)
            h2 (index 1): revolute,  axis  = omega ⊥ prior     (nominally horizontal)
            h3 (index 2): prismatic, axis  = omega from prior

        Weight rule: the hypothesis whose joint-type matches the prior's
        top-1 prediction receives 0.6; the remaining K-1 hypotheses each get
        0.2, so weights always sum to 1.  Resolving ambiguity within the
        revolute family (h1 vs h2) is left to the first ``step()`` call.

        Args:
            prior_output: dict from Module 01 (PriorArticulationEstimator):
                - 'omega':      [B, 3]  predicted axis direction (need not be unit).
                - 'p':          [B, 3]  predicted axis origin.
                - 'joint_type': [B]     predicted joint type long (0=rev, 1=pris).
                - 'confidence': [B]     not used here; reserved for future weighting.

        Returns:
            hypotheses: dict with
                - 'omega_k': [B, K, 3]  unit axis per hypothesis.
                - 'p_k':     [B, K, 3]  axis origin per hypothesis.
                - 'type_k':  [B, K]     long: 0=revolute, 1=prismatic.
                - 'w_k':     [B, K]     non-negative, sum to 1 per sample.
        """
        if self.K != 3:
            raise ValueError(
                f"init_hypotheses is hardcoded for K=3; got self.K={self.K}. "
                "The three hypothesis types (rev-vertical, rev-horizontal, prismatic) "
                "are a v1 design constant (spec §3 Module 04)."
            )

        omega = prior_output["omega"]          # [B, 3]
        p = prior_output["p"]                  # [B, 3]
        joint_type = prior_output["joint_type"]  # [B] long

        B = omega.shape[0]
        device = omega.device
        dtype = omega.dtype

        # Normalize prior axis.
        omega_unit = omega / omega.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, 3]

        # h2 axis: any unit vector perpendicular to omega_unit.
        # Use cross(omega, z_hat) with a y_hat fallback when omega ≈ ±z.
        z_hat = omega_unit.new_zeros(B, 3); z_hat[:, 2] = 1.0  # [B, 3]
        perp = torch.linalg.cross(omega_unit, z_hat, dim=-1)    # [B, 3]
        near_z = perp.norm(dim=-1, keepdim=True) < 0.1          # [B, 1] bool

        y_hat = omega_unit.new_zeros(B, 3); y_hat[:, 1] = 1.0   # [B, 3]
        perp_y = torch.linalg.cross(omega_unit, y_hat, dim=-1)  # [B, 3]

        perp = torch.where(near_z, perp_y, perp)                # [B, 3]
        omega2 = perp / perp.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, 3] unit

        # Stack K=3 hypothesis axes; clone so returned tensors own their storage.
        omega_k = torch.stack([omega_unit, omega2, omega_unit], dim=1)  # [B, K, 3]
        p_k = p.unsqueeze(1).expand(-1, self.K, -1).clone()             # [B, K, 3]

        # Fixed type assignment: h1=revolute, h2=revolute, h3=prismatic.
        type_k = torch.zeros(B, self.K, dtype=torch.long, device=device)
        type_k[:, 2] = JOINT_TYPE_PRISMATIC

        # Weight assignment: hypothesis matching prior's type gets 0.6; others 0.2.
        # Revolute prior → top index 0 (h1).  Prismatic prior → top index 2 (h3).
        is_revolute = joint_type == JOINT_TYPE_REVOLUTE  # [B] bool
        top_idx = torch.where(
            is_revolute,
            torch.zeros(B, dtype=torch.long, device=device),    # h1
            torch.full((B,), 2, dtype=torch.long, device=device),  # h3
        )  # [B]

        w_k = torch.full((B, self.K), 0.2, dtype=dtype, device=device)
        w_k.scatter_(1, top_idx.unsqueeze(1), 0.6)  # top hypothesis → 0.6

        return {
            "omega_k": omega_k,  # [B, K, 3]
            "p_k": p_k,          # [B, K, 3]
            "type_k": type_k,    # [B, K] long
            "w_k": w_k,          # [B, K] sums to 1.0 per sample
        }

    def step(
        self,
        hypotheses: dict,
        window: dict,
    ) -> dict:
        """Refine hypotheses given the last T steps of evidence.

        Called every N=10 control steps by the outer rollout loop.

        **Frame convention (DECISION 2026-05-14)**:
            All part poses and point clouds flowing through ``window`` must be
            in the **world frame**. The caller is responsible for applying
            ``artiresidual.utils.geometry.transform_poses(poses, cam_to_world)``
            before packing them into the window dict. This function does NOT
            silently accept camera-frame input.

        Update sequence (spec §4.3 + §4.4):
            1. Encode per-step features over the T-step window.
            2. Run per-hypothesis transformer encoder → K summary tokens.
            3. Cross-hypothesis attention → mode-mixing.
            4. ll_head  → ℓ_k → weight update:
                   w_k_new ∝ w_k · exp(ℓ_k), with floor w_min.
            5. residual_head → Δω_k, Δp_k → apply via exp_map_sphere (ω)
                   and clipped translation (p) with learning rate η.
            6. type_k is NEVER changed by the refiner (v1: discrete, immutable).

        Args:
            hypotheses: dict (same layout as ``init_hypotheses`` output or a
                prior ``step`` return):
                - 'omega_k': [B, K, 3]
                - 'p_k':     [B, K, 3]
                - 'type_k':  [B, K]     long
                - 'w_k':     [B, K]
            window: dict with T steps of evidence (all in **world frame**):
                - 'delta_flow_k': [B, K, T, N, 3]   Δflow = f_pred − f_ana(hyp_k)
                - 'wrench':       [B, T, 12]          observed bimanual wrench
                - 'wrench_res_k': [B, K, T, 3]        per-hyp wrench residual dir
                - 'action':       [B, T, 14]           bimanual joint commands

        Returns:
            hypotheses_new: dict with same keys as ``hypotheses``;
                (omega_k, p_k, w_k) updated; type_k unchanged.
        """
        omega_k = hypotheses["omega_k"]      # [B, K, 3]
        p_k     = hypotheses["p_k"]          # [B, K, 3]
        type_k  = hypotheses["type_k"]       # [B, K] long — never mutated
        w_k     = hypotheses["w_k"]          # [B, K]

        delta_flow_k = window["delta_flow_k"]  # [B, K, T, N, 3]
        wrench       = window["wrench"]         # [B, T, 12]
        wrench_res_k = window["wrench_res_k"]  # [B, K, T, 3]
        action       = window["action"]         # [B, T, 14]

        B, K, T, N, _ = delta_flow_k.shape

        # ── 1. Per-step feature encoding ──────────────────────────────────────
        # Flow residual: PointNet-mini per (batch, hypothesis, step).
        # [B, K, T, N, 3] → [B*K*T, N, 3] → [B*K*T, dim]
        flow_feat = self.flow_encoder(
            delta_flow_k.reshape(B * K * T, N, 3)
        )  # [B*K*T, dim]

        # Wrench + per-hypothesis wrench residual.
        # wrench [B, T, 12] → broadcast over K → [B, K, T, 12] → [B*K*T, 12]
        wrench_bkt = wrench.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K * T, 12)
        wrench_res = wrench_res_k.reshape(B * K * T, 3)
        wrench_feat = self.wrench_mlp(
            torch.cat([wrench_bkt, wrench_res], dim=-1)
        )  # [B*K*T, 64]

        # Action: [B, T, 14] → broadcast over K → [B*K*T, 14] → [B*K*T, 64]
        action_feat = self.action_mlp(
            action.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K * T, 14)
        )  # [B*K*T, 64]

        # Concat (dim+64+64=384) → step_proj → dim; reshape for transformer.
        step_tokens = self.step_proj(
            torch.cat([flow_feat, wrench_feat, action_feat], dim=-1)
        ).reshape(B * K, T, self.dim)  # [B*K, T, dim]

        # ── 2. Per-hypothesis transformer encoder → mean-pool → [B, K, dim] ──
        # Shared weights across K: all K hypotheses pass through the same net.
        hyp_tokens = self.hyp_encoder(step_tokens).mean(dim=1)  # [B*K, dim]
        hyp_tokens = hyp_tokens.reshape(B, K, self.dim)          # [B, K, dim]

        # ── 3. Cross-hypothesis attention (IMM mode-mixing) ───────────────────
        hyp_tokens = self.cross_hyp_attn(hyp_tokens)  # [B, K, dim]

        # ── 4a. Weight update (spec §4.3) ─────────────────────────────────────
        # ℓ_k = NeuralLogLikelihood(window, hyp_k) from ll_head.
        ll = self.ll_head(hyp_tokens).squeeze(-1)  # [B, K] log-likelihood scores
        w_unnorm = w_k * torch.exp(ll)             # [B, K] unnorm. posterior
        # renormalize_with_floor guarantees min weight ≥ w_min after renorm.
        # (The spec's naive clamp+divide can violate the floor — see DECISIONS.md.)
        w_new = renormalize_with_floor(w_unnorm, w_min=self.w_min)  # [B, K]

        # ── 4b. Tangent-space residual application (spec §4.4) ────────────────
        delta_mu = self.residual_head(hyp_tokens)  # [B, K, 6]
        d_omega  = delta_mu[..., :3]               # [B, K, 3]  tangent correction for ω
        d_p      = delta_mu[..., 3:]               # [B, K, 3]  position correction for p

        # ω: scale by η → clip to 30° cone → exp-map back onto S².
        d_omega_clipped = _batched_clip_norm(
            self.eta * d_omega, self.omega_clip_rad
        )  # [B, K, 3], ‖·‖ ≤ 0.524 rad
        omega_new = _batched_exp_map_sphere(omega_k, d_omega_clipped)  # [B, K, 3]

        # p: scale by η → clip to 5 cm → translate.
        d_p_clipped = _batched_clip_norm(
            self.eta * d_p, self.p_clip_m
        )  # [B, K, 3], ‖·‖ ≤ 0.05 m
        p_new = p_k + d_p_clipped  # [B, K, 3]

        return {
            "omega_k": omega_new,  # [B, K, 3] unit vectors on S²
            "p_k":     p_new,      # [B, K, 3]
            "type_k":  type_k,     # [B, K] long — unchanged (v1 invariant)
            "w_k":     w_new,      # [B, K] ≥ w_min, sums to 1
        }

    def get_f_cond(
        self,
        hypotheses: dict,
        theta_t: Tensor,
        pcd: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute belief-weighted flow and entropy for the DiT policy.

        Called every control step (not just at refiner update steps) so the
        policy always has up-to-date conditioning.

        Args:
            hypotheses: dict with current hypothesis state (world frame).
            theta_t: [B]     current joint angle (rad) or displacement (m)
                             from Module 06 JointStateEstimator.
            pcd:     [B, N, 3] current point cloud in world frame.

        Returns:
            f_cond:  [B, N, 3]  Σ_k w_k · f_ana(ω_k, p_k, type_k) — the
                                belief-weighted analytical flow (spec §4.5).
                                This is the primary conditioning signal for
                                the DiT policy's cross-attention 1.
            entropy: [B]        H(w) = −Σ_k w_k log(w_k) (spec §4.6).
                                Injected as the DiT policy's "entropy token"
                                (cross-attention 2) so the policy knows how
                                uncertain the current belief is.

        Notes:
            - ``theta_t`` is accepted for API compatibility with spec §3 Module 04
              but is not used in the per-part-normalized flow computation (per-part
              max-norm normalization absorbs the δθ scalar factor — see
              ``analytical_flow.belief_weighted_flow`` docstring for context).
            - Per-hypothesis flows are max-norm-normalized **before** the belief
              weighting; the mixed result is **not** re-normalized so low-weight
              hypotheses contribute proportionally less (FlowBot3D convention).
        """
        omega_k = hypotheses["omega_k"]  # [B, K, 3]
        p_k     = hypotheses["p_k"]      # [B, K, 3]
        type_k  = hypotheses["type_k"]   # [B, K] long
        w_k     = hypotheses["w_k"]      # [B, K]

        del theta_t  # accepted for API compat; see Notes above.

        B, K, _ = omega_k.shape
        N = pcd.shape[1]

        # Normalize hypothesis axes onto S² (defensive — step() already does this).
        omega_unit = omega_k / omega_k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        # [B, K, 3]

        # Revolute flow:  f_rev[b, k, n, :] = ω[b, k] × (pcd[b, n] − p[b, k])
        # Build rel[b, k, n, :] via broadcast:  [B, 1, N, 3] − [B, K, 1, 3] = [B, K, N, 3]
        rel = pcd.unsqueeze(1) - p_k.unsqueeze(2)              # [B, K, N, 3]
        omega_bcast = omega_unit.unsqueeze(2).expand(B, K, N, 3)  # [B, K, N, 3]
        f_rev = torch.linalg.cross(omega_bcast, rel, dim=-1)   # [B, K, N, 3]

        # Prismatic flow: constant ω over N (broadcast).
        f_pri = omega_bcast                                     # [B, K, N, 3]

        # Per-(b,k) type selection.  Use unsqueeze to align with [B, K, N, 3].
        is_rev = (type_k == JOINT_TYPE_REVOLUTE).unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
        flows = torch.where(is_rev, f_rev, f_pri)               # [B, K, N, 3]

        # FIXED (if any) → zero flow.  In v1 this branch is unused because
        # init_hypotheses only emits REVOLUTE / PRISMATIC, but the guard makes
        # the function robust to externally-constructed hypothesis dicts.
        is_fix = (type_k == JOINT_TYPE_FIXED).unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
        flows = torch.where(is_fix, torch.zeros_like(flows), flows)

        # Per-hypothesis max-norm normalization (FlowBot3D convention).
        # Each (b, k) slice is scaled so that max_n ‖f[b, k, n, :]‖ = 1.
        max_norm = flows.norm(dim=-1).amax(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, K, 1]
        flows = flows / max_norm.unsqueeze(-1)                  # [B, K, N, 3]

        # Belief-weighted sum (spec §4.5).  Broadcast w_k to [B, K, 1, 1].
        f_cond = (w_k.unsqueeze(-1).unsqueeze(-1) * flows).sum(dim=1)  # [B, N, 3]

        # Entropy (spec §4.6).  ``hypothesis_entropy`` clamps weights at eps before
        # log, which is equivalent to the spec's log(w + ε) up to the constant
        # offset of log(1/(1+Kε)) ≈ 0.
        entropy = hypothesis_entropy(w_k)  # [B]

        return f_cond, entropy

    def loss_terms(
        self,
        hypotheses: dict,
        gt_omega: Tensor,
        gt_p: Tensor,
        gt_type_idx: Tensor,
        *,
        lambda_p: float = 100.0,
    ) -> dict[str, Tensor]:
        """Compute Stage-1 refiner training losses (spec §4.8–§4.10).

        Args:
            hypotheses:  dict output of ``step()`` — gradients flow through
                         (omega_k, p_k, w_k).
            gt_omega:    [B, 3]  ground-truth joint axis (need not be unit).
            gt_p:        [B, 3]  ground-truth axis origin.
            gt_type_idx: [B]     long, correct hypothesis index in {0, …, K−1}.
            lambda_p:    scaling factor for the position error term (default
                         100.0 — gives ~equal scale as cosine loss at σ_p≈0.1 m).

        Returns:
            dict with scalar tensors:
                - 'L_NLL':         −log(w_{k*}),  NLL for ground-truth hypothesis.
                - 'L_mu_residual': Σ_k w_k·[(1−cos(ω_k,ω*))+λ_p‖p_k−p*‖²].
                - 'L_H':           −λ_H·H(w),  entropy regularizer (≤ 0).
                - 'L_total':       L_NLL + L_mu_residual + L_H.
        """
        omega_k = hypotheses["omega_k"]   # [B, K, 3]
        p_k     = hypotheses["p_k"]       # [B, K, 3]
        w_k     = hypotheses["w_k"]       # [B, K]

        eps = 1e-8

        # L_NLL: -log(w_{k*}) where k* is the correct hypothesis index.
        w_gt = w_k.gather(1, gt_type_idx.unsqueeze(1)).squeeze(1)  # [B]
        L_NLL = -torch.log(w_gt.clamp(min=eps)).mean()

        # L_mu_residual: belief-weighted axis + position regression.
        gt_omega_unit = (
            gt_omega / gt_omega.norm(dim=-1, keepdim=True).clamp(min=eps)
        )  # [B, 3]
        omega_unit = (
            omega_k / omega_k.norm(dim=-1, keepdim=True).clamp(min=eps)
        )  # [B, K, 3]
        cos_k = (omega_unit * gt_omega_unit.unsqueeze(1)).sum(dim=-1)  # [B, K]
        loss_cos = 1.0 - cos_k                                          # [B, K], ∈[0,2]
        p_err_sq = (p_k - gt_p.unsqueeze(1)).pow(2).sum(dim=-1)        # [B, K]
        per_hyp = loss_cos + lambda_p * p_err_sq
        L_mu_residual = (w_k * per_hyp).sum(dim=-1).mean()

        # L_H: -λ_H·H(w).  Minimizing -H(w) = maximizing H(w) → anti-collapse.
        L_H = (-self.lambda_H * hypothesis_entropy(w_k)).mean()

        return {
            "L_NLL": L_NLL,
            "L_mu_residual": L_mu_residual,
            "L_H": L_H,
            "L_total": L_NLL + L_mu_residual + L_H,
        }

    def forward(self, hypotheses: dict, window: dict) -> dict:
        """DDP-compatible forward pass; dispatches to step(). See step() docs."""
        return self.step(hypotheses, window)
