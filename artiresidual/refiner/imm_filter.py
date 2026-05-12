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

__all__: list[str] = []
