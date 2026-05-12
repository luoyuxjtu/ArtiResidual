"""Stage 3 — Joint fine-tune of refiner + policy.

See artiresidual_tech_spec.md §4.8 (joint loss) + §6.

Total loss (spec §4.8):
    L = L_diffusion + λ_refiner · (L_NLL + L_mu_residual + λ_H · (-H(w)))
        + λ_belief · L_belief_consistency
    with λ_refiner = 0.1, λ_H = 0.01, λ_belief = 0.05.

Both networks unfrozen but with a low refiner-loss weight so the policy
remains the dominant gradient sink. The belief-consistency term keeps
f_cond aligned with the learned f_pred (Module 02).
"""
from __future__ import annotations

__all__: list[str] = []
