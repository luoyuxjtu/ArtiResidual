"""Stage 1 — Train the IMM articulation refiner with the policy frozen.

See artiresidual_tech_spec.md §3 Module 04 + §4.9-§4.10 (L_NLL, L_mu_residual)
+ §6 (implementation order).

Hydra entry point (intended):
    python -m artiresidual.training.train_refiner --config-name=base

Training data:
    Perturbed expert demos from Module 11 (12 tasks × 100 demos × 5 replays).
    Each replay carries: wrong initial (omega, p, type), the unchanged ground-
    truth trajectory, observed wrenches and pcds. The refiner is supervised to
    (a) converge its top weight to the ground-truth hypothesis and (b) reduce
    its omega/p residual.

Loss (spec §4.8 with policy disabled):
    L = L_NLL + L_mu_residual + λ_H · (-H(w))

wandb panels to watch (once implemented):
    - loss/total, loss/nll, loss/mu_residual, loss/entropy
    - refine/top1_acc, refine/omega_err_deg, refine/p_err_cm
    - refine/weights_histogram (entropy over training)
"""
from __future__ import annotations

__all__: list[str] = []
