"""Stage 2 — Train the DiT diffusion policy with the refiner frozen.

See artiresidual_tech_spec.md §6 (implementation order) + §3 Module 08.

Hydra entry point (intended):
    python -m artiresidual.training.train_policy --config-name=base

Training data:
    Clean expert demos (Module 10). At each step, the frozen refiner produces
    f_cond + entropy from the prior estimate (Module 01) and the in-trajectory
    observations; the policy regresses actions in v-prediction with the cosine
    schedule.

wandb panels:
    - loss/diffusion_v
    - rollout/success_rate (every vis_every steps via sim eval)
    - sample/action_chunk_smoothness
"""
from __future__ import annotations

__all__: list[str] = []
