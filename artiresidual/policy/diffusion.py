"""Diffusion utilities for the ArtiResidual DiT policy.

Cosine schedule, v-prediction loss, DDIM sampler. Spec §4.7 + §3 Module 08.

The v-prediction loss (spec §4.7):
    given x_t = α_t · x_0 + σ_t · ε,
    v_target = α_t · ε - σ_t · x_0
    L_diffusion = || v_pred - v_target ||²
"""
from __future__ import annotations

__all__: list[str] = []
