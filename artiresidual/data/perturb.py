"""Module 11 — Training Perturbation.

See artiresidual_tech_spec.md §3 Module 11 for the authoritative API.

Generate perturbed-initial-estimate variants of expert demos for refiner
training (Innovation 2's training signal). The expert ACTION trajectory is
unchanged; only the initial (omega, p, type) handed to the refiner is wrong.
This trains the refiner to detect its own initial error from the resulting
residual flow + wrench signature, and correct it.

Perturbation distribution (small + medium only):
    omega:       rotate by angle in Uniform(5°, 30°) about a random S² tangent axis.
    p:           Gaussian σ=2 cm, clipped at 5 cm.
    joint_type:  swap revolute ↔ prismatic with probability 0.20.

Volume: 12 tasks × 100 demos × 5 replays = 6000 perturbation episodes.
"""
from __future__ import annotations

__all__: list[str] = []
