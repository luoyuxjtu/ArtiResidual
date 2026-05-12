"""Module 12 — Failure Mode Analysis.

See artiresidual_tech_spec.md §3 Module 12 for the authoritative API.

Categorizes every failed rollout (from any method on the benchmark) into:
    (a) joint axis misestimation     |estimated_omega - true_omega| > 30°
    (b) inter-arm force conflict     ||wrench|| > threshold for > 1 second
    (c) handle slip                  gripper-object distance jumps > 3 cm/step
    (d) coordination desync          stabilizer / actor out of phase

In the paper, we argue our self-correction explains improvement on (a) and
wrench feedback explains improvement on (b); we should be comparable on
(c) and (d) — failure to be comparable would be a reviewer red flag.
"""
from __future__ import annotations

__all__: list[str] = []
