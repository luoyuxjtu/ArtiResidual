"""Module 06 — Joint State Estimator.

See artiresidual_tech_spec.md §3 Module 06 for the authoritative API.

At every control step, estimate the scalar joint configuration `theta_t` from
the part's current pose relative to its t=0 reference pose:
    revolute   → project rotation about omega
    prismatic  → project translation along omega

Pure geometry, no ML. Need K instances (one per IMM hypothesis), since each
hypothesis has its own (omega, p, joint_type).

Acceptance test:
    On synthetic trajectories with known theta_t: ≤ 1° (revolute) / ≤ 1 mm (prismatic).
    Robust to ±0.5 cm Gaussian noise on part pose.
"""
from __future__ import annotations

__all__: list[str] = []
