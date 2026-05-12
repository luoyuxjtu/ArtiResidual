"""Module 01 — Prior Articulation Estimator.

See artiresidual_tech_spec.md §3 Module 01 for the authoritative API.

At task start (t=0), produce an initial estimate of (omega, p, joint_type) for
each manipulable part. This estimate may be wrong — that is WHY the IMM refiner
(Module 04) exists. Do NOT over-engineer this module: once it hits ~70% joint-
type accuracy on held-out PartNet-Mobility, move on.

References:
    - FlowBot3D backbone:    https://github.com/r-pad/flowbot3d
    - PartNet-Mobility:      via SAPIEN URDF parser
"""
from __future__ import annotations

__all__: list[str] = []
