"""Module 03 — Stabilizer-Actor Selector.

See artiresidual_tech_spec.md §3 Module 03 for the authoritative API.

A "borrowed" component: implement cleanly, don't innovate here. At t=0,
classifies each arm's role (stabilizer vs. actor) over the detected parts
and regresses an SE(3) grasp pose for each arm. Once roles are fixed, this
module is not called again during the trajectory.

Architecture: 4-layer transformer encoder, dim=256, 4 heads. Trained on
scripted demo data which carries explicit role labels (see Module 10).

References:
    - VoxAct-B stabilizer-actor formulation: https://github.com/VoxAct-B/voxact-b
"""
from __future__ import annotations

__all__: list[str] = []
