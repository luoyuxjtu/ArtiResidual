"""Module 10 — Scripted Expert Demo Generation.

See artiresidual_tech_spec.md §3 Module 10 for the authoritative API and the
per-demo data schema.

For each task:
    1. Read URDF ground truth → (omega, p, type) per part.
    2. Decide role assignment (stabilizer / actor) with simple heuristics.
    3. Plan stabilizer reach + grasp with cuRobo.
    4. Plan actor reach + grasp + articulation-following trajectory using
       analytical flow (Module 05).
    5. Execute in sim, record everything in the schema, drop failed trajectories.

Each demo carries ground-truth (omega_gt, p_gt, joint_type_gt, theta_t_gt) so
that the perturbation module (11) can synthesize wrong-initial-estimate replays
without re-running sim.

References:
    - ArticuBot gen_demo branch:    https://github.com/yufeiwang63/articubot/tree/gen_demo
    - RoboTwin expert demo gen:     https://robotwin-platform.github.io/doc/usage/control-robot.html
    - cuRobo motion planning:       https://github.com/NVlabs/curobo
"""
from __future__ import annotations

__all__: list[str] = []
