"""Refiner subpackage — Modules 04 (IMM filter ★), 05 (analytical flow), 06 (state estimator).

The IMM Articulation Refiner is the paper's central technical contribution: it
maintains K parallel hypotheses about (omega, p, joint_type) and refines them
every N control steps using residual flow (f_pred - f_ana) + wrench evidence.

Module 05's three analytical-flow variants are colocated in
`analytical_flow.py`: ``analytical_flow_hard`` (alias of ``analytical_flow``),
``analytical_flow_diff`` (gradient through ω, p), and ``analytical_flow_soft``
(gradient through joint-type logits).
"""
from artiresidual.refiner.analytical_flow import (
    JOINT_TYPE_FIXED,
    JOINT_TYPE_PRISMATIC,
    JOINT_TYPE_REVOLUTE,
    analytical_flow,
    analytical_flow_batched,
    analytical_flow_diff,
    analytical_flow_hard,
    analytical_flow_soft,
    belief_weighted_flow,
    constraint_directions,
    normalize_axis,
)

__all__ = [
    "JOINT_TYPE_FIXED",
    "JOINT_TYPE_PRISMATIC",
    "JOINT_TYPE_REVOLUTE",
    "analytical_flow",
    "analytical_flow_batched",
    "analytical_flow_diff",
    "analytical_flow_hard",
    "analytical_flow_soft",
    "belief_weighted_flow",
    "constraint_directions",
    "normalize_axis",
]
