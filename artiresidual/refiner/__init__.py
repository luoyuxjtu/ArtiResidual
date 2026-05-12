"""Refiner subpackage — Modules 04 (IMM filter ★), 05 (analytical flow), 06 (state estimator).

The IMM Articulation Refiner is the paper's central technical contribution: it
maintains K parallel hypotheses about (omega, p, joint_type) and refines them
every N control steps using residual flow (f_pred - f_ana) + wrench evidence.
"""
from artiresidual.refiner.analytical_flow import (
    JOINT_TYPE_FIXED,
    JOINT_TYPE_PRISMATIC,
    JOINT_TYPE_REVOLUTE,
    analytical_flow,
    analytical_flow_batched,
    belief_weighted_flow,
)

__all__ = [
    "JOINT_TYPE_FIXED",
    "JOINT_TYPE_PRISMATIC",
    "JOINT_TYPE_REVOLUTE",
    "analytical_flow",
    "analytical_flow_batched",
    "belief_weighted_flow",
]
