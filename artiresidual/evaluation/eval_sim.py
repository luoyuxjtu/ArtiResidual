"""Sim evaluation harness.

Runs the trained policy + refiner over the MultiHandleBenchmark (Module 09),
records full trajectories, computes success rate per task, and pipes each
trajectory through failure_analysis (Module 12) for breakdown plots.

Intended invocation (server-side):
    python -m artiresidual.evaluation.eval_sim --config-name=base \\
        ckpt_path=/workspace/ckpts/joint_stage3.pt \\
        eval.n_trials_per_task=100
"""
from __future__ import annotations

__all__: list[str] = []
