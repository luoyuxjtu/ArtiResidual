"""ArtiResidual — bimanual diffusion policy with self-correcting articulation perception.

See artiresidual_tech_spec.md (repo root) for the full specification.

Top-level subpackages:
    perception/   Modules 01-02: prior articulation estimator, run-time flow predictor.
    refiner/      Modules 04-06: IMM filter (★ core), analytical flow, state estimator.
    selector/     Module 03:     stabilizer-actor selector.
    policy/       Modules 07-08: DiT block, bimanual DiT diffusion policy.
    data/         Modules 09-11: benchmark, demo gen, perturbation.
    training/     Stage 1/2/3 trainers.
    evaluation/   eval_sim, eval_real, failure_analysis (Module 12).
    utils/        geometry, visualization.
"""
__version__ = "0.0.1"
