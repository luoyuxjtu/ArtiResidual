"""Perception subpackage — Modules 01 (prior estimator) and 02 (run-time flow predictor).

Module 01 runs once at t=0 to produce an initial guess of (omega, p, joint_type).
Module 02 runs every control step (30 Hz) to produce per-point predicted flow `f_pred`.
"""
