"""Training subpackage — three-stage training pipeline.

Stage 1 (train_refiner):  refiner only, policy frozen. Uses perturbed demos.
Stage 2 (train_policy):   policy only, refiner frozen. Uses expert demos.
Stage 3 (train_joint):    everything unfrozen, low refiner-loss weight (λ_refiner=0.1).

Spec §6 (implementation order) + §4.8 (joint loss formula).
"""
