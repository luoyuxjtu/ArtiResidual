"""Module 02 — Run-Time Flow Predictor.

See artiresidual_tech_spec.md §3 Module 02 for the authoritative API.

This is the perception network that runs at 30 Hz on EVERY control step,
producing `f_pred: [B, N, 3]` from the current world-frame point cloud
(N=128 after FPS). It is distinct from the prior estimator (Module 01) and
trained against analytical-flow ground truth derived from sim URDFs.

Architecture: PointNet++ (3 SA + 3 FP layers) → 3-channel per-point head.
Optional warm-start from PAct / FlowBot3D pretrained weights.

References:
    - FlowBot3D flow head:    https://github.com/r-pad/flowbot3d
    - DP3 point encoder:      https://github.com/YanjieZe/3D-Diffusion-Policy
"""
from __future__ import annotations

__all__: list[str] = []
