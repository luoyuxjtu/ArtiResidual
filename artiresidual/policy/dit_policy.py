"""Module 08 — Full ArtiResidual DiT Policy.

See artiresidual_tech_spec.md §3 Module 08 for the authoritative API.

Bimanual diffusion policy that consumes:
    - pcd        [B, 128, 3]
    - proprio    [B, 14]    (7+7 joint positions)
    - wrench     [B, 12]    (6+6 wrench)
    - stab_pose  [B, 7]     (3 pos + 4 quat for the stabilizer arm)
    - theta_t    [B]        (estimated joint configuration scalar)
    - f_cond     [B, 128, 3] from refiner.get_f_cond()
    - entropy    [B]        H(w) from the IMM refiner
and outputs a 7-DoF actor-arm joint-delta action chunk over T_a=16 steps.

Diffusion specifics:
    v-prediction (Salimans & Ho 2022), cosine schedule, DDIM 10-step inference.
Target params: ~50M (reduce depth or dim if you exceed 80M).

References:
    - DiT:    https://github.com/facebookresearch/DiT
    - RDT-1B: https://github.com/thu-ml/RoboticsDiffusionTransformer
    - DP3:    https://github.com/YanjieZe/3D-Diffusion-Policy
"""
from __future__ import annotations

__all__: list[str] = []
