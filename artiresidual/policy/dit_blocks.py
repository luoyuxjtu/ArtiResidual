"""Module 07 — ArtiResidual DiT Block (with dual cross-attention).

See artiresidual_tech_spec.md §3 Module 07 for the authoritative API.

A single DiT transformer block with TWO cross-attention layers:
    Cross-Attention 1:  Q=x, K=V=f_cond_tokens     [B, 128, dim]
    Cross-Attention 2:  Q=x, K=V=entropy_token     [B, 1,   dim]

Modulation: AdaLN on the self-attention and FFN paths (scale/shift/gate from
t_emb); cross-attention is NOT modulated by timestep — keeping the conditioning
clean is a key design choice.

References:
    - DiT (Peebles & Xie):     https://github.com/facebookresearch/DiT
    - RDT-1B bimanual DiT:     https://github.com/thu-ml/RoboticsDiffusionTransformer
"""
from __future__ import annotations

__all__: list[str] = []
