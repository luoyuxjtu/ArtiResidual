"""Policy subpackage — Modules 07 (DiT block) and 08 (bimanual DiT diffusion policy).

The DiT block has TWO cross-attention layers — one consuming the belief-weighted
analytical flow tokens `f_cond` (per-point), one consuming the entropy token
`H(w)`. This is how the IMM refiner's belief enters the policy.

v-prediction, cosine schedule, 10-step DDIM sampler. Target ~50M params; see
configs/policy/dit_50m.yaml.
"""
