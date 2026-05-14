"""Mock perturbed dataset for Stage-1 IMM refiner training.

Stand-in for Module 11 (real RoboTwin perturbed demos) until the full
data pipeline is available.  Generates synthetic samples that match the
Module 11 data contract while providing a clear learning signal for L_NLL.

Perturbation distribution mirrors spec §3 Module 11:
    omega:      rotate by Uniform(5°, 30°) about a random tangent axis.
    p:          Gaussian σ=2 cm, clipped at 5 cm.
    joint_type: swap revolute↔prismatic with probability 0.20.

Each sample has keys:
    prior_omega   [3]        prior (wrong) axis direction (unit)
    prior_p       [3]        prior (wrong) axis origin
    prior_type    []         long, 0=revolute or 1=prismatic
    prior_conf    []         float, fixed at 0.8
    delta_flow_k  [K,T,N,3]  flow residuals f_pred − f_ana(hyp_k)
    wrench        [T,12]     simulated bimanual wrench
    wrench_res_k  [K,T,3]    per-hypothesis wrench residual direction
    action        [T,14]     bimanual joint commands
    gt_omega      [3]        ground-truth axis direction (unit)
    gt_p          [3]        ground-truth axis origin
    gt_type_idx   []         long, index of correct hypothesis (0=rev, 2=pris)

Informative vs. uninformative:
    informative=True  (default):
        delta_flow_k[gt_type_idx] ~ N(0, σ_ok)   σ_ok  = 0.10  (small residual)
        delta_flow_k[others]      ~ N(0, σ_bad)  σ_bad = 0.50  (large residual)
        The 5× contrast gives L_NLL a gradient signal from the first step.
    informative=False:
        All delta_flow_k ~ N(0, 1).  No signal; useful for ablations.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.utils.data as data
from torch import Tensor

__all__ = ["MockPerturbedDataset"]

# Hypothesis index constants (match IMMArticulationRefiner.init_hypotheses)
_IDX_REVOLUTE  = 0  # h1: revolute, nominally vertical axis
_IDX_PRISMATIC = 2  # h3: prismatic

# Joint type encoding (matches analytical_flow constants)
_JTYPE_REVOLUTE  = 0
_JTYPE_PRISMATIC = 1


class MockPerturbedDataset(data.Dataset):
    """Synthetic perturbed demo dataset for Stage-1 refiner training.

    Args:
        n_samples:    Number of synthetic samples.
        K:            Number of hypotheses (must be 3 to match init_hypotheses).
        T:            Evidence window length (must equal refiner's window_T).
        N:            Points per cloud in the window.
        seed:         Base random seed; sample i uses seed+i for reproducibility.
        informative:  If True, correct hypothesis has 5× smaller flow residuals.
    """

    def __init__(
        self,
        n_samples: int,
        K: int = 3,
        T: int = 30,
        N: int = 64,
        seed: int = 42,
        informative: bool = True,
    ) -> None:
        super().__init__()
        if K != 3:
            raise ValueError(
                f"MockPerturbedDataset only supports K=3 (matches init_hypotheses); "
                f"got K={K}."
            )
        self.n_samples   = n_samples
        self.K           = K
        self.T           = T
        self.N           = N
        self.seed        = seed
        self.informative = informative

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        rng = torch.Generator()
        rng.manual_seed(self.seed + idx)
        return self._generate(rng)

    # ------------------------------------------------------------------
    # Sample generation
    # ------------------------------------------------------------------

    def _generate(self, rng: torch.Generator) -> dict[str, Tensor]:
        K, T, N = self.K, self.T, self.N

        # ── Ground-truth articulation ─────────────────────────────────
        # Joint type: 50/50 revolute/prismatic.
        gt_joint_type = int(torch.randint(0, 2, (1,), generator=rng).item())
        gt_type_idx   = _IDX_REVOLUTE if gt_joint_type == _JTYPE_REVOLUTE else _IDX_PRISMATIC

        # Axis direction: random unit vector on S².
        gt_omega = _random_unit_vec(rng)           # [3]
        # Axis origin: random position (±0.3 m around workspace centre).
        gt_p = torch.randn(3, generator=rng) * 0.3  # [3]

        # ── Perturbed prior ───────────────────────────────────────────
        prior_omega = _perturb_omega(gt_omega, rng)                        # [3]
        prior_p     = _perturb_p(gt_p, rng, sigma_m=0.02, clip_m=0.05)    # [3]
        # Swap joint type with probability 0.20.
        swap_type = torch.rand(1, generator=rng).item() < 0.20
        prior_type_int = (1 - gt_joint_type) if swap_type else gt_joint_type
        prior_type = torch.tensor(prior_type_int, dtype=torch.long)

        # ── Evidence window ───────────────────────────────────────────
        if self.informative:
            sigma_ok  = 0.10   # correct hypothesis: small residual (good fit)
            sigma_bad = 0.50   # wrong hypotheses: large residual (poor fit)
            delta_flow = torch.randn(K, T, N, 3, generator=rng) * sigma_bad
            # Replace correct hypothesis slice with small-sigma noise.
            delta_flow[gt_type_idx] = (
                torch.randn(T, N, 3, generator=rng) * sigma_ok
            )
        else:
            delta_flow = torch.randn(K, T, N, 3, generator=rng)

        wrench       = torch.randn(T, 12, generator=rng) * 0.5
        wrench_res_k = torch.randn(K, T, 3,  generator=rng) * 0.3
        action       = torch.randn(T, 14, generator=rng) * 0.1

        return {
            "prior_omega":   prior_omega,                                    # [3]
            "prior_p":       prior_p,                                        # [3]
            "prior_type":    prior_type,                                     # []  long
            "prior_conf":    torch.tensor(0.8),                              # []  float
            "delta_flow_k":  delta_flow,                                     # [K,T,N,3]
            "wrench":        wrench,                                         # [T,12]
            "wrench_res_k":  wrench_res_k,                                   # [K,T,3]
            "action":        action,                                         # [T,14]
            "gt_omega":      gt_omega,                                       # [3]
            "gt_p":          gt_p,                                           # [3]
            "gt_type_idx":   torch.tensor(gt_type_idx, dtype=torch.long),   # []  long
        }


# ---------------------------------------------------------------------------
# Module-private geometry helpers
# ---------------------------------------------------------------------------


def _random_unit_vec(rng: torch.Generator) -> Tensor:
    """Sample a uniformly random unit vector on S²."""
    v = torch.randn(3, generator=rng)
    return v / v.norm().clamp(min=1e-8)


def _perturb_omega(omega: Tensor, rng: torch.Generator) -> Tensor:
    """Rotate omega by Uniform(5°, 30°) about a random tangent axis (spec §3 M11)."""
    angle_rad = (5.0 + 25.0 * torch.rand(1, generator=rng).item()) * math.pi / 180.0

    # Random vector tangent to omega: subtract the parallel component.
    rand_vec = torch.randn(3, generator=rng)
    tangent  = rand_vec - (rand_vec * omega).sum() * omega
    norm_t   = tangent.norm()
    if norm_t < 1e-6:
        # Degenerate case (rand_vec ≈ ±omega): use a fixed perpendicular.
        ref     = torch.tensor([1.0, 0.0, 0.0]) if abs(omega[0].item()) < 0.9 \
                  else torch.tensor([0.0, 1.0, 0.0])
        tangent = ref - (ref * omega).sum() * omega
        norm_t  = tangent.norm()
    tangent = tangent / norm_t.clamp(min=1e-8)

    # Rodrigues rotation: ω_new = cos(θ)·ω + sin(θ)·tangent.
    omega_new = math.cos(angle_rad) * omega + math.sin(angle_rad) * tangent
    return omega_new / omega_new.norm().clamp(min=1e-8)


def _perturb_p(
    p: Tensor,
    rng: torch.Generator,
    sigma_m: float = 0.02,
    clip_m: float  = 0.05,
) -> Tensor:
    """Gaussian σ=2 cm perturbation, clipped at 5 cm (spec §3 M11)."""
    noise = torch.randn(3, generator=rng) * sigma_m
    norm  = noise.norm()
    if norm > clip_m:
        noise = noise * (clip_m / norm)
    return p + noise
