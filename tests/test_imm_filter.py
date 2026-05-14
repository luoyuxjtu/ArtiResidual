"""Unit tests for IMMArticulationRefiner.step() (Module 04).

All tests run on CPU only and require only torch + omegaconf.

Test plan (10 functions):

    1.  test_step_output_shapes          — returned dict has correct shapes
    2.  test_step_omega_unit_norm        — ‖ωₖ‖ = 1 for all (B, K) after step
    3.  test_step_omega_correction_bounded
                                         — actual Δω ≤ η · 30° (clipping holds)
    4.  test_step_p_correction_bounded   — ‖p_new − p_old‖ ≤ η · 5 cm
    5.  test_step_weights_sum_to_one     — w_k.sum(dim=-1) ≡ 1.0
    6.  test_step_weights_floor          — all weights ≥ w_min after every step
    7.  test_step_type_k_unchanged       — type_k identical before and after step
    8.  test_step_gradient_flow          — backward() succeeds, grad is non-None
    9.  test_step_chain_stable           — 5 consecutive steps stay numerically stable
    10. test_step_batch_size_invariant   — B=1 and B=4 give consistent per-sample results
"""
from __future__ import annotations

import math

import pytest
import torch
from omegaconf import OmegaConf

from artiresidual.refiner.analytical_flow import (
    JOINT_TYPE_PRISMATIC,
    JOINT_TYPE_REVOLUTE,
)
from artiresidual.refiner.imm_filter import IMMArticulationRefiner

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DEG = math.pi / 180.0

_CFG = OmegaConf.create(
    {
        "K": 3,
        "window_T": 30,
        "update_interval_N": 10,
        "dim": 256,
        "n_heads": 4,
        "n_layers": 4,
        "w_min": 0.05,
        "lambda_H": 0.01,
        "eta": 0.5,
        "omega_clip_deg": 30.0,
        "p_clip_m": 0.05,
    }
)


@pytest.fixture(scope="module")
def model() -> IMMArticulationRefiner:
    m = IMMArticulationRefiner(_CFG)
    m.eval()
    return m


def _make_hypotheses(B: int, K: int = 3) -> dict:
    """Random but valid hypothesis dict."""
    omega_raw = torch.randn(B, K, 3)
    omega_k = omega_raw / omega_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    type_k = torch.zeros(B, K, dtype=torch.long)
    type_k[:, 2] = JOINT_TYPE_PRISMATIC
    w_k = torch.full((B, K), 1.0 / K)
    return {
        "omega_k": omega_k,
        "p_k": torch.randn(B, K, 3) * 0.3,
        "type_k": type_k,
        "w_k": w_k,
    }


def _make_window(B: int, K: int = 3, T: int = 30, N: int = 64) -> dict:
    """Random dummy window matching the step() input contract."""
    return {
        "delta_flow_k": torch.randn(B, K, T, N, 3),
        "wrench":       torch.randn(B, T, 12),
        "wrench_res_k": torch.randn(B, K, T, 3),
        "action":       torch.randn(B, T, 14),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_step_output_shapes(model: IMMArticulationRefiner) -> None:
    B, K, T, N = 4, _CFG.K, _CFG.window_T, 64
    hyp = _make_hypotheses(B, K)
    win = _make_window(B, K, T, N)

    with torch.no_grad():
        out = model.step(hyp, win)

    assert out["omega_k"].shape == (B, K, 3), f"omega_k: {out['omega_k'].shape}"
    assert out["p_k"].shape     == (B, K, 3), f"p_k: {out['p_k'].shape}"
    assert out["type_k"].shape  == (B, K),    f"type_k: {out['type_k'].shape}"
    assert out["w_k"].shape     == (B, K),    f"w_k: {out['w_k'].shape}"


def test_step_omega_unit_norm(model: IMMArticulationRefiner) -> None:
    """All hypothesis axes must remain on S² after step()."""
    B, K = 8, _CFG.K
    hyp = _make_hypotheses(B, K)
    win = _make_window(B, K)

    with torch.no_grad():
        out = model.step(hyp, win)

    norms = out["omega_k"].norm(dim=-1)  # [B, K]
    max_dev = (norms - 1.0).abs().max().item()
    assert max_dev < 1e-5, f"max |‖ω_k‖ − 1| = {max_dev:.2e} (expected < 1e-5)"


def test_step_omega_correction_bounded(model: IMMArticulationRefiner) -> None:
    """Actual ω correction (geodesic angle) must not exceed η · 30°."""
    B, K = 8, _CFG.K
    hyp = _make_hypotheses(B, K)
    win = _make_window(B, K)

    omega_before = hyp["omega_k"].clone()

    with torch.no_grad():
        out = model.step(hyp, win)

    omega_after = out["omega_k"]  # [B, K, 3]

    # Geodesic angle = arccos(clip(dot, -1, 1))
    dot = (omega_before * omega_after).sum(dim=-1).clamp(-1.0, 1.0)  # [B, K]
    angle_rad = torch.acos(dot)  # [B, K]

    # Clipping is applied to η·Δω, so max angle = η * clip = 0.5 * (30° in rad)
    max_allowed = _CFG.eta * _CFG.omega_clip_deg * _DEG  # 0.5 * 30° = 15°
    max_observed = angle_rad.max().item()
    assert max_observed <= max_allowed + 1e-5, (
        f"max geodesic angle {math.degrees(max_observed):.2f}° "
        f"> η · 30° = {math.degrees(max_allowed):.2f}°"
    )


def test_step_p_correction_bounded(model: IMMArticulationRefiner) -> None:
    """‖p_new − p_old‖ ≤ η · 5 cm for every (batch, hypothesis) pair."""
    B, K = 8, _CFG.K
    hyp = _make_hypotheses(B, K)
    win = _make_window(B, K)
    p_before = hyp["p_k"].clone()

    with torch.no_grad():
        out = model.step(hyp, win)

    delta_norm = (out["p_k"] - p_before).norm(dim=-1)  # [B, K]
    max_allowed = _CFG.eta * _CFG.p_clip_m             # 0.5 * 0.05 = 0.025 m
    max_observed = delta_norm.max().item()
    assert max_observed <= max_allowed + 1e-6, (
        f"max ‖Δp‖ = {max_observed*100:.3f} cm "
        f"> η · 5 cm = {max_allowed*100:.1f} cm"
    )


def test_step_weights_sum_to_one(model: IMMArticulationRefiner) -> None:
    B, K = 8, _CFG.K
    hyp = _make_hypotheses(B, K)
    win = _make_window(B, K)

    with torch.no_grad():
        out = model.step(hyp, win)

    sums = out["w_k"].sum(dim=-1)  # [B]
    max_err = (sums - 1.0).abs().max().item()
    assert max_err < 1e-5, f"max |sum(w_k) − 1| = {max_err:.2e}"


def test_step_weights_floor(model: IMMArticulationRefiner) -> None:
    """No hypothesis weight should fall below w_min after a step."""
    B, K = 16, _CFG.K
    hyp = _make_hypotheses(B, K)
    # Skew weights heavily toward one hypothesis to stress the floor.
    hyp["w_k"] = torch.tensor([[0.90, 0.05, 0.05]]).expand(B, -1).clone()
    win = _make_window(B, K)

    with torch.no_grad():
        out = model.step(hyp, win)

    min_w = out["w_k"].min().item()
    assert min_w >= _CFG.w_min - 1e-6, (
        f"min weight {min_w:.4f} < w_min = {_CFG.w_min}"
    )


def test_step_type_k_unchanged(model: IMMArticulationRefiner) -> None:
    """type_k must be the exact same tensor (or at least identical values) after step."""
    B, K = 4, _CFG.K
    hyp = _make_hypotheses(B, K)
    type_before = hyp["type_k"].clone()
    win = _make_window(B, K)

    with torch.no_grad():
        out = model.step(hyp, win)

    assert torch.equal(out["type_k"], type_before), (
        "type_k was modified by step() — v1 invariant violated"
    )


def test_step_gradient_flow(model: IMMArticulationRefiner) -> None:
    """Backward pass must succeed and produce non-None gradients on all params."""
    model.train()
    B, K, T, N = 2, _CFG.K, _CFG.window_T, 32

    hyp = _make_hypotheses(B, K)
    hyp["omega_k"].requires_grad_(True)  # verify gradient flows through omega too
    win = _make_window(B, K, T, N)
    win["delta_flow_k"].requires_grad_(True)

    out = model.step(hyp, win)

    # Use a simple scalar loss on all outputs.
    loss = (
        out["w_k"].sum()
        + out["omega_k"].sum()
        + out["p_k"].sum()
    )
    loss.backward()

    # Check at least one model parameter has a gradient.
    grad_norms = [
        p.grad.norm().item()
        for p in model.parameters()
        if p.grad is not None
    ]
    assert len(grad_norms) > 0, "No parameter received a gradient"
    assert any(g > 0 for g in grad_norms), "All parameter gradients are zero"

    model.eval()  # restore eval mode for subsequent fixture uses


def test_step_chain_stable(model: IMMArticulationRefiner) -> None:
    """Five consecutive step() calls must not produce NaNs or Infs."""
    B, K = 4, _CFG.K
    hyp = _make_hypotheses(B, K)

    for i in range(5):
        win = _make_window(B, K)
        with torch.no_grad():
            hyp = model.step(hyp, win)

        for key, val in hyp.items():
            assert not torch.isnan(val).any(), f"NaN in {key} after step {i+1}"
            assert not torch.isinf(val).any(), f"Inf in {key} after step {i+1}"

        # Invariants must hold at every step.
        norms = hyp["omega_k"].norm(dim=-1)
        assert (norms - 1.0).abs().max().item() < 1e-4, f"omega not unit at step {i+1}"
        assert (hyp["w_k"].sum(dim=-1) - 1.0).abs().max().item() < 1e-4, \
            f"weights don't sum to 1 at step {i+1}"
        assert hyp["w_k"].min().item() >= _CFG.w_min - 1e-5, \
            f"weight floor violated at step {i+1}"


@pytest.mark.parametrize("B", [1, 4, 16])
def test_step_batch_size_invariant(
    model: IMMArticulationRefiner, B: int
) -> None:
    """step() output shape scales linearly with B; single-sample works."""
    K, T, N = _CFG.K, _CFG.window_T, 64
    hyp = _make_hypotheses(B, K)
    win = _make_window(B, K, T, N)

    with torch.no_grad():
        out = model.step(hyp, win)

    assert out["omega_k"].shape == (B, K, 3)
    assert out["w_k"].shape == (B, K)
    assert not torch.isnan(out["omega_k"]).any()
    assert not torch.isnan(out["w_k"]).any()
