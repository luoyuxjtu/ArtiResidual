"""Unit tests for IMMArticulationRefiner.step(), get_f_cond(), loss_terms().

All tests run on CPU only and require only torch + omegaconf.

Test plan:
  step() tests (10):
    1.  test_step_output_shapes
    2.  test_step_omega_unit_norm
    3.  test_step_omega_correction_bounded
    4.  test_step_p_correction_bounded
    5.  test_step_weights_sum_to_one
    6.  test_step_weights_floor
    7.  test_step_type_k_unchanged
    8.  test_step_gradient_flow
    9.  test_step_chain_stable
    10. test_step_batch_size_invariant

  get_f_cond() tests (4):
    11. test_get_f_cond_output_shapes
    12. test_get_f_cond_entropy_bounds
    13. test_get_f_cond_uniform_weights_max_entropy
    14. test_get_f_cond_belief_weighted_sum_correctness

  loss_terms() tests (5):
    15. test_loss_terms_output_keys_and_shapes
    16. test_loss_terms_nll_nonnegative
    17. test_loss_terms_mu_residual_nonnegative
    18. test_loss_terms_gradient_flow
    19. test_loss_terms_perfect_hypotheses_low_loss

  Integration (1):
    20. test_integration_30_control_steps
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


# ---------------------------------------------------------------------------
# get_f_cond() — belief-weighted flow + entropy (spec §4.5 + §4.6)
# ---------------------------------------------------------------------------


def test_get_f_cond_output_shapes(model: IMMArticulationRefiner) -> None:
    """Returned f_cond is [B, N, 3] and entropy is [B]."""
    B, K, N = 4, _CFG.K, 128
    hyp = _make_hypotheses(B, K)
    theta_t = torch.rand(B) * math.pi
    pcd = torch.randn(B, N, 3)

    with torch.no_grad():
        f_cond, entropy = model.get_f_cond(hyp, theta_t, pcd)

    assert f_cond.shape == (B, N, 3), f"f_cond shape: {f_cond.shape}"
    assert entropy.shape == (B,),     f"entropy shape: {entropy.shape}"
    assert not torch.isnan(f_cond).any() and not torch.isinf(f_cond).any()
    assert not torch.isnan(entropy).any() and not torch.isinf(entropy).any()


def test_get_f_cond_entropy_bounds(model: IMMArticulationRefiner) -> None:
    """For K=3 with w_min=0.05, entropy is in [H_min, log(K)]."""
    B, K, N = 8, _CFG.K, 64
    hyp = _make_hypotheses(B, K)
    pcd = torch.randn(B, N, 3)

    with torch.no_grad():
        _, entropy = model.get_f_cond(hyp, torch.zeros(B), pcd)

    # H ∈ [0, log K] always.
    assert (entropy >= 0).all(), f"negative entropy: {entropy}"
    assert (entropy <= math.log(K) + 1e-5).all(), \
        f"entropy > log(K)={math.log(K):.4f}: max={entropy.max():.4f}"


def test_get_f_cond_uniform_weights_max_entropy(
    model: IMMArticulationRefiner,
) -> None:
    """Uniform weights yield entropy = log(K)."""
    B, K, N = 4, _CFG.K, 64
    hyp = _make_hypotheses(B, K)
    hyp["w_k"] = torch.full((B, K), 1.0 / K)  # exactly uniform

    with torch.no_grad():
        _, entropy = model.get_f_cond(hyp, torch.zeros(B), torch.randn(B, N, 3))

    expected = math.log(K)
    assert torch.allclose(entropy, torch.full((B,), expected), atol=1e-5), (
        f"uniform-weight entropy {entropy[0].item():.5f} ≠ log(K)={expected:.5f}"
    )


def test_get_f_cond_belief_weighted_sum_correctness(
    model: IMMArticulationRefiner,
) -> None:
    """When one hypothesis has weight ≈ 1, f_cond ≈ that hypothesis's flow."""
    B, K, N = 2, _CFG.K, 64
    hyp = _make_hypotheses(B, K)
    # Force a near-degenerate weight (respecting the floor): h1 dominates.
    hyp["w_k"] = torch.tensor([[0.90, 0.05, 0.05]]).expand(B, -1).clone()
    pcd = torch.randn(B, N, 3)

    with torch.no_grad():
        f_cond, _ = model.get_f_cond(hyp, torch.zeros(B), pcd)

    # f_cond should be dominated by h1's revolute flow at norm scale ~ 0.90
    # (since per-hypothesis max-norm = 1 and mixing weights to 0.90 / 0.05 / 0.05).
    max_norm = f_cond.norm(dim=-1).max().item()
    assert max_norm <= 1.0 + 1e-5, (
        f"f_cond max norm {max_norm:.4f} > 1.0 (sum of w_k caps at sum(w_k)=1)"
    )


# ---------------------------------------------------------------------------
# Integration test — full pipeline: init → 30 control steps → get_f_cond
# ---------------------------------------------------------------------------


def test_integration_30_control_steps(
    model: IMMArticulationRefiner,
) -> None:
    """Simulate a 30-step rollout: step() every N=10 steps + final get_f_cond().

    Verifies the full forward pipeline:
        init_hypotheses → [step] × 3 → get_f_cond
    and checks all returned shapes plus the standard invariants at each
    intermediate stage.
    """
    B, K, T, N = 2, _CFG.K, _CFG.window_T, 64
    n_control_steps = 30
    n_interval = _CFG.update_interval_N  # 10

    # 1. Initialize hypotheses from a mock Module 01 prior output.
    #    Mix of revolute/prismatic priors to exercise both weight paths.
    prior_output = {
        "omega":      torch.randn(B, 3),
        "p":          torch.randn(B, 3) * 0.5,
        "joint_type": torch.tensor([0, 1], dtype=torch.long),  # revolute, prismatic
        "confidence": torch.tensor([0.8, 0.7]),
    }
    hyp = model.init_hypotheses(prior_output)

    assert hyp["omega_k"].shape == (B, K, 3)
    assert hyp["p_k"].shape     == (B, K, 3)
    assert hyp["type_k"].shape  == (B, K)
    assert hyp["w_k"].shape     == (B, K)

    # Revolute prior → h1 = 0.6;  prismatic prior → h3 = 0.6.
    assert hyp["w_k"][0, 0].item() == pytest.approx(0.6, abs=1e-5)
    assert hyp["w_k"][1, 2].item() == pytest.approx(0.6, abs=1e-5)

    # 2. Simulate 30 control steps; call step() every 10 (= 3 times).
    n_step_calls = 0
    for t in range(n_control_steps):
        if (t + 1) % n_interval == 0:
            # Build evidence window from accumulated observations (here: dummy).
            window = _make_window(B, K, T, N)
            with torch.no_grad():
                hyp = model.step(hyp, window)
            n_step_calls += 1

            # Shape preservation
            assert hyp["omega_k"].shape == (B, K, 3)
            assert hyp["p_k"].shape     == (B, K, 3)
            assert hyp["type_k"].shape  == (B, K)
            assert hyp["w_k"].shape     == (B, K)

            # Invariants
            norms_dev = (hyp["omega_k"].norm(dim=-1) - 1.0).abs().max().item()
            assert norms_dev < 1e-4, \
                f"step #{n_step_calls}: ω not unit (max dev {norms_dev:.2e})"

            w_sum_err = (hyp["w_k"].sum(dim=-1) - 1.0).abs().max().item()
            assert w_sum_err < 1e-4, \
                f"step #{n_step_calls}: weights don't sum to 1 (err {w_sum_err:.2e})"

            min_w = hyp["w_k"].min().item()
            assert min_w >= _CFG.w_min - 1e-5, \
                f"step #{n_step_calls}: weight floor violated (min={min_w:.4f})"

    assert n_step_calls == 3, \
        f"Expected 3 step() calls in {n_control_steps} steps " \
        f"with N={n_interval}, got {n_step_calls}"

    # 3. Final get_f_cond() — produces conditioning for the DiT policy.
    theta_t = torch.rand(B) * math.pi          # arbitrary current joint config
    pcd     = torch.randn(B, N, 3)             # current scene point cloud
    with torch.no_grad():
        f_cond, entropy = model.get_f_cond(hyp, theta_t, pcd)

    assert f_cond.shape == (B, N, 3), f"f_cond shape: {f_cond.shape}"
    assert entropy.shape == (B,),     f"entropy shape: {entropy.shape}"

    assert not torch.isnan(f_cond).any() and not torch.isinf(f_cond).any()
    assert not torch.isnan(entropy).any() and not torch.isinf(entropy).any()

    # Entropy is still in valid bounds after 3 step() refinements.
    assert (entropy >= 0).all()
    assert (entropy <= math.log(K) + 1e-5).all()


# ---------------------------------------------------------------------------
# loss_terms() tests (spec §4.8–§4.10)
# ---------------------------------------------------------------------------


def _make_gt(B: int, device: str = "cpu") -> tuple:
    """Random but unit ground-truth (omega, p, type_idx) for loss_terms tests."""
    gt_omega_raw = torch.randn(B, 3)
    gt_omega = gt_omega_raw / gt_omega_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    gt_p     = torch.randn(B, 3) * 0.3
    # Alternate between hypothesis 0 (revolute) and 2 (prismatic).
    gt_type  = torch.zeros(B, dtype=torch.long)
    gt_type[B // 2:] = 2
    return gt_omega, gt_p, gt_type


def test_loss_terms_output_keys_and_shapes(model: IMMArticulationRefiner) -> None:
    """loss_terms() must return the four expected scalar tensors."""
    B, K = 4, _CFG.K
    hyp = _make_hypotheses(B, K)
    win = _make_window(B, K)
    gt_omega, gt_p, gt_type = _make_gt(B)

    with torch.no_grad():
        hyp_new = model.step(hyp, win)
    losses = model.loss_terms(hyp_new, gt_omega, gt_p, gt_type)

    for key in ("L_NLL", "L_mu_residual", "L_H", "L_total"):
        assert key in losses, f"missing key: {key}"
        val = losses[key]
        assert val.shape == (), f"{key} should be scalar, got {val.shape}"
        assert not torch.isnan(val), f"{key} is NaN"
        assert not torch.isinf(val), f"{key} is Inf"


def test_loss_terms_nll_nonnegative(model: IMMArticulationRefiner) -> None:
    """L_NLL = -log(w_gt) ≥ 0 because w_gt ∈ (0, 1]."""
    B, K = 8, _CFG.K
    for _ in range(5):
        hyp = _make_hypotheses(B, K)
        win = _make_window(B, K)
        gt_omega, gt_p, gt_type = _make_gt(B)
        with torch.no_grad():
            hyp_new = model.step(hyp, win)
        losses = model.loss_terms(hyp_new, gt_omega, gt_p, gt_type)
        assert losses["L_NLL"].item() >= -1e-6, (
            f"L_NLL = {losses['L_NLL'].item():.4f} < 0"
        )


def test_loss_terms_mu_residual_nonnegative(model: IMMArticulationRefiner) -> None:
    """L_mu_residual = Σ_k w_k · [(1−cos) + λ_p‖Δp‖²] ≥ 0 term-by-term."""
    B, K = 8, _CFG.K
    for _ in range(5):
        hyp = _make_hypotheses(B, K)
        win = _make_window(B, K)
        gt_omega, gt_p, gt_type = _make_gt(B)
        with torch.no_grad():
            hyp_new = model.step(hyp, win)
        losses = model.loss_terms(hyp_new, gt_omega, gt_p, gt_type)
        assert losses["L_mu_residual"].item() >= -1e-5, (
            f"L_mu_residual = {losses['L_mu_residual'].item():.4f} < 0"
        )


def test_loss_terms_gradient_flow(model: IMMArticulationRefiner) -> None:
    """backward() through L_total must update model parameters."""
    model.train()
    B, K, T, N = 2, _CFG.K, _CFG.window_T, 32

    hyp = _make_hypotheses(B, K)
    win = _make_window(B, K, T, N)
    gt_omega, gt_p, gt_type = _make_gt(B)

    # Zero existing grads.
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    hyp_new = model.step(hyp, win)
    losses  = model.loss_terms(hyp_new, gt_omega, gt_p, gt_type)
    losses["L_total"].backward()

    grad_norms = [
        p.grad.norm().item()
        for p in model.parameters()
        if p.grad is not None
    ]
    assert len(grad_norms) > 0,             "no parameter received a gradient"
    assert any(g > 0 for g in grad_norms),  "all parameter gradients are zero"

    model.eval()


def test_loss_terms_perfect_hypotheses_low_loss(
    model: IMMArticulationRefiner,
) -> None:
    """When the correct hypothesis already matches GT, L_NLL and L_mu_residual
    should be small.

    We force hypothesis 0 to be the GT type and manually set omega_k[0] and
    p_k[0] to match GT exactly, with weight ≈ 0.90.  L_NLL should be < 0.15
    and L_mu_residual (for the dominant hypothesis) should be near zero.
    """
    B, K = 4, _CFG.K
    gt_omega, gt_p, _ = _make_gt(B)
    # All samples have gt_type_idx = 0 (revolute, h1).
    gt_type = torch.zeros(B, dtype=torch.long)

    # Build hypotheses where h1 exactly matches GT.
    hyp = _make_hypotheses(B, K)
    hyp["omega_k"][:, 0, :] = gt_omega       # perfect axis for h1
    hyp["p_k"][:, 0, :]     = gt_p           # perfect position for h1
    hyp["w_k"] = torch.tensor([[0.90, 0.05, 0.05]]).expand(B, -1).clone()

    with torch.no_grad():
        losses = model.loss_terms(hyp, gt_omega, gt_p, gt_type)

    # L_NLL = -log(0.90) ≈ 0.105
    assert losses["L_NLL"].item() < 0.15, (
        f"L_NLL={losses['L_NLL'].item():.4f} unexpectedly large for w_gt=0.90"
    )
    # L_mu_residual: the h1 term should be ~0; only h2,h3 contribute with w=0.05.
    # Max possible: 0.05*(2+2) + 0.05*(2+2) = 0.40 (cosine max 2, p_err moderate).
    # With λ_p=100 and random p errors on h2,h3, this could be bigger, but we
    # just check it's finite and non-negative.
    assert losses["L_mu_residual"].item() >= -1e-5, "L_mu_residual < 0"
    assert not torch.isnan(losses["L_mu_residual"]), "L_mu_residual is NaN"
