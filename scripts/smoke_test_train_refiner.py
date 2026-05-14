"""Smoke test for the Stage-1 refiner training loop.

Runs 100 training steps on synthetic data and asserts that the total loss
is lower in the last 10 steps than in the first 10 steps.  Also runs the
full-length training if --full is passed.

Usage:
    # Quick 100-step check (CPU, ~30 s):
    python scripts/smoke_test_train_refiner.py

    # Full 100 k-step run (needs CUDA):
    python scripts/smoke_test_train_refiner.py --full
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
from omegaconf import OmegaConf

# Make sure the project root is on the path when running as a script.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from artiresidual.training.train_refiner import train

PASS = "✓"
FAIL = "✗"


def check(cond: bool, msg: str) -> None:
    sym = PASS if cond else FAIL
    print(f"  {sym} {msg}")
    if not cond:
        sys.exit(1)


def _smoke_cfg(device: str = "cpu") -> object:
    """Build a minimal OmegaConf config for the 100-step smoke test."""
    return OmegaConf.create({
        # Keys under "global" are reserved in Python; access via cfg["global"].
        "global": {
            "seed": 42,
            "device": device,
            "dtype": "float32",
        },
        "refiner": {
            "_target_": "artiresidual.refiner.imm_filter.IMMArticulationRefiner",
            "K": 3,
            "window_T": 10,        # small T for speed
            "update_interval_N": 10,
            "dim": 64,             # small model for speed
            "n_heads": 2,
            "n_layers": 2,
            "w_min": 0.05,
            "lambda_H": 0.01,
            "eta": 0.5,
            "omega_clip_deg": 30.0,
            "p_clip_m": 0.05,
        },
        "training": {
            "batch_size": 8,
            "num_workers": 0,
            "lr": 3.0e-3,          # higher LR → faster convergence in smoke test
            "weight_decay": 1.0e-4,
            "grad_clip": 1.0,
            "warmup_steps": 5,
            "total_steps": 100,
            "amp_dtype": "fp32",   # no AMP on CPU
            "compile": False,
            "ddp": False,
        },
        "logging": {
            "use_wandb": False,
            "project": "artiresidual-smoke",
            "entity": None,
            "tags": [],
            "log_every": 10,
            "ckpt_every": 999_999,
            "vis_every": 999_999,
        },
        "paths": {
            "data_root": "/tmp/artiresidual_smoke",
            "ckpt_root": "/tmp/artiresidual_smoke/ckpts",
            "output_root": "/tmp/artiresidual_smoke/outputs",
        },
        "stage1": {
            "lambda_p": 100.0,
            "acceptance_every": 999_999,
            "dataset": "mock",
            "n_samples": 200,      # small dataset for speed
            "N_points": 32,        # small N for speed
        },
        "perception": {
            "use_gt_part_pose": True,
            "sam2_checkpoint": None,
            "sam2_device": "cuda",
        },
    })


def _full_cfg() -> object:
    """Config for the full 100 k-step run; needs CUDA."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = _smoke_cfg(device)
    OmegaConf.update(base, "refiner.dim", 256)
    OmegaConf.update(base, "refiner.n_heads", 4)
    OmegaConf.update(base, "refiner.n_layers", 4)
    OmegaConf.update(base, "refiner.window_T", 30)
    OmegaConf.update(base, "training.batch_size", 32)
    OmegaConf.update(base, "training.num_workers", 4)
    OmegaConf.update(base, "training.lr", 1.0e-4)
    OmegaConf.update(base, "training.warmup_steps", 1000)
    OmegaConf.update(base, "training.total_steps", 100_000)
    OmegaConf.update(base, "training.amp_dtype", "bf16" if device == "cuda" else "fp32")
    OmegaConf.update(base, "logging.log_every", 50)
    OmegaConf.update(base, "logging.ckpt_every", 5000)
    OmegaConf.update(base, "stage1.n_samples", 6000)
    OmegaConf.update(base, "stage1.N_points", 64)
    OmegaConf.update(base, "stage1.acceptance_every", 10000)
    return base


def run_smoke(full: bool = False) -> None:
    import logging
    logging.basicConfig(level=logging.WARNING)  # suppress step-level INFO spam

    print()
    print("=" * 64)
    mode = "FULL (100 k steps)" if full else "SMOKE (100 steps)"
    print(f"  Stage-1 refiner training — {mode}")
    print("=" * 64)

    cfg = _full_cfg() if full else _smoke_cfg()
    device_str = str(cfg["global"]["device"])
    print(f"  device={device_str}  dim={cfg.refiner.dim}  "
          f"T={cfg.refiner.window_T}  N={cfg.stage1.N_points}  "
          f"steps={cfg.training.total_steps}")
    print()

    t0 = time.time()
    history = train(cfg)
    elapsed = time.time() - t0

    loss = history["loss_total"]
    nll  = history["loss_nll"]
    top1 = history["top1_acc"]

    n = len(loss)
    check(n > 0, f"history is non-empty ({n} entries)")

    if not full:
        # Smoke: verify loss decreases over 100 steps.
        seg = max(1, n // 10)
        first_avg = sum(loss[:seg]) / seg
        last_avg  = sum(loss[-seg:]) / seg
        print(f"  Loss: first {seg} steps avg = {first_avg:.4f}")
        print(f"  Loss: last  {seg} steps avg = {last_avg:.4f}")
        check(
            last_avg < first_avg,
            f"loss decreasing: {last_avg:.4f} < {first_avg:.4f}",
        )

        nll_first = sum(nll[:seg]) / seg
        nll_last  = sum(nll[-seg:]) / seg
        print(f"  NLL:  first {seg} steps avg = {nll_first:.4f}")
        print(f"  NLL:  last  {seg} steps avg = {nll_last:.4f}")
        check(
            nll_last < nll_first,
            f"NLL decreasing: {nll_last:.4f} < {nll_first:.4f}",
        )

        top1_last = sum(top1[-seg:]) / seg
        print(f"  Top1 accuracy (last {seg} steps): {top1_last:.3f}")

    else:
        # Full: just report final metrics.
        seg = min(50, n)
        last_loss = sum(loss[-seg:]) / seg
        last_top1 = sum(top1[-seg:]) / seg
        omega_err = sum(history["omega_err_deg"][-seg:]) / seg
        p_err     = sum(history["p_err_cm"][-seg:]) / seg
        print(f"  Final (last {seg} steps):")
        print(f"    loss  = {last_loss:.4f}")
        print(f"    top1  = {last_top1:.3f}")
        print(f"    ω_err = {omega_err:.1f}°")
        print(f"    p_err = {p_err:.2f} cm")
        check(last_top1 >= 0.70, f"top1 ≥ 0.70 after full training ({last_top1:.3f})")

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print()
    print("=" * 64)
    print(f"  Training smoke test PASSED")
    print("=" * 64)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Run the full 100 k-step training (needs CUDA)")
    args = parser.parse_args()
    run_smoke(full=args.full)
