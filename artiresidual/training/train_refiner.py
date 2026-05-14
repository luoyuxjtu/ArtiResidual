"""Stage 1 — Train the IMM articulation refiner with the policy frozen.

See artiresidual_tech_spec.md §3 Module 04 + §4.9-§4.10 (L_NLL, L_mu_residual)
+ §6 (implementation order).

Hydra entry point (single-GPU):
    python -m artiresidual.training.train_refiner --config-name=refiner_stage1

DDP entry point (2× A800):
    torchrun --nproc_per_node=2 -m artiresidual.training.train_refiner \
        --config-name=refiner_stage1

Training data:
    Perturbed expert demos from Module 11 (12 tasks × 100 demos × 5 replays).
    Each replay carries: wrong initial (omega, p, type), the unchanged ground-
    truth trajectory, observed wrenches and pcds. The refiner is supervised to
    (a) converge its top weight to the ground-truth hypothesis and (b) reduce
    its omega/p residual.

Loss (spec §4.8 with policy disabled):
    L = L_NLL + L_mu_residual + λ_H · (-H(w))

wandb panels to watch:
    - loss/total, loss/nll, loss/mu_residual, loss/H
    - metric/top1_acc, metric/omega_err_deg, metric/p_err_cm, metric/entropy
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from artiresidual.refiner.affordance_utils import hypothesis_entropy
from artiresidual.refiner.imm_filter import IMMArticulationRefiner
from artiresidual.training.datasets import MockPerturbedDataset

try:
    import hydra
    _HYDRA_AVAILABLE = True
except ImportError:
    _HYDRA_AVAILABLE = False

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

log = logging.getLogger(__name__)

__all__ = ["train"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(cfg: DictConfig, *keys: str, default: object = None) -> object:
    """OmegaConf-safe nested get with fallback (handles missing sub-configs)."""
    obj = cfg
    for k in keys:
        try:
            obj = obj[k]
        except (KeyError, AttributeError):
            return default
    return obj


def build_dataset(cfg: DictConfig) -> MockPerturbedDataset:
    """Build the training dataset (mock until Module 11 is ready)."""
    s1 = _get(cfg, "stage1") or OmegaConf.create({})
    K       = int(cfg.refiner.K)
    T       = int(cfg.refiner.window_T)
    N       = int(_get(s1, "N_points") or 64)
    n       = int(_get(s1, "n_samples") or 6000)
    dataset = str(_get(s1, "dataset") or "mock")

    if dataset != "mock":
        raise NotImplementedError(
            f"Real dataset '{dataset}' not yet implemented; "
            "Module 11 (RoboTwin perturbed demos) is pending."
        )
    return MockPerturbedDataset(n_samples=n, K=K, T=T, N=N, seed=42, informative=True)


def build_dataloader(
    dataset: MockPerturbedDataset,
    cfg: DictConfig,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, Optional[DistributedSampler]]:
    """Build a DataLoader, optionally with DistributedSampler."""
    use_ddp  = world_size > 1
    sampler  = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                  shuffle=True, drop_last=True) if use_ddp else None
    loader   = DataLoader(
        dataset,
        batch_size  = int(cfg.training.batch_size),
        sampler     = sampler,
        shuffle     = (sampler is None),
        num_workers = int(cfg.training.num_workers),
        pin_memory  = True,
        drop_last   = True,
        persistent_workers = (int(cfg.training.num_workers) > 0),
    )
    return loader, sampler


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay with linear warmup from 0."""

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


def _compute_metrics(
    hyp_new: dict,
    gt_omega: Tensor,
    gt_p: Tensor,
    gt_type_idx: Tensor,
) -> dict[str, Tensor]:
    """Per-batch metrics: top1_acc, omega_err_deg, p_err_cm, entropy."""
    w_k    = hyp_new["w_k"]       # [B, K]
    omega_k = hyp_new["omega_k"]  # [B, K, 3]
    p_k     = hyp_new["p_k"]      # [B, K, 3]

    # Top-1 accuracy: fraction where argmax(w_k) == gt.
    top1_pred = w_k.argmax(dim=-1)  # [B]
    top1_acc  = (top1_pred == gt_type_idx).float().mean()

    # Angular error of the top-1 hypothesis vs. GT (degrees).
    top1_idx     = top1_pred.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3)  # [B,1,3]
    omega_top1   = F.normalize(omega_k.gather(1, top1_idx).squeeze(1), dim=-1)  # [B,3]
    gt_omega_u   = F.normalize(gt_omega, dim=-1)
    cos_sim      = (omega_top1 * gt_omega_u).sum(dim=-1).clamp(-1.0, 1.0)
    omega_err    = torch.acos(cos_sim).mean() * (180.0 / math.pi)

    # Position error in cm.
    p_top1   = p_k.gather(1, top1_idx).squeeze(1)  # [B, 3]
    p_err_cm = (p_top1 - gt_p).norm(dim=-1).mean() * 100.0

    # Entropy of the belief distribution.
    entropy = hypothesis_entropy(w_k).mean()

    return {
        "top1_acc":     top1_acc,
        "omega_err_deg": omega_err,
        "p_err_cm":      p_err_cm,
        "entropy":       entropy,
    }


def run_acceptance_test(
    model_raw: IMMArticulationRefiner,
    cfg: DictConfig,
    device: torch.device,
    lambda_p: float = 100.0,
    n_eval: int = 100,
    seed: int = 9999,
) -> dict[str, float]:
    """Evaluate on held-out perturbed samples; 3 step() calls per sample.

    Acceptance criteria (logged; training continues regardless):
        top1_acc   ≥ 0.85
        omega_err  ≤ 10°
        p_err      ≤ 2 cm

    Returns dict with keys: top1_acc, omega_err_deg, p_err_cm, entropy, pass.
    """
    K = int(cfg.refiner.K)
    T = int(cfg.refiner.window_T)
    N = int(_get(cfg, "stage1", "N_points") or 64)

    eval_ds = MockPerturbedDataset(n_eval, K=K, T=T, N=N, seed=seed, informative=True)
    loader  = DataLoader(eval_ds, batch_size=n_eval, shuffle=False, num_workers=0)
    batch   = {k: v.to(device) for k, v in next(iter(loader)).items()}

    prior_output = {
        "omega":      batch["prior_omega"],
        "p":          batch["prior_p"],
        "joint_type": batch["prior_type"],
        "confidence": batch["prior_conf"],
    }
    window = {
        "delta_flow_k": batch["delta_flow_k"],
        "wrench":       batch["wrench"],
        "wrench_res_k": batch["wrench_res_k"],
        "action":       batch["action"],
    }

    model_raw.eval()
    with torch.no_grad():
        hyp = model_raw.init_hypotheses(prior_output)
        for _ in range(3):
            hyp = model_raw.step(hyp, window)
        m = _compute_metrics(hyp, batch["gt_omega"], batch["gt_p"], batch["gt_type_idx"])
    model_raw.train()

    result = {k: float(v.item()) for k, v in m.items()}
    result["pass"] = (
        result["top1_acc"]     >= 0.85
        and result["omega_err_deg"] <= 10.0
        and result["p_err_cm"]      <= 2.0
    )
    return result


# ---------------------------------------------------------------------------
# Core training function (callable without Hydra for smoke tests)
# ---------------------------------------------------------------------------


def train(cfg: DictConfig) -> dict[str, list]:
    """Train the refiner for Stage 1.  Returns per-step metric history.

    Args:
        cfg: Hydra config or OmegaConf dict with the same schema as
             configs/refiner_stage1.yaml.

    Returns:
        history: dict[metric_name → list of float], one entry per log step.
    """
    # ── DDP setup ─────────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    use_ddp    = bool(_get(cfg, "training", "ddp") or False) and (local_rank >= 0)

    if use_ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device   = torch.device(f"cuda:{local_rank}")
        rank     = dist.get_rank()
        world_sz = dist.get_world_size()
        is_main  = (rank == 0)
    else:
        _dev_str = str(_get(cfg, "global", "device") or "cpu")
        device   = torch.device(_dev_str if torch.cuda.is_available()
                                or _dev_str == "cpu" else "cpu")
        rank, world_sz = 0, 1
        is_main  = True

    # ── Reproducibility ────────────────────────────────────────────────────────
    seed = int(_get(cfg, "global", "seed") or 42)
    torch.manual_seed(seed + rank)

    # ── Model ─────────────────────────────────────────────────────────────────
    model_raw = IMMArticulationRefiner(cfg.refiner).to(device)
    if use_ddp:
        model: nn.Module = nn.parallel.DistributedDataParallel(
            model_raw, device_ids=[local_rank]
        )
    else:
        model = model_raw

    # ── Dataset + DataLoader ──────────────────────────────────────────────────
    loader, sampler = build_dataloader(
        build_dataset(cfg), cfg, rank=rank, world_size=world_sz
    )

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    lr           = float(_get(cfg, "training", "lr") or 1e-4)
    wd           = float(_get(cfg, "training", "weight_decay") or 1e-4)
    warmup       = int(_get(cfg, "training", "warmup_steps") or 1000)
    total_steps  = int(_get(cfg, "training", "total_steps") or 100000)
    grad_clip    = float(_get(cfg, "training", "grad_clip") or 1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = cosine_with_warmup(optimizer, warmup, total_steps)

    # ── AMP ───────────────────────────────────────────────────────────────────
    amp_dtype_str = str(_get(cfg, "training", "amp_dtype") or "fp32")
    if amp_dtype_str == "bf16" and device.type == "cuda":
        amp_dtype    = torch.bfloat16
        amp_enabled  = True
    elif amp_dtype_str == "fp16" and device.type == "cuda":
        amp_dtype    = torch.float16
        amp_enabled  = True
    else:
        amp_dtype    = torch.float32
        amp_enabled  = False
    use_scaler = (amp_dtype_str == "fp16")
    scaler     = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # ── Stage-1 hyperparams ───────────────────────────────────────────────────
    lambda_p         = float(_get(cfg, "stage1", "lambda_p") or 100.0)
    acceptance_every = int(_get(cfg, "stage1", "acceptance_every") or 10000)
    log_every        = int(_get(cfg, "logging", "log_every") or 50)
    ckpt_every       = int(_get(cfg, "logging", "ckpt_every") or 5000)

    # ── W&B ──────────────────────────────────────────────────────────────────
    use_wandb = bool(_get(cfg, "logging", "use_wandb") or False) and _WANDB_AVAILABLE
    if is_main and use_wandb:
        wandb.init(
            project = str(_get(cfg, "logging", "project") or "artiresidual"),
            entity  = _get(cfg, "logging", "entity") or None,
            tags    = list(_get(cfg, "logging", "tags") or []),
            config  = OmegaConf.to_container(cfg, resolve=True),
        )

    # ── Checkpointing dir ─────────────────────────────────────────────────────
    ckpt_root = Path(str(_get(cfg, "paths", "ckpt_root") or "/tmp/artiresidual/ckpts"))
    if is_main:
        ckpt_root.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    data_iter = iter(loader)
    epoch     = 0
    history: dict[str, list] = {
        "loss_total": [], "loss_nll": [], "loss_mu_residual": [], "loss_H": [],
        "top1_acc": [], "omega_err_deg": [], "p_err_cm": [], "entropy": [],
    }

    for step in range(total_steps):

        # Refresh data iterator at epoch boundary.
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            batch     = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        # ── Build prior output and evidence window from batch ──────────────────
        prior_output = {
            "omega":      batch["prior_omega"],
            "p":          batch["prior_p"],
            "joint_type": batch["prior_type"],
            "confidence": batch["prior_conf"],
        }
        window = {
            "delta_flow_k": batch["delta_flow_k"],
            "wrench":       batch["wrench"],
            "wrench_res_k": batch["wrench_res_k"],
            "action":       batch["action"],
        }

        # ── Forward pass ───────────────────────────────────────────────────────
        # init_hypotheses has no learnable parameters; no grad needed.
        with torch.no_grad():
            hyp = model_raw.init_hypotheses(prior_output)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_enabled):
            hyp_new = model(hyp, window)   # DDP forward → model.step()
            losses  = model_raw.loss_terms(
                hyp_new,
                batch["gt_omega"],
                batch["gt_p"],
                batch["gt_type_idx"],
                lambda_p=lambda_p,
            )

        # ── Backward ──────────────────────────────────────────────────────────
        if use_scaler:
            scaler.scale(losses["L_total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["L_total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()

        # ── Metrics + logging (rank 0 only) ────────────────────────────────────
        with torch.no_grad():
            m = _compute_metrics(
                hyp_new, batch["gt_omega"], batch["gt_p"], batch["gt_type_idx"]
            )

        # Always record to history (for smoke-test loss-decreasing check).
        history["loss_total"].append(losses["L_total"].item())
        history["loss_nll"].append(losses["L_NLL"].item())
        history["loss_mu_residual"].append(losses["L_mu_residual"].item())
        history["loss_H"].append(losses["L_H"].item())
        history["top1_acc"].append(m["top1_acc"].item())
        history["omega_err_deg"].append(m["omega_err_deg"].item())
        history["p_err_cm"].append(m["p_err_cm"].item())
        history["entropy"].append(m["entropy"].item())

        if is_main and step % log_every == 0:
            cur_lr = scheduler.get_last_lr()[0]
            log_dict = {
                "loss/total":        losses["L_total"].item(),
                "loss/nll":          losses["L_NLL"].item(),
                "loss/mu_residual":  losses["L_mu_residual"].item(),
                "loss/H":            losses["L_H"].item(),
                "metric/top1_acc":   m["top1_acc"].item(),
                "metric/omega_err_deg": m["omega_err_deg"].item(),
                "metric/p_err_cm":   m["p_err_cm"].item(),
                "metric/entropy":    m["entropy"].item(),
                "lr": cur_lr,
            }
            log.info(
                "[step %6d]  loss=%.4f  nll=%.4f  top1=%.3f  "
                "ω_err=%.1f°  p_err=%.2f cm  H=%.3f  lr=%.2e",
                step,
                log_dict["loss/total"], log_dict["loss/nll"],
                log_dict["metric/top1_acc"], log_dict["metric/omega_err_deg"],
                log_dict["metric/p_err_cm"], log_dict["metric/entropy"],
                cur_lr,
            )
            if use_wandb:
                wandb.log(log_dict, step=step)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if is_main and (step + 1) % ckpt_every == 0:
            ckpt = {
                "step":      step + 1,
                "model":     model_raw.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            path = ckpt_root / f"refiner_step{step+1:07d}.pt"
            torch.save(ckpt, path)
            log.info("Checkpoint saved → %s", path)

        # ── Acceptance test ───────────────────────────────────────────────────
        if is_main and (step + 1) % acceptance_every == 0:
            acc = run_acceptance_test(model_raw, cfg, device, lambda_p=lambda_p)
            status = "PASS" if acc["pass"] else "FAIL"
            log.info(
                "[acceptance @%d]  %s  top1=%.3f  ω_err=%.1f°  p_err=%.2f cm",
                step + 1, status, acc["top1_acc"], acc["omega_err_deg"], acc["p_err_cm"],
            )
            if use_wandb:
                wandb.log({
                    "accept/top1_acc":    acc["top1_acc"],
                    "accept/omega_err":   acc["omega_err_deg"],
                    "accept/p_err_cm":    acc["p_err_cm"],
                    "accept/entropy":     acc["entropy"],
                    "accept/pass":        float(acc["pass"]),
                }, step=step)

    # ── Teardown ──────────────────────────────────────────────────────────────
    if use_wandb and is_main:
        wandb.finish()
    if use_ddp:
        dist.destroy_process_group()

    return history


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

if _HYDRA_AVAILABLE:
    @hydra.main(
        config_path="../../configs",
        config_name="refiner_stage1",
        version_base=None,
    )
    def main(cfg: DictConfig) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        )
        log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
        history = train(cfg)
        final = {k: v[-1] for k, v in history.items() if v}
        log.info("Final metrics: %s", final)

else:  # pragma: no cover
    def main(cfg: DictConfig) -> None:  # type: ignore[misc]
        raise RuntimeError("hydra is not installed; cannot use @hydra.main entry point.")


if __name__ == "__main__":
    main()
