# ArtiResidual

Bimanual diffusion policy with **self-correcting articulation perception**. A residual-flow IMM (Interacting Multiple Model) refiner lets the robot recover from an incorrect initial guess of a joint's axis / type / location *during* manipulation, rather than failing open-loop.

- Authoritative spec: **[`artiresidual_tech_spec.md`](./artiresidual_tech_spec.md)** — module-by-module API, tensor shapes, acceptance tests.
- Visual design: [`docs/PIPELINE_v3.html`](./docs/PIPELINE_v3.html), [`docs/DIT_ARCHITECTURE.html`](./docs/DIT_ARCHITECTURE.html).
- Paper target: AAAI 2027 (~Aug 2026).

---

## Workflow: GitHub for code, Ubuntu server for execution

Code is written and version-controlled here on GitHub (Claude Code edits + light CPU-only checks). Training, demo generation, simulator rollouts, and any GPU-using code run **only on the Ubuntu GPU server** (2× A800 80GB). The codespace does not, and cannot, run GPU code.

```
+-------------------------------+   git push   +-------------------------------+
| GitHub repo / codespace       | -----------> | Ubuntu server (2x A800)       |
|                               |              |                               |
| - Claude Code edits           |              | - conda env: artiresidual     |
| - ruff / mypy / pytest (CPU)  |              | - RoboTwin 2.0 / ManiSkill 3  |
| - hydra config validation     |              | - training + sim rollouts     |
| - no torch import needed      | <----------- | - wandb logging               |
+-------------------------------+   git pull   +-------------------------------+
```

### The loop

1. **Edit on GitHub.** Claude Code modifies code, runs `ruff check` and `pytest -m "not gpu"`.
2. **Commit & push** to the working branch.
3. **On the server**, `git pull` and run the exact command Claude prints.
4. **Watch wandb** for the panels Claude pointed to.
5. Pull *small* artifacts back into the repo (figures, eval JSON). Never commit checkpoints, datasets, or videos — they're `.gitignore`d.

> **Rule for Claude Code:** never `python -m artiresidual.training.*`, never `python scripts/gen_demos.py`, never launch a simulator from the codespace. Always emit: (a) the server-side command, (b) expected wall-clock, (c) which wandb panels to watch. Then stop.

---

## Quickstart

### One-time setup on the Ubuntu server

```bash
git clone <repo-url> ~/artiresidual
cd ~/artiresidual
bash setup_server.sh        # creates conda env "artiresidual" w/ torch+cu121
conda activate artiresidual
wandb login
```

`setup_server.sh` installs PyTorch with CUDA 12.1, this package in editable mode, and the `train` + `sim` + `dev` extras. RoboTwin 2.0 and ManiSkill 3 are installed separately (see `scripts/setup_robotwin.sh` once it exists — they have their own non-pip install steps).

### Every server session

```bash
cd ~/artiresidual
git pull
conda activate artiresidual
# then run the command Claude printed, e.g.:
# python -m artiresidual.training.train_refiner --config-name=base
```

### In the codespace (Claude's environment)

```bash
pip install -e ".[dev]"       # no torch, no CUDA
ruff check .
pytest -m "not gpu"
```

GPU-requiring tests are marked `@pytest.mark.gpu` in `tests/conftest.py` and skipped by default in the codespace.

---

## Repository layout

```
artiresidual/
├── perception/        # Modules 01-02: prior + run-time flow predictor
├── refiner/           # Modules 04-06: IMM filter ★, analytical flow, state estimator
├── selector/          # Module 03: stabilizer-actor selector
├── policy/            # Modules 07-08: DiT block + DiT bimanual policy
├── data/              # Modules 09-11: benchmark, demo gen, perturb
├── training/          # Stage 1/2/3 trainers
├── evaluation/        # eval_sim, eval_real, failure_analysis (Module 12)
└── utils/             # geometry, visualization

configs/               # Hydra: base.yaml + refiner/, policy/, task/
scripts/               # server-side setup + one-shot scripts
tests/                 # unit tests, CPU-only by default
docs/                  # design HTMLs
```

See `artiresidual_tech_spec.md` §1 for the canonical structure, §3 for module specs, §6 for build order.

---

## Status

**Skeleton commit.** Only Module 05 (analytical flow, `artiresidual/refiner/analytical_flow.py`) and its differentiable companion (`affordance_utils.py`) are ported from the student's prior PAct work. Everything else is a stub awaiting implementation; build order is fixed in spec §6.
