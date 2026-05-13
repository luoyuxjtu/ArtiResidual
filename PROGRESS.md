# Project progress — ArtiResidual

Living doc. Read **Today's plan** first when resuming work.

---

## Today's plan (next session: 2026-05-15)

1. **Implement Module 04 (IMM refiner ★)** — the paper's core contribution.
   - Read spec §3 Module 04 + §4.3–§4.4 + `docs/PIPELINE_v3.html` first.
   - Sketch per-hypothesis encoder dimensions BEFORE writing code (30-step window × 384-dim feature is the non-trivial piece).
   - Hard deps (Modules 05 + 06) are done; mock Module 01–03 outputs at the shape level per spec §2.5.

2. Settle the **frame-convention TODO**: `SAM2PartTracker.estimate()` returns camera-frame poses. Either extend the API to accept `camera→world`, or document that downstream applies the transform. Decide before Module 04 touches poses.

---

## In progress

Nothing actively in progress — day ended at a clean state (working tree clean, all commits pushed).

---

## Blocked

| Item | Blocker | How it unblocks |
|------|---------|-----------------|
| Module 09 real RoboTwin binding | RoboTwin 2.0 not installed on server | flesh out `scripts/setup_robotwin.sh` (currently stub), pin a commit |
| Module 10 demo_gen | RoboTwin 2.0 + cuRobo install | same; then implement the scripted policy per spec §3 Module 10 |
| Module 11 perturb | Module 10 must produce real demos | depends on Module 10 |
| Section 3.0 acceptance test | needs 10 real cabinet trajectories + mocap GT | physical NERO arm + L515 + mocap setup |
| Step 16 of build order (real-robot eval) | NERO hardware + calibrated camera extrinsic | end-of-project, not blocking now |

---

## Recently done

### 2026-05-13 — day 2, full session

Server-side validation of all 2026-05-13 code. Fixed a torch version bug introduced by SAM2 install. Added end-to-end pipeline test for the SAM2 → JointStateEstimator chain.

**Validation results:**
- ✓ 45/45 CPU tests pass in 3.22 s
- ✓ `visualize_analytical_flow.py` — PNG written, revolute arrows have visible length gradient
- ✓ `visualize_state_estimator.py` — revolute max err 0.000008° / prismatic 0.000000 mm (spec ≤ 1°/1 mm)
- ✓ `setup_sam2.sh` — SAM2 + open3d installed; **bug fixed** (SAM2 upgraded torch to 2.11+cu130, breaking CUDA on driver-12.8 server; fixed with `--no-deps` + explicit dep pins)
- ✓ `smoke_test_part_tracker.py` — 4/4 PASS (`_C` UserWarning is benign per SAM2 docs)
- ✓ `visualize_part_tracker.py` — IoU=1.000, trans=41.16 mm (expected ~47 mm)

**New script:**
- ✓ `scripts/e2e_test_state_estimator.py` — end-to-end revolute door test (synthetic rasterizer → SAM2PartTracker → JointStateEstimator). Establishes **1° as the end-to-end acceptance threshold**:

| Door depth noise σ | max \|err\| | mean \|err\| |
|--------------------|------------|-------------|
| 0 mm               | 0.000°     | 0.000°      |
| 2 mm (L515 typical)| 0.069°     | 0.027°      |
| 5 mm               | 0.316°     | 0.134°      |
| 10 mm (extreme)    | 0.831°     | 0.354°      |

**Commits pushed to `claude/setup-robot-manipulation-dSiF9`:**

| Hash | Subject |
|------|---------|
| `2c933ea` | Fix setup_sam2.sh: install SAM2 with --no-deps to preserve torch cu121 |
| `15bbcbb` | Add e2e state-estimator pipeline test on synthetic revolute door |

---

### 2026-05-13 — day 1, full session

Started from a fresh "Add files via upload" commit (only the tech spec + 2 .py files at root). Ended with project skeleton, three modules implemented or skeleton'd, and Section 3.0 SAM2 tracker shipped.

**Modules implemented (working tree, with tests):**
- ✓ **Module 05** — `artiresidual/refiner/analytical_flow.py`. Ported from student's prior PAct code; `_hard / _soft / _diff` variants colocated. **5 acceptance tests** (door @ 45° about z, matches `ω × (x − p)` to FP precision). 2-panel viz.
- ✓ **Module 06** — `artiresidual/refiner/state_estimator.py`. `JointStateEstimator` with twist-swing decomposition (revolute) + axis projection (prismatic). **16 effective tests** (angles {0°, 45°, 90°, 180°}; displacements {0, 5 cm, 20 cm}; ±0.5 cm ICP-noise robustness for both joint types; batch ∈ {1, 16, 32}; K=3 IMM pattern; sign correctness; FIXED returns 0; p-irrelevance). 2×2 viz.
- ✓ **Module 09 (skeleton)** — `artiresidual/data/benchmark.py`. `MultiHandleBenchmark` + `MockArticulationEnv` exposing the Gymnasium 5-tuple API. **24 effective tests** including spec §3 acceptance ("12 tasks × 100 random-policy steps, no crash"). Real RoboTwin/ManiSkill binding flagged `TODO` at the call site. `artiresidual/data/README.md` documents the per-task rationale.
- ✓ **Section 3.0** — `artiresidual/utils/part_tracker.py`. `SAM2PartTracker` per the new spec section: SAM2 video predictor + open3d ICP. Installer `scripts/setup_sam2.sh`, 4-step smoke `scripts/smoke_test_part_tracker.py`, 6-panel viz `scripts/visualize_part_tracker.py`. **NOT YET run on the server.**

**Infrastructure:**
- Whole project skeleton (8 subpackages, configs, tests dir, scripts dir, docs).
- `.gitignore` (Python + wandb + datasets + checkpoints), root-anchored after we caught it silently dropping `artiresidual/data/` on the first commit.
- `.gitattributes` (force LF).
- `setup_server.sh` (one-click conda env w/ torch + cu121 + extras).
- `pyproject.toml` with `train` / `sim` / `dev` optional extras.
- Hydra configs: `base.yaml`, `refiner/{imm_k3, imm_k2_minimum}.yaml`, `policy/{dit_50m, dit_lora_rdt}.yaml`, `task/{multi_handle, single_handle}.yaml`. Added `perception.use_gt_part_pose: true` toggle for spec §3.0.
- `tests/conftest.py` with `@pytest.mark.gpu / sim` markers auto-skipping in codespace.
- `scripts/run_tests.sh` generic pytest wrapper.

**Spec change handled mid-day:**
- User pushed `47fcf91 Update artiresidual_tech_spec.md` adding **Section 3.0** (SAM2PartTracker) and reframing Module 06 as a "pure geometric conversion" (no perception). Algorithm unchanged. Module 06 docstrings rewritten to point at Section 3.0 as the canonical upstream source and to spell out the `use_gt_part_pose` sim/real toggle.

**Bug fixes from same-day issues:**
- pytest verbosity cancellation: `addopts = "-ra -q"` was eating the user's `-v` flag (pytest treats `-q` and `-v` as a single counter).
- Quiver `normalize=True` was hiding `|ω × (x − p)|` magnitude variation in the revolute viz panel.
- `.gitignore`'s unanchored `data/` was nuking the `artiresidual/data/` Python subpackage.

**Commits pushed to `claude/setup-robot-manipulation-dSiF9`:**

| Hash | Subject |
|------|---------|
| `c9dc778` | Scaffold ArtiResidual project skeleton per tech spec |
| `536bc4b` | Colocate the three analytical-flow variants in Module 05 |
| `2d85e82` | Fix pytest -v cancellation + revolute quiver normalization |
| `7197110` | Implement Module 06 JointStateEstimator with 27 unit tests |
| `dae83d7` | Add Module 06 state-estimator visualization script |
| `06153c8` | Implement Module 09 skeleton with MockArticulationEnv |
| `543e0c5` | Implement Section 3.0 SAM2PartTracker + installer + smoke + viz |
| `569a84c` | Align Module 06 with updated spec §3 Module 06 + §3.0 |
