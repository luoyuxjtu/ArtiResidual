# Server runbook — tests + visualizations

Everything below runs on the Ubuntu GPU server with the `artiresidual` conda env active. **Nothing here works in the codespace** (no torch / no SAM2 / no GPU).

Repository:  `~/artiresidual`
Conda env:   `artiresidual`
Today's commit: `569a84c` (on branch `claude/setup-robot-manipulation-dSiF9`)

---

## 0. First-time setup (skip if already done)

```bash
# Clone the repo (only the first time)
git clone <repo-url> ~/artiresidual

# One-time: create conda env, install torch+cu121 + all extras (~5-10 min)
cd ~/artiresidual
bash setup_server.sh
conda activate artiresidual
wandb login                   # one-time, paste wandb key
```

`setup_server.sh` installs the project in editable mode with `train` + `sim` + `dev` extras, so all CPU tests should work immediately after.

---

## 1. Sync + activate (run at the start of every session)

```bash
cd ~/artiresidual
git pull
conda activate artiresidual
```

The branch is `claude/setup-robot-manipulation-dSiF9`. If `main` is behind, that's expected — we develop on the feature branch.

---

## 2. Run the full CPU test suite (Module 05 + 06 + 09)

```bash
bash scripts/run_tests.sh
```

**Expected:** ~45 tests pass in under 30 seconds. Output ends with `=== 45 passed in X.XXs ===` (count may vary by ±1 if a parametrize fan-out drifts; the per-module breakdown below is authoritative).

**Per-module breakdown:**

| Module | Test file | Count | Notes |
|--------|-----------|-------|-------|
| Module 05 | `tests/test_analytical_flow.py` | 5 | door @ 45° about +z — spec §3 Module 05 acceptance |
| Module 06 | `tests/test_state_estimator.py` | 16 | angles {0, 45, 90, 180}° + displacements {0, 5, 20} cm + noise + batch sizes + K=3 |
| Module 09 | `tests/test_benchmark.py` | 24 | 12 task parametrize + mock env Gymnasium 5-tuple + spec acceptance "12 × 100 steps, no crash" |

To run one module at a time:

```bash
bash scripts/run_tests.sh tests/test_analytical_flow.py        # ~5s
bash scripts/run_tests.sh tests/test_state_estimator.py        # ~5s
bash scripts/run_tests.sh tests/test_benchmark.py              # ~10s
```

Filter by name:

```bash
bash scripts/run_tests.sh tests/test_state_estimator.py -k "at_required_angles or at_required_displacements"
# → 7 tests: pure revolute (4) + pure prismatic (3) — spec §3 Module 06 acceptance core
```

Stop on first failure with full traceback:

```bash
bash scripts/run_tests.sh -x --tb=long
```

---

## 3. Visualizations (CPU-only, write PNGs to `/tmp`)

These do NOT need SAM2 — pure torch + matplotlib.

### Module 05 — analytical flow on a 45° door + a prismatic drawer

```bash
python scripts/visualize_analytical_flow.py --out /tmp/flow_demo.png
```

**What to look for in `/tmp/flow_demo.png`:**
- **Left panel (revolute):** arrow lengths should be visibly **gradient** — short near the hinge, ~6.7× longer at the outer edge of the door. Direction is tangent to the rotation circle.
- **Right panel (prismatic):** all arrows **uniform length + parallel**, aligned with the 45°-in-xy drawer axis.

If the revolute arrows look uniform, that's the `quiver normalize=True` bug — re-pull, the fix is in commit `2d85e82`.

### Module 06 — state estimator diagnostic

```bash
python scripts/visualize_state_estimator.py --out /tmp/state_est.png
```

**stdout summary** prints alongside:
```
[OK] wrote /tmp/state_est.png
     revolute  max error: 0.000XX°  (spec ≤ 1°)
     prismatic max error: 0.000XX mm (spec ≤ 1 mm)
     K-hyp     θ_est:     rev +ẑ=+0.5236 rad, rev +ŷ=+0.0000 rad, prism +x̂=+0.0000 m
     noise     sample std: 5.0XX mm (theory σ_proj = 5.0 mm)
```

**What to look for in `/tmp/state_est.png`** (2×2 layout):
- `[0,0]` revolute sweep — 50 dots on the `y = x` line.
- `[0,1]` prismatic sweep — 50 dots on the `y = x` line.
- `[1,0]` K=3 bars — green ≈ 30° GT, red & purple ≈ 0 (correct collapse).
- `[1,1]` noise histogram — bell around 150 mm, sample std ≈ 5 mm.

---

## 4. Section 3.0 — SAM2 part tracker (GPU-only)

### 4a. One-time SAM2 install (~5–10 min, mostly checkpoint download)

```bash
bash scripts/setup_sam2.sh
```

This script:
- Clones `facebookresearch/sam2` into `$HOME/third_party/sam2` (override via `SAM2_DIR=/path`).
- `pip install -e` from that dir.
- Downloads `sam2_hiera_base_plus.pt` (~80 MB) into `$SAM2_DIR/checkpoints/`.
- `pip install open3d>=0.18`.
- Runs an import sanity check at the end.

Export the checkpoint path so other scripts find it without `--sam2-checkpoint`:

```bash
export SAM2_CHECKPOINT=$HOME/third_party/sam2/checkpoints/sam2_hiera_base_plus.pt
```

Add this `export` to `~/.bashrc` to make it persistent.

### 4b. Smoke test (4 sequential steps, fails fast)

```bash
python scripts/smoke_test_part_tracker.py --sam2-checkpoint $SAM2_CHECKPOINT
```

**Expected stdout** (each step prints `PASS` or `FAIL`):

```
[STEP 1/4] constructing SAM2PartTracker...
[PASS 1/4] construction OK
[STEP 2/4] running initialize()...
[PASS 2/4] initialize() OK
[STEP 3/4] running estimate() — bootstrap call (no SAM2 propagation)...
[PASS 3/4] shape=(1, 7)  pose[0]=[ ... ]
[STEP 4/4] running estimate() — tracked call (SAM2 + ICP)...
[PASS 4/4] shape=(1, 7)  pose[0]=[ ... ]

==============================================================
[OK] all 4 smoke-test steps passed.
==============================================================
```

Exit code 0 on full pass. Each step failing means a specific subsystem broke (see the table in `artiresidual/utils/part_tracker.py`'s module docstring).

### 4c. Visualization (synthetic translating-door scene)

```bash
python scripts/visualize_part_tracker.py \
    --sam2-checkpoint $SAM2_CHECKPOINT \
    --out /tmp/part_tracker_demo.png
```

**stdout summary** prints:

```
[OK] wrote /tmp/part_tracker_demo.png
     IoU=0.XXX  trans=XX.X mm  rot=X.XXX°  (expected: trans ≈ ~47 mm)
```

The synthetic scene shifts the "door" rectangle 20 pixels in x between t=0 and t=1, which at depth=0.6 m and fx=W=256 corresponds to ~47 mm in 3D. ICP should recover that within a few mm if SAM2 propagation produced a sane mask.

**`/tmp/part_tracker_demo.png` is 2×3** — see `scripts/visualize_part_tracker.py` docstring for the panel layout.

---

## 5. Pass criteria (single-line per target)

| Target | "Pass" means |
|--------|--------------|
| `setup_server.sh` | exits 0, prints `[OK] torch <version>  cuda_available=True  device_count=<n>` |
| `bash scripts/run_tests.sh` | exits 0, `=== ~45 passed ===` |
| `test_analytical_flow.py` | 5/5 PASSED |
| `test_state_estimator.py` | 16/16 PASSED |
| `test_benchmark.py` | 24/24 PASSED |
| `visualize_analytical_flow.py` | PNG written, revolute arrows have visible length gradient |
| `visualize_state_estimator.py` | PNG written, stdout reports max errors well under spec bounds |
| `setup_sam2.sh` | exits 0, prints `[OK] sam2:`, `[OK] open3d:`, `[OK] PIL:`, `[OK] numpy:` |
| `smoke_test_part_tracker.py` | exits 0, all 4 `[PASS]` lines printed |
| `visualize_part_tracker.py` | PNG written, ICP `‖Δt‖` within a few mm of 47 mm |

---

## 6. Pulling artifacts back to local (for PR / paper figures)

```bash
# From your laptop:
scp <server>:/tmp/flow_demo.png        .
scp <server>:/tmp/state_est.png        .
scp <server>:/tmp/part_tracker_demo.png .
```

**Do not commit PNGs** — they're `.gitignore`d (large media patterns). If you need a paper figure permanently in the repo, put it under `docs/` and explicitly `git add -f`.

---

## 7. Quick triage cheatsheet

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `pytest` shows `.....` instead of per-test names | someone re-added `-q` to `addopts` | check `pyproject.toml [tool.pytest.ini_options] addopts` |
| `ImportError: torch` in tests | wrong env active | `conda activate artiresidual` |
| `ModuleNotFoundError: artiresidual` | repo not pip-installed | `pip install -e ".[dev]"` from repo root |
| SAM2 smoke step 1 fails | checkpoint missing / config name wrong | `ls $SAM2_CHECKPOINT`; check filename matches `sam2_hiera_base_plus.pt` |
| SAM2 smoke step 4 fails (open3d) | open3d not installed | `pip install open3d>=0.18` |
| `visualize_*.py` says "matplotlib required" | extras not installed | `pip install -e ".[train]"` |
| Tests pass on day 1 but fail on day 2 with no code change | upstream env drift (CUDA / driver / torch update) | rerun `setup_server.sh`; check `nvidia-smi` |
