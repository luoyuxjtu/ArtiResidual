# Code index — module status & public API

Map of which modules are real vs. stubs. Spec contract lives in `artiresidual_tech_spec.md`; this is the operational status snapshot.

Update on every module flip Not implemented → Implemented.

---

## Implemented

### Module 05 — Analytical flow

`artiresidual/refiner/analytical_flow.py` (companion: `affordance_utils.py`)

Geometric primitives + their differentiable / soft-type variants. Pure math, no learnable parameters.

Public API:
- `analytical_flow(coords_xyz, omega, p, joint_type, *, normalize_per_part=True) -> Tensor`
- `analytical_flow_hard` — alias of `analytical_flow`
- `analytical_flow_diff(coords_xyz, omega, p, joint_type, *, normalize_per_part=True)` — gradient flows through `omega`, `p`
- `analytical_flow_soft(coords_xyz, omega, p, joint_type_logits, *, temperature=1.0)` — gradient through type logits
- `analytical_flow_batched(coords_xyz, omega, p, joint_type)` — `[K, 3]` ω + `[K, 3]` p + `[K]` types → `[K, N, 3]`
- `belief_weighted_flow(coords_xyz, omega, p, joint_type, weights)` — `Σₖ wₖ · f_ana_k`
- `constraint_directions(x_ee, omega, p, joint_type)` — EE-frame constraint vector (used by Module 04's wrench head)
- `normalize_axis(omega, eps=1e-8)`
- Constants: `JOINT_TYPE_REVOLUTE`, `JOINT_TYPE_PRISMATIC`, `JOINT_TYPE_FIXED`

Tests: `tests/test_analytical_flow.py` (5 tests, CPU-only)
Viz: `scripts/visualize_analytical_flow.py`

Companion `affordance_utils.py` exports: `exp_map_sphere`, `clip_axis_correction`, `clip_position_correction`, `cosine_similarity_loss`, `consistency_loss`, `residual_flow`, `residual_flow_summary`, `hypothesis_entropy`, `renormalize_with_floor`, `gt_flow_from_articulation`.

---

### Module 06 — Joint state estimator

`artiresidual/refiner/state_estimator.py`

Pure geometric projection. Converts `current_part_pose: [B, 7]` (from sim GT or SAM2PartTracker — see §3.0) into a scalar `theta_t: [B]`. No ML, no perception.

Public API:
- `JointStateEstimator(omega, p, joint_type, initial_part_pose, *, eps=1e-8)`
  - `.estimate(current_part_pose: [B, 7]) -> Tensor [B]`
  - `.omega`, `.p`, `.joint_type`, `.x_init`, `.q_init` attributes

For K hypotheses: instantiate K separate estimators, stack their outputs (canonical IMM pattern in spec §3 Module 04).

Quat convention: scalar-first `(w, x, y, z)`.

Tests: `tests/test_state_estimator.py` (16 effective tests, CPU-only)
Viz: `scripts/visualize_state_estimator.py`

---

### Module 09 — Benchmark (skeleton with mock env)

`artiresidual/data/benchmark.py`

Multi-handle bimanual benchmark facade. The mock env unblocks downstream development; real RoboTwin/ManiSkill binding is a `# TODO` at `make_env()`.

Public API:
- `MultiHandleBenchmark(sim="robotwin2", config_path=None)`
  - `.list_tasks() -> list[str]` — canonical 12, in spec order
  - `.make_env(task_name, seed=0, **kwargs) -> MockArticulationEnv`
  - `.evaluate(policy, task_name, n_trials=100, max_steps=400, seed_base=0) -> dict`
  - `.SUPPORTED_BACKENDS = ("robotwin2", "maniskill3")`
- `MockArticulationEnv(task_name, seed=0, max_steps=400)`
  - `.reset(*, seed=None, options=None) -> (obs: dict, info: dict)`
  - `.step(action: [7]) -> (obs, reward, terminated, truncated, info)` — Gymnasium 5-tuple
  - `.task_name`, `.max_steps` attributes
- Constants: `TASKS_V1` (12-tuple of task names), `PCD_N_RAW=1024`, `Q_DIM=14`, `WRENCH_DIM=12`, `EE_POSE_SHAPE=(2,7)`, `ACTION_DIM=7`, `MAX_STEPS_DEFAULT=400`

Tests: `tests/test_benchmark.py` (24 effective tests, CPU-only)
Docs: `artiresidual/data/README.md` (per-task selection rationale)

---

### Section 3.0 — SAM2 part tracker (non-trainable upstream)

`artiresidual/utils/part_tracker.py`

SAM2 video predictor + open3d ICP. Produces `current_part_pose` for Module 06 in real-robot eval. Not loaded in sim (gated by `perception.use_gt_part_pose: true`).

Public API:
- `SAM2PartTracker(sam2_checkpoint: str, device: str = "cuda")`
  - `.initialize(rgb_frame: np.uint8 [H, W, 3], part_masks: Sequence[np.ndarray]) -> None`
  - `.estimate(rgb_frame, depth_frame, camera_intrinsics) -> np.ndarray [P, 7]`
    - First call (bootstrap): returns `(centroid, identity_quat)` per part.
    - Subsequent calls: ICP-composed pose relative to bootstrap reference.
  - `.close()` — releases the tempdir frame buffer

Frame convention: returned poses are in the **camera frame** (no extrinsic accepted by `estimate()`).

Smoke: `scripts/smoke_test_part_tracker.py` (4 sequential steps, exits 0 on full pass)
Viz: `scripts/visualize_part_tracker.py` (2×3 panel on a synthetic translating-door scene)
Installer: `scripts/setup_sam2.sh` (server-only)

Sanity verified: codespace syntax check + YAML parse. **Server smoke test not yet run** — pending tomorrow's session.

---

## Not implemented yet

Each entry is a docstring-only stub with the spec section pinned in the file header. Build order in spec §6.

### Module 01 — Prior articulation estimator
`artiresidual/perception/prior_estimator.py`. Spec §3 Module 01. At t=0, predict per-part `(omega, p, joint_type, segmentation, confidence)` from a point cloud. Train on PartNet-Mobility; ≥70 % joint-type accuracy target. Port FlowBot3D backbone.

### Module 02 — Run-time flow predictor
`artiresidual/perception/flow_predictor.py`. Spec §3 Module 02. Every 30 Hz step, predict per-point flow from current pcd. PointNet++ (3 SA + 3 FP) + 3-channel head. Trained against analytical flow GT.

### Module 03 — Stabilizer-actor selector
`artiresidual/selector/role_selector.py`. Spec §3 Module 03. Decide which arm stabilizes vs. acts at t=0; regress grasp SE(3) poses for both. Trained on demo role labels.

### Module 04 — IMM articulation refiner ★ CORE
`artiresidual/refiner/imm_filter.py`. Spec §3 Module 04. **Paper's central technical contribution.** K=3 hypotheses, updated every N=10 steps via residual-flow + wrench evidence. Per-hypothesis transformer encoder + cross-hypothesis attention + tangent-space residual head.

### Module 07 — DiT block
`artiresidual/policy/dit_blocks.py`. Spec §3 Module 07. Single transformer block with AdaLN modulation + **two** cross-attentions (one for `f_cond` tokens, one for entropy token).

### Module 08 — DiT diffusion policy
`artiresidual/policy/dit_policy.py`. Spec §3 Module 08. ~50 M params, v-prediction, cosine schedule, 10-step DDIM. Bimanual policy outputs `[B, 16, 7]` actor-arm chunks.

### Module 10 — Demo generation
`artiresidual/data/demo_gen.py`. Spec §3 Module 10. Scripted expert demos with role labels. Uses cuRobo + RoboTwin. **BLOCKED on RoboTwin install.**

### Module 11 — Training perturbation
`artiresidual/data/perturb.py`. Spec §3 Module 11. Replays demos with perturbed initial `(omega, p, type)` to train the refiner. Depends on Module 10.

### Module 12 — Failure mode analysis
`artiresidual/evaluation/failure_analysis.py`. Spec §3 Module 12. Categorize failures into (a) axis miss, (b) force conflict, (c) handle slip, (d) coordination desync. Applied post-hoc to every rollout.

### Training drivers (Stages 1/2/3)
`artiresidual/training/train_refiner.py`, `train_policy.py`, `train_joint.py`. All stubs. Hydra entry points; depend on the corresponding model modules being implemented first.

### Evaluation drivers
`artiresidual/evaluation/eval_sim.py`, `eval_real.py`. Stubs; thin wrappers over `MultiHandleBenchmark.evaluate()` + failure analysis.

### Utility stubs
`artiresidual/utils/geometry.py`, `artiresidual/utils/visualization.py`. Docstring-only.
