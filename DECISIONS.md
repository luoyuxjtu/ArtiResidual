# Design decisions log

Append-only. Each entry: `## [date] decision  ·  one-line rationale  ·  spec ref`.

The point of this log is so that *future me* (and reviewers) can answer "why did you do X instead of Y?" without re-deriving the trade-off.

---

## 2026-05-14  ·  End-to-end angle error threshold: 1°

Established by running `scripts/e2e_test_state_estimator.py` at σ ∈ {0, 2, 5, 10} mm door-depth noise. At σ=10 mm (far beyond L515 spec of ~1–2 mm) max |err| = 0.831° < 1°. Adopting 1° as the acceptance threshold for the full SAM2PartTracker → JointStateEstimator chain, consistent with the Module 06 unit-test spec bound.  · spec §3 Module 06 + §3.0

## 2026-05-14  ·  `setup_sam2.sh` installs SAM2 with `--no-deps` to pin torch cu121

SAM2 declares `torch>=2.5.1`; a plain `pip install -e sam2` upgraded torch to 2.11+cu130, which broke CUDA on the server's driver-12.8. Fixed by installing SAM2 with `--no-deps` and listing its non-torch deps (`iopath`, `hydra-core`, `tqdm`) explicitly. Torch stays at 2.4.0+cu121.  · n/a (operational)

## 2026-05-13  ·  Quaternion convention: scalar-first `(w, x, y, z)`

Used in `artiresidual/refiner/state_estimator.py` and `artiresidual/utils/part_tracker.py`. Matches PyTorch3D / mujoco / the ML stack we're already on; documented in both module docstrings. The spec doesn't fix this — we did.  · spec §2 silent on quat order

## 2026-05-13  ·  `JointStateEstimator` stores `p` but never reads it

For revolute the angle depends only on the relative orientation; for prismatic the dot product with `ω̂` doesn't need an origin. Kept `p` in the constructor signature for API completeness with the spec and so a position-augmented variant can later reuse the ctor. Protected by `test_revolute_with_offset_hinge_p_is_irrelevant_to_angle`.  · spec §3 Module 06

## 2026-05-13  ·  `SAM2PartTracker` reference PCDs bootstrap on first `estimate()`, not in `initialize()`

The spec API gives `initialize()` only RGB + masks (no depth, no intrinsics), so we cannot compute the reference part PCD at init time. First `estimate()` call therefore *(a)* computes reference PCDs from its own depth, *(b)* returns `(centroid, identity_quat)` per part — and that pair becomes the "initial_part_pose" Module 06 caches. Every subsequent `estimate()` returns the ICP-composed pose relative to that bootstrap.  · spec §3.0

## 2026-05-13  ·  `perception.use_gt_part_pose` is a config toggle, not a class-level switch

Module 06 is identical regardless of source; the caller picks which upstream feeds it. The toggle lives in `configs/base.yaml → perception.use_gt_part_pose`. `SAM2PartTracker` only loads when this is `false`, so sim training never pays SAM2's import cost.  · spec §3.0

## 2026-05-13  ·  `MockArticulationEnv.terminated` is permanently `False` + `info["is_mock"]: True`

The mock has no physics, so it can't honestly report success. Forcing `terminated=False` plus the `is_mock` flag lets training code add a hard assertion (`assert not info["is_mock"]` before logging metrics) and so accidentally training on mock data fails loudly instead of producing meaningless wandb curves.  · spec §3 Module 09

## 2026-05-13  ·  Mock-first for Module 09; real RoboTwin binding deferred

`make_env()` always returns `MockArticulationEnv` regardless of `self.sim`. The dispatch on `self.sim ∈ {robotwin2, maniskill3}` is a `# TODO` at the call site. This unblocks Module 04 / 07 / 08 development today; once RoboTwin is installed only `make_env` needs to change.  · spec §3 Module 09

## 2026-05-13  ·  SAM2 video predictor: tempdir frame buffer + per-call state re-init

The official SAM2 video predictor expects a video directory upfront. For online (streaming) inference we buffer frames in `tempfile.mkdtemp()` and re-call `init_state()` + `propagate_in_video()` each `estimate()`. Simplest correct usage; O(t) per call. In-memory streaming (incremental memory-bank updates) is a future optimization once the smoke + acceptance pass.  · spec §3.0

## 2026-05-13  ·  Module 05: colocate `_hard / _soft / _diff` in `analytical_flow.py`

Originally `_diff` and `_soft` lived in `affordance_utils.py` (companion file). Moved them into `analytical_flow.py` so the file layout matches the spec wording ("three variants in prior code"). `affordance_utils.py` keeps the training utilities (exp_map, clips, losses, entropy, residual_flow).  · spec §3 Module 05

## 2026-05-13  ·  `SAM2PartTracker` returns poses in the **camera frame**

`estimate()`'s signature only accepts intrinsics, not extrinsics. World-frame poses require an extra camera→world rigid transform that the caller must apply. Documented at the class docstring; flagged as a future-API extension point when moving cameras (real-robot eval) come into play.  · spec §3.0 (spec is silent)

## 2026-05-13  ·  Section 3.0 dependencies are lazy-imported

`sam2`, `open3d`, and `PIL` are imported inside functions / the constructor, not at module load. Result: `import artiresidual.utils.part_tracker` works in the codespace even though none of those are installed there. SAM2 only actually loads when somebody instantiates `SAM2PartTracker`.  · spec §3.0 (purely operational)

## 2026-05-13  ·  K hypotheses via K separate `JointStateEstimator` instances, not vectorized

Each instance takes a single (ω, p, joint_type). Multiple types can't be cleanly vectorized inside one instance because the math branches at the type level. Matches spec wording "K instances (one per hypothesis)" and matches how the IMM refiner (Module 04) will naturally construct them.  · spec §3 Module 06

## 2026-05-13  ·  `.gitignore` patterns must be root-anchored

First version had unanchored `data/`, which silently matched `artiresidual/data/` and dropped the entire Python subpackage from the first skeleton commit. Fixed to `/data/`, `/datasets/`, `/demos/`. Lesson: when adding ignore patterns for "dataset directories", anchor them.  · n/a (operational)

## 2026-05-13  ·  12 benchmark task names are an API surface

`TASKS_V1` order is stable. Paper figures index tasks by position; demo-gen and eval assume same order across runs. Procedure for adding a 13th: append, don't insert. Documented in `artiresidual/data/README.md`.  · spec §3 Module 09

## 2026-05-13  ·  Module 06 uses torch even though spec says "no autograd needed"

Spec §3 Module 06 says "no autograd needed". Implementation still uses torch ops because they broadcast cleanly, stay on the caller's device, and incidentally support autograd if the caller wants it (the IMM refiner detaches in practice). A numpy-only rewrite would be a downgrade for negligible benefit.  · spec §3 Module 06
