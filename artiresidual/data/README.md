# Multi-Handle Bimanual Benchmark — 12 Tasks

The benchmark targets the **coordination problem** in multi-handle bimanual articulated manipulation: a single object with two or more independently movable parts, where both arms must act and the choice of *who stabilizes vs. who actuates* is decisive.

Most prior benchmarks (PartNet-Mobility tasks, FlowBot3D, ArticuBot, single-arm DP3) test single-handle objects or single-arm execution. Failures driven by **inter-arm coordination** — force conflict, role misassignment, sequential dependency, opposing torques — are largely invisible in those settings. This benchmark is built to expose them, and is *Innovation 1* of the ArtiResidual paper.

See `artiresidual_tech_spec.md` §3 Module 09 for the API contract; this README explains the **selection rationale**.

---

## Selection axes

Each of the 12 tasks contributes at least one combination along these four axes:

| Axis | Values that appear in the benchmark |
|------|-------------------------------------|
| Joint-type mixture | revolute×revolute / revolute×prismatic / prismatic×prismatic / non-articulated bimanual / continuous rotation |
| Coordination mode | parallel-both-act / one-stabilize-one-act / opposing-forces / sequential-dependent |
| Object class | cabinet, drawer, fridge, microwave, oven, suitcase, bottle, laptop, storage box, pot |
| Difficulty | easy (symmetric parallel) → hard (continuous rotation with required stabilization) |

The 12 tasks are chosen so every combination is hit at least once, with a deliberate over-sampling of the "mixed type" and "one-stabilize-one-act" patterns, which are exactly the regimes most underrepresented in prior benchmarks.

---

## Per-task table

| # | Task | Joint types | Coordination | Difficulty | Why it's in |
|---|------|-------------|--------------|-----------|-------------|
| 1 | `open_double_door_cabinet` | rev + rev | parallel both-act | easy | Canonical "two handles, no stabilizer" baseline. Symmetric — should be the easiest task and a sanity-rate ceiling. |
| 2 | `open_double_drawer` | pris + pris | parallel both-act | easy | Same coordination as #1, pure prismatic. Pair (#1, #2) isolates joint-type discrimination from coordination — if a method does well on #1 but not #2, its issue is type estimation, not coordination. |
| 3 | `open_fridge_with_freezer` | rev + rev | parallel both-act, asymmetric mass | medium | Two doors of very different size / friction. Tests role asymmetry and stabilizer-aware force allocation. |
| 4 | `open_microwave_with_tray` | rev + pris | sequential (door, then tray) | medium | Mixed type inside one object + temporal dependency. The actor's plan must change kinematic model mid-trajectory. |
| 5 | `lift_pot_by_two_handles` | rigid bimanual (no articulation) | opposing forces | medium | No joint to estimate; the only failure mode is force conflict (Module 12 category b). Pins down how much of our improvement comes from wrench feedback vs. articulation refinement. |
| 6 | `open_suitcase_two_clasps` | rev + rev (+ rev lid) | sequential-dependent (clasps **before** lid) | hard | Multi-step bimanual with a hard ordering constraint. The dominant failure mode is coordination desync (Module 12 category d). |
| 7 | `open_oven_door_and_tray` | rev + pris | one-stabilize-one-act (door held open, tray pulled) | medium | The cleanest stabilizer-actor case. Without a stabilizer the door swings closed before the tray is out. |
| 8 | `open_double_door_microwave` | rev + rev | parallel both-act | easy-medium | Smaller scale than #1 → handle-reachability constraints become tighter. Tests grasp-pose regression accuracy of Module 03. |
| 9 | `unscrew_bottle` | rev (continuous, no joint limit) | one-stabilize-one-rotate | hard | Continuous rotation past π. Stress-tests both rotation-tracking (state estimator at large θ) and grip slip (category c). |
| 10 | `open_laptop` | rev + (slide-stabilize on table) | one-stabilize-one-act | hard | Base slides on the table — without an active stabilizer the base flies away mid-lift. The "if you skip stabilizer, you fail" case. |
| 11 | `open_storage_box` | rev (+ content access) | sequential | easy-medium | Simple lid + access pattern; serves as a sanity-rate floor under sequential reasoning. |
| 12 | `open_cabinet_with_drawer` | rev + pris | mixed sequential | medium | Mirrors the most common real-world multi-handle furniture pattern (kitchen, bedroom). |

---

## Mapping to the paper's failure-mode analysis

Spec §3 Module 12 categorizes failures into four buckets. Each task is included at least partly because it exposes one of them.

| Failure mode | Highest-leverage tasks |
|--------------|------------------------|
| (a) joint axis misestimation | every task — but #2 (drawer), #4 (microwave+tray), #7 (oven+tray) are the highest-leverage since prismatic joints make axis errors immediately visible in residual flow. |
| (b) inter-arm force conflict | #5 (lift_pot), #9 (unscrew_bottle) — explicitly opposing-force tasks. |
| (c) handle slip | #9 (unscrew), #10 (laptop) — large grip torques / lateral forces. |
| (d) coordination desync | #6 (suitcase), #11 (storage_box) — sequential ordering required. |

The paper's argument is: our IMM refiner specifically improves (a); wrench-feature conditioning specifically improves (b); we should be comparable to baselines on (c) and (d). The benchmark distribution above lets reviewers verify this claim directly.

---

## Coordination-mode distribution (12 tasks → 5 modes)

| Mode | Count | Tasks |
|------|-------|-------|
| Parallel both-act | 4 | #1, #2, #3, #8 |
| One-stabilize-one-act | 3 | #7, #10, plus #9's stabilizer-rotator pattern |
| Opposing forces | 1 | #5 |
| Sequential-dependent | 3 | #4, #6, #11 |
| Mixed-type sequential | 1 | #12 |

Parallel-both-act is oversampled (4/12) because that's where the symmetry-breaking decision (which arm grabs which handle) is most ambiguous — it's where the **stabilizer-actor selector** (Module 03) has the easiest baselines to beat.

---

## Why no continuous-control non-articulated tasks?

A reasonable critique is that our benchmark is *all* articulated-object tasks (except #5). We made that choice because the paper's mechanism — IMM articulation refinement — only adds value on objects with non-trivial joint kinematics. Including, say, peg-in-hole tasks would let us claim a "general bimanual" benchmark, but it would dilute the signal: a method that's stronger on articulation but weaker on peg-in-hole would be hard to interpret. Future versions (v2) may add non-articulated tasks once the articulation story is settled.

---

## How to add a 13th task (later)

The intended template (once RoboTwin 2.0 is wired up):

1. Drop the URDF under `data/urdfs/<task_name>/`.
2. Subclass `robotwin2.envs.BaseTask` in `artiresidual/data/tasks/<task_name>.py`.
3. Implement `_get_success_criterion()` and `_get_initial_pose_distribution()`.
4. Add the task name to `TASKS_V1` in `benchmark.py` (preserving the canonical order — append, don't insert mid-list).
5. Document the new task in this README under "Per-task table" with explicit failure-mode mapping.

The order of `TASKS_V1` is **part of the API** — paper figures index tasks by position, and demo gen / eval scripts assume the same order across runs.
