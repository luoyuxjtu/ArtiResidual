# ArtiResidual Technical Specification

> **For Claude Code**: This is the authoritative technical spec for the ArtiResidual project. Read this entire document before writing any code. When asked to implement a module, refer back to the relevant section here for API contracts, dimension specs, and reference repos.

---

## 0. Project Identity

**Name**: ArtiResidual

**One-sentence pitch**: A bimanual diffusion policy framework where the residual between predicted and observed articulation flow drives in-execution refinement of joint kinematics beliefs, enabling robust manipulation under articulation uncertainty.

**Paper target**: AAAI 2027 (submission deadline ~Aug 1, 2026)

**Hardware**:
- 2× AgileX NERO 7-DoF arms (with force feedback, ±0.1mm repeatability, redundant kinematics)
- 2× A800 80GB GPUs
- Intel RealSense L515 cameras (top + 2× wrist) — NOT D435 (DP3 fails with D435 point clouds)
- (Optional but recommended) 2× ATI Mini45 or Bota SensONE wrist F/T sensors

**Simulation**:
- Primary: RoboTwin 2.0 (https://github.com/RoboTwin-Platform/RoboTwin) — bimanual native, AgileX support
- Secondary: ManiSkill 3 (https://github.com/haosulab/ManiSkill) — for cross-sim ablation

**Core innovation (single mechanism)**: Self-correcting articulation perception via residual flow + wrench evidence in a bimanual setting.

---

## 1. Repository Structure

Create this structure under `/workspace/artiresidual/`:

```
artiresidual/
├── README.md
├── pyproject.toml
├── configs/
│   ├── base.yaml
│   ├── refiner/
│   │   ├── imm_k3.yaml
│   │   └── imm_k2_minimum.yaml
│   ├── policy/
│   │   ├── dit_50m.yaml
│   │   └── dit_lora_rdt.yaml
│   └── task/
│       ├── single_handle.yaml
│       └── multi_handle.yaml
├── artiresidual/
│   ├── __init__.py
│   ├── perception/
│   │   ├── flow_predictor.py      # Module 02
│   │   └── prior_estimator.py     # Module 01
│   ├── refiner/
│   │   ├── imm_filter.py          # Module 04 ★
│   │   ├── analytical_flow.py     # Module 05
│   │   └── state_estimator.py     # Module 06
│   ├── selector/
│   │   └── role_selector.py       # Module 03
│   ├── policy/
│   │   ├── dit_blocks.py          # Module 07
│   │   ├── dit_policy.py          # Module 08
│   │   └── diffusion.py
│   ├── data/
│   │   ├── benchmark.py           # Module 09
│   │   ├── demo_gen.py            # Module 10
│   │   └── perturb.py             # Module 11
│   ├── training/
│   │   ├── train_refiner.py       # Stage 1
│   │   ├── train_policy.py        # Stage 2
│   │   └── train_joint.py         # Stage 3
│   ├── evaluation/
│   │   ├── eval_sim.py
│   │   ├── eval_real.py
│   │   └── failure_analysis.py    # Module 12
│   └── utils/
│       ├── geometry.py
│       └── visualization.py
├── scripts/
│   ├── setup_robotwin.sh
│   ├── gen_demos.py
│   └── run_baselines.sh
├── tests/
│   └── ... (unit tests per module)
└── docs/
    ├── PIPELINE_v3.html
    └── DIT_ARCHITECTURE.html
```

---

## 2. Naming and Tensor Shape Conventions

These are LAW. Every module follows them.

### 2.1 Naming
- `omega` (ω): joint axis direction, unit 3-vector
- `p`: joint axis origin point, 3-vector (in object/world frame)
- `joint_type`: 0=revolute, 1=prismatic
- `theta_t`: scalar joint configuration at time t
- `f_pred`: per-point predicted flow from perception net, [N, 3]
- `f_ana`: per-point analytical flow from (ω, p, type, θ_t), [N, 3]
- `delta_flow`: f_pred − f_ana, [N, 3]
- `K`: number of hypotheses (default 3)
- `w_k`: hypothesis weights, [K], sums to 1
- `H_w`: entropy of w_k, scalar
- `B`: batch size
- `T`: time / window length
- `f_cond`: belief-weighted analytical flow Σₖ wₖ · f_ana_k, [N, 3]

### 2.2 Point cloud
- Object point cloud: `pcd` of shape `[B, N, 3]` where N=1024 (raw) or N=128 (after FPS)
- Always coordinates in WORLD frame unless explicitly noted

### 2.3 Robot state
- Bimanual joint positions: `q` of shape `[B, 14]` (7 left + 7 right)
- Bimanual wrench: `wrench` of shape `[B, 12]` (6 left + 6 right; [Fx,Fy,Fz,Mx,My,Mz])
- End-effector pose: `ee_pose` of shape `[B, 2, 7]` (per arm: 3 pos + 4 quat)

### 2.4 Action
- Action chunk: `action` of shape `[B, T_a, 7]` where T_a=16, 7-DoF actor delta in joint space
- Stabilizer action is NOT in this tensor; it's a separate one-shot pose

### 2.5 Hypothesis
- A hypothesis is a dict: `{omega: [3], p: [3], joint_type: int, weight: float}`
- Batched: `omega_k: [B, K, 3]`, `p_k: [B, K, 3]`, `type_k: [B, K]` (long), `w_k: [B, K]`

---

## 3. Module Specifications

For each module: purpose, exact API, reference repo, implementation notes, acceptance test.

### Module 01: Prior Articulation Estimator

**File**: `artiresidual/perception/prior_estimator.py`

**Purpose**: At task start (t=0), produce initial estimate of (ω, p, joint_type) for each manipulable part of the articulated object. This estimate may be wrong — that's WHY the refiner exists. Don't over-engineer this module.

**API**:
```python
class PriorArticulationEstimator(nn.Module):
    def __init__(self, config):
        # config.backbone: 'flowbot3d' | 'pointnet++'
        ...
    
    def forward(self, pcd: Tensor) -> Dict:
        """
        Args:
            pcd: [B, N, 3] world-frame point cloud at t=0
        Returns:
            {
                'omega': [B, P, 3],          # P parts detected, axis per part
                'p': [B, P, 3],              # origin per part
                'type_logits': [B, P, 2],    # logits over [revolute, prismatic]
                'part_segmentation': [B, N, P],  # per-point part assignment
                'confidence': [B, P]         # estimator's own confidence per part
            }
        """
```

**Reference repo**: Reuse PAct/FlowBot3D backbone:
- FlowBot3D: https://github.com/r-pad/flowbot3d
- PartNet-Mobility loading: SAPIEN URDF parser

**Implementation notes**:
- Train as standard supervised predictor on PartNet-Mobility URDF ground truth.
- Student's prior PAct work likely contains a similar network — port it.
- DO NOT spend more than 2 weeks on this. Once it's "decent" (≥70% accuracy on held-out), move on. The refiner cleans up the rest.

**Acceptance test**:
- On held-out PartNet-Mobility cabinets/drawers: joint type accuracy ≥ 70%, ω angle error ≤ 30° (90th percentile), p error ≤ 5 cm.

---

### Module 02: Run-Time Flow Predictor

**File**: `artiresidual/perception/flow_predictor.py`

**Purpose**: At every control step during execution, predict per-point articulation flow from current point cloud. This is `f_pred`.

**CRITICAL**: This is NOT the same network as the prior estimator. It runs at 30 Hz on every frame.

**API**:
```python
class FlowPredictor(nn.Module):
    def __init__(self, config):
        ...
    
    def forward(self, pcd: Tensor) -> Tensor:
        """
        Args:
            pcd: [B, N, 3] world-frame, N=128 (FPS-downsampled)
        Returns:
            f_pred: [B, N, 3] per-point predicted articulation flow
        """
```

**Architecture**:
- PointNet++ backbone (3 SA layers + 3 FP layers)
- Input: pcd [B, 128, 3]
- Output: 3-channel per-point flow vector

**Reference**:
- FlowBot3D's flow head: https://github.com/r-pad/flowbot3d
- DP3 point cloud encoder: https://github.com/YanjieZe/3D-Diffusion-Policy/tree/master/3D-Diffusion-Policy/diffusion_policy_3d/model/vision

**Training supervision**:
- Use simulation ground truth: from PartNet-Mobility URDF, compute analytical flow at each timestep as the GT for `f_pred`.
- Loss: cosine + L2 on per-point flow vectors.

**Acceptance test**:
- On held-out trajectories: per-point cosine similarity with GT flow ≥ 0.85, L2 error ≤ 0.02 m/step.

**Key design decision**: Optionally initialize from PAct/FlowBot3D pretrained weights for fast convergence.

---

### Module 03: Stabilizer-Actor Selector

**File**: `artiresidual/selector/role_selector.py`

**Purpose**: At task start, decide which arm stabilizes which part and which arm acts on which part. Outputs grasp poses for both arms.

**API**:
```python
class StabilizerActorSelector(nn.Module):
    def forward(
        self,
        pcd_feat: Tensor,           # [B, N, D] from perception backbone
        omega: Tensor,              # [B, P, 3]
        p: Tensor,                  # [B, P, 3]
        type_logits: Tensor,        # [B, P, 2]
        part_seg: Tensor,           # [B, N, P]
    ) -> Dict:
        """
        Returns:
            {
                'left_role': [B, P],   # one-hot: which part for left arm
                'right_role': [B, P],  # one-hot: which part for right arm
                'left_grasp_pose': [B, 7],   # 3 pos + 4 quat
                'right_grasp_pose': [B, 7],
                'role_assignment': [B] ∈ {'left_stab', 'right_stab', 'both_act'}
            }
        """
```

**Architecture**:
- 4-layer transformer encoder, dim=256, 4 heads
- Input tokens: per-point features (128) + per-part summary tokens (P, typically 2)
- Output heads: role assignment + per-arm grasp pose (regression on SE(3))

**Reference**:
- VoxAct-B's stabilizer-actor formulation: https://github.com/VoxAct-B/voxact-b

**Training**:
- Supervised by scripted demo data (the demos already have explicit role labels — see Module 10).
- Loss: cross-entropy on role + L2 on pose + quaternion geodesic distance.

**Implementation notes**:
- This is a "borrowed" component. Don't innovate here. Implement cleanly and move on.
- Selector runs once at t=0; afterward roles are fixed.

**Acceptance test**:
- Role accuracy ≥ 90%, grasp pose error ≤ 2 cm position, ≤ 15° orientation on held-out demos.

---

### Module 04: IMM Articulation Refiner ★ (CORE MODULE)

**File**: `artiresidual/refiner/imm_filter.py`

**Purpose**: The HEART of the paper. Maintains K parallel hypotheses about (ω, p, joint_type) and refines them every N control steps using residual flow + wrench evidence.

**API**:
```python
class IMMArticulationRefiner(nn.Module):
    def __init__(self, config):
        # config.K: number of hypotheses (default 3)
        # config.window_T: history window (default 30)
        # config.update_interval_N: update every N steps (default 10)
        # config.dim: 256
        ...
    
    def init_hypotheses(self, prior_output: Dict) -> Dict:
        """
        Initialize K hypotheses from prior estimator output.
        Returns:
            {
                'omega_k': [B, K, 3],
                'p_k': [B, K, 3],
                'type_k': [B, K],     # long: 0=rev, 1=pris
                'w_k': [B, K],         # weights, sum to 1
            }
        """
    
    def step(
        self,
        hypotheses: Dict,
        window: Dict,
    ) -> Dict:
        """
        Refine hypotheses given last T steps of evidence. Called every N control steps.
        """
    
    def get_f_cond(
        self,
        hypotheses: Dict,
        theta_t: Tensor,           # [B]
        pcd: Tensor,                # [B, N, 3]
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute belief-weighted analytical flow + entropy.
        Returns:
            f_cond: [B, N, 3]    # Σₖ wₖ · f_ana(ωₖ, pₖ, typeₖ, θ_t)
            entropy: [B]          # H(w)
        """
```

**Architecture details**:
1. **Per-hypothesis encoder** (shared weights across K hypotheses):
   - 4-layer transformer encoder, dim=256, 4 heads
   - Input: per-step feature vector of dim 384 over window of T=30 steps
   - Per-step feature = concat of:
     - Δ_flow_k for this hypothesis: PointNet-mini (3 layers, 64→128→256) → mean-pool → 256 dim
     - Wrench (12 dim) + per-hypothesis wrench residual (3 dim) → MLP → 64 dim
     - Action (14 dim) → MLP → 64 dim
   - Mean-pool over T to get a 256-dim "hypothesis summary" token
2. **Cross-hypothesis attention** (1 layer, 4 heads) over K hypothesis tokens → IMM mode-mixing
3. **Output heads**:
   - Log-likelihood score `ℓ_k` ∈ ℝ per hypothesis → weight update: `w_k_new ∝ w_k * exp(ℓ_k)`
   - Residual correction `Δμ_k ∈ ℝ⁶` per hypothesis in tangent space:
     - For ω: 3-vector in tangent of S² at current ωₖ, applied via exp map, **clipped to 30° cone**
     - For p: 3-vector with L² norm **clipped to 5 cm**
   - Apply with learning rate η=0.5: `μₖ ← exp_map(μₖ, η·Δμₖ)`

**Reference repos**:
- Differentiable filter patterns: https://github.com/akloss/differentiable_filters
- PF-net (NOT used, for reference): https://github.com/AdaCompNUS/pfnet
- IMM classical: implement from Blom & Bar-Shalom 1988 formulation

**Implementation notes**:
- K=3 hypotheses: {h₁=revolute-vertical-axis, h₂=revolute-horizontal-axis, h₃=prismatic}
- Initialize from prior estimator: top hypothesis weighted 0.6, others 0.2 each
- Weight floor `w_min = 0.05` (prevent permanent collapse)
- Entropy regularization in training: `L_H = -λ_H · H(w)` with `λ_H = 0.01`
- **DO NOT add hypothesis spawning/pruning** in v1. K=3 fixed throughout.

**Acceptance test**:
- On held-out perturbed trajectories (ω perturbed by Uniform(5°,30°), p by σ=2cm, type swap with p=0.2):
  - Within 3 update steps (30 control steps), top-1 hypothesis accuracy ≥ 85%
  - ω angle error ≤ 10° after refinement
  - p error ≤ 2 cm after refinement

---

### Module 05: Analytical Flow

**File**: `artiresidual/refiner/analytical_flow.py`

**Purpose**: Compute f_analytical from (ω, p, joint_type, θ_t). Pure geometric, no learning.

**API**:
```python
def analytical_flow(
    pcd: Tensor,         # [B, N, 3] points in world frame
    omega: Tensor,       # [B, 3]
    p: Tensor,           # [B, 3]
    joint_type: int,     # 0=revolute, 1=prismatic
    theta_t: Tensor,     # [B] current joint configuration
    delta_theta: float = 0.01,
) -> Tensor:             # returns [B, N, 3]
    """
    Compute instantaneous articulation flow:
      revolute: f(x) = ω × (x − p) · δθ
      prismatic: f(x) = ω · δθ
    """
```

**Reference**: Student already has this from prior PAct work (`affordance_utils.py` with `analytical_flow_*` functions). Port directly.

**Implementation notes**:
- Must be fully differentiable (for refiner training).
- Three variants in prior code: hard, soft, diff. Use the **hard** version for run-time.

**Acceptance test**:
- For known synthetic configurations (door at 45° about z-axis), computed flow matches analytical expectation to floating-point precision.

---

### Module 06: State Estimator

**File**: `artiresidual/refiner/state_estimator.py`

**Purpose**: At every control step, estimate θ_t from part pose change since t=0.

**API**:
```python
class JointStateEstimator:
    def __init__(self, omega, p, joint_type, initial_part_pose):
        # Stores reference state at t=0
        ...
    
    def estimate(self, current_part_pose: Tensor) -> Tensor:
        """
        Args:
            current_part_pose: [B, 7] (3 pos + 4 quat) of articulated part at current t
        Returns:
            theta_t: [B] current joint configuration
        """
```

**Implementation notes**:
- For revolute: project rotation from initial to current pose onto axis ω.
- For prismatic: project translation from initial to current pose onto axis ω.
- Need K instances (one per hypothesis).
- ~150 lines of Python. No ML.

**Acceptance test**:
- On synthetic trajectories with known θ_t: error ≤ 1° (revolute) or ≤ 1 mm (prismatic).
- Robust to ±0.5 cm Gaussian noise on part pose.

---

### Module 07: DiT Block (with dual cross-attention)

**File**: `artiresidual/policy/dit_blocks.py`

**Purpose**: A single DiT transformer block with TWO cross-attention layers — one for belief-weighted flow, one for entropy.

**API**:
```python
class ArtiResidualDiTBlock(nn.Module):
    def __init__(self, dim=512, n_heads=8, mlp_ratio=4):
        ...
    
    def forward(
        self,
        x: Tensor,             # [B, S, dim] action+context sequence
        t_emb: Tensor,         # [B, dim] timestep embedding
        f_cond_tokens: Tensor, # [B, 128, dim] per-point belief flow tokens
        entropy_token: Tensor, # [B, 1, dim] uncertainty token
    ) -> Tensor:
        ...
```

**Architecture** (precisely as in DIT_ARCHITECTURE.html):
1. LayerNorm + AdaLN modulation (scale_msa, shift_msa from t_emb)
2. Self-Attention with AdaLN gate
3. Residual connection
4. LayerNorm (no AdaLN)
5. **Cross-Attention 1: Q=x, K=V=f_cond_tokens**
6. **Cross-Attention 2: Q=x, K=V=entropy_token**
7. Residual connection (sum of both cross-attn outputs)
8. LayerNorm + AdaLN modulation (scale_mlp, shift_mlp)
9. FFN (Linear→GELU→Linear) with AdaLN gate
10. Residual connection

**Reference**:
- DiT (Peebles & Xie, ICCV 2023): https://github.com/facebookresearch/DiT — for AdaLN
- RDT-1B: https://github.com/thu-ml/RoboticsDiffusionTransformer — for bimanual diffusion

**Implementation notes**:
- `adaLN_modulation` outputs 6×dim (scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp).
- Modulate: `modulate(x, scale, shift) = x * (1 + scale) + shift`.
- Cross-attention is NOT modulated by timestep (key design — conditioning enters cleanly).

**Acceptance test**:
- Forward pass with dummy inputs of correct shapes returns correct output shape.
- Gradient flow test: backward pass produces gradients on all parameters.

---

### Module 08: Full DiT Policy

**File**: `artiresidual/policy/dit_policy.py`

**Purpose**: The full bimanual diffusion policy that consumes belief + observation and outputs actor-arm action chunks.

**API**:
```python
class ArtiResidualDiTPolicy(nn.Module):
    def __init__(self, config):
        # config.dim: 512, config.depth: 8, config.n_heads: 8
        # config.action_horizon: 16, config.action_dim: 7
        ...
    
    def forward(
        self,
        pcd: Tensor,           # [B, 128, 3]
        proprio: Tensor,       # [B, 14]
        wrench: Tensor,        # [B, 12]
        stab_pose: Tensor,     # [B, 7]
        theta_t: Tensor,       # [B]
        f_cond: Tensor,        # [B, 128, 3] from refiner.get_f_cond
        entropy: Tensor,       # [B]
        action_noisy: Tensor,  # [B, 16, 7] noisy action chunk
        diffusion_t: Tensor,   # [B] timestep
    ) -> Tensor:               # predicted velocity [B, 16, 7]
        ...
    
    def sample(self, /* same args without action_noisy/diffusion_t */) -> Tensor:
        """DDIM sampling, 10 steps. Returns [B, 16, 7] clean action chunk."""
```

**Architecture**:
- Input encoders: PointNet++ (pcd→[B,128,512]) + MLPs (proprio, wrench, stab_pose, theta_t) each → [B, 1, 512]
- Concat all inputs + action_noisy projection → sequence of length 148 (= 128 + 4 + 16)
- + positional embedding
- t_emb from diffusion_t (sinusoidal + 2-layer MLP)
- f_cond projection: 3 → 512 per point → [B, 128, 512]
- entropy projection: 1 → 512 → [B, 1, 512]
- Stack of 8 `ArtiResidualDiTBlock`s
- Final LayerNorm + AdaLN
- Linear 512 → 7
- Slice last 16 tokens → [B, 16, 7]

**Diffusion specifics**:
- v-prediction (NOT ε-prediction): predict `v = α_t·ε - σ_t·x_0` (Salimans & Ho 2022).
- Cosine schedule.
- DDIM sampler with 10 steps for inference.

**Reference**:
- DiT: https://github.com/facebookresearch/DiT
- RDT-1B: https://github.com/thu-ml/RoboticsDiffusionTransformer
- DP3: https://github.com/YanjieZe/3D-Diffusion-Policy
- v-prediction: https://arxiv.org/abs/2202.00512

**Total params**: ~50M (target). If you exceed 80M, reduce depth or dim.

**Acceptance test**:
- Forward pass: returns [B, 16, 7].
- Sample on a trained checkpoint: actor-arm trajectory in joint space, smooth (no jumps > 0.1 rad/step).
- Inference speed: ≥ 25 FPS on a single A800 with batch=1.

---

### Module 09: Multi-Handle Benchmark

**File**: `artiresidual/data/benchmark.py`

**Purpose**: Define the 12-20 multi-handle bimanual articulated tasks (Innovation 1).

**Tasks (initial 12)**:
1. `open_double_door_cabinet` — two-door cabinet, both revolute
2. `open_double_drawer` — two prismatic drawers
3. `open_fridge_with_freezer` — fridge door (revolute) + freezer door (revolute)
4. `open_microwave_with_tray` — door (revolute) + sliding tray (prismatic)
5. `lift_pot_by_two_handles` — pot with two side handles (bimanual rigid)
6. `open_suitcase_two_clasps` — two clasps + one main hinge
7. `open_oven_door_and_tray` — door (revolute) + tray (prismatic)
8. `open_double_door_microwave` — two-door microwave
9. `unscrew_bottle` — bimanual: one stabilizes, one rotates cap
10. `open_laptop` — lid revolute, stabilizer needed (slides on table)
11. `open_storage_box` — lid (revolute) + content access
12. `open_cabinet_with_drawer` — one cabinet door + one drawer

**API**:
```python
class MultiHandleBenchmark:
    def __init__(self, sim='robotwin2', config_path: str = ...): ...
    
    def list_tasks(self) -> List[str]: ...
    
    def make_env(self, task_name: str, seed: int = 0) -> Env:
        """Returns gym-style env compatible with RoboTwin's interface."""
    
    def evaluate(self, policy: Callable, task_name: str, n_trials: int = 100) -> Dict:
        """Run policy n_trials times. Returns success rate, failure modes, etc."""
```

**Reference**:
- RoboTwin 2.0 task framework: https://github.com/RoboTwin-Platform/RoboTwin/tree/main/envs
- ManiSkill 3 templates: https://github.com/haosulab/ManiSkill/tree/main/mani_skill/envs

**Implementation notes**:
- Use RoboTwin 2.0's task definition pattern as template.
- Each task is a Python class inheriting from RoboTwin's `BaseTask`.
- Domain randomization: inherit RoboTwin 2.0's 5-axis randomization.
- AgileX URDF: use RoboTwin's built-in NERO support.

**Acceptance test**:
- All 12 tasks load and run a random policy for 100 steps without crash.
- Success criterion verified by visual inspection on ≥ 5 successful trajectories per task.

---

### Module 10: Demo Generation

**File**: `artiresidual/data/demo_gen.py`

**Purpose**: Generate scripted expert demonstrations for training. Each demo includes stabilizer/actor role labels, full observation, action, wrench, articulation state ground truth.

**API**:
```python
def generate_demos(
    task_name: str,
    n_demos: int = 100,
    output_dir: str = ...,
    use_curobo: bool = True,
) -> None:
    """Writes n_demos trajectories to output_dir as zarr/hdf5."""
```

**Per-demo data schema**:
```python
{
    'pcd': [T, 1024, 6],          # T timesteps, RGB-D point cloud
    'q': [T, 14],                  # bimanual joint positions
    'q_vel': [T, 14],
    'wrench': [T, 12],
    'ee_pose': [T, 2, 7],
    'action': [T, 7],              # actor arm joint delta
    'stab_pose': [T, 7],           # stabilizer pose
    'stab_arm': str,               # 'left' or 'right'
    
    # Ground-truth articulation
    'omega_gt': [3],
    'p_gt': [3],
    'joint_type_gt': int,
    'theta_t_gt': [T],
    
    # For multi-handle:
    'parts': [{
        'omega': [3], 'p': [3], 'type': int,
        'assigned_arm': str,
        'role': str,                # 'stab' or 'act'
    }],
}
```

**Reference**:
- ArticuBot's gen_demo branch: https://github.com/yufeiwang63/articubot/tree/gen_demo
- RoboTwin 2.0's expert demo generation: https://robotwin-platform.github.io/doc/usage/control-robot.html
- cuRobo motion planning: https://github.com/NVlabs/curobo

**Implementation notes**:
- For each task, write a `scripted_policy(task_name)` function that:
  1. Reads URDF ground truth → (ω, p, type) for each part.
  2. Decides role assignment (stab vs act) using simple heuristics.
  3. Plans stabilizer reach + grasp with cuRobo.
  4. Plans actor reach + grasp + articulation-following trajectory using analytical flow.
  5. Executes in sim, records.
- Drop failed trajectories.

**Acceptance test**:
- For each of the 12 tasks: ≥ 80 of 100 attempted demos succeed.
- Verify recorded actions are smooth, no joint limit violations.

---

### Module 11: Training Perturbation

**File**: `artiresidual/data/perturb.py`

**Purpose**: Generate perturbed-initial-estimate variants of expert demos for refiner training. This is Innovation 2's training signal source.

**API**:
```python
def perturb_initial_estimate(
    demo: Dict,
    n_replays: int = 5,
    omega_perturb_range_deg: Tuple[float, float] = (5.0, 30.0),
    p_perturb_sigma_cm: float = 2.0,
    p_perturb_max_cm: float = 5.0,
    type_swap_prob: float = 0.20,
) -> List[Dict]:
    """
    For each demo, generate n_replays variants where the initial (ω, p, type)
    estimate is perturbed. The expert action trajectory is unchanged
    (this trains the refiner to recognize and correct its own initial error).
    """
```

**Perturbation distribution** (small + medium only):
- **ω**: rotate by angle drawn from `Uniform(5°, 30°)` about a random axis in the tangent space of S² at the true ω.
- **p**: 3D Gaussian with σ=2 cm, clipped at 5 cm.
- **joint_type**: with probability 0.20, swap revolute ↔ prismatic.

**Implementation notes**:
- The perturbed (ω, p, type) is the INITIAL state for the refiner; the demo proceeds as if this perturbed estimate were the prior estimator's output.
- Ground truth is unchanged in the recorded trajectory (used as training target).
- 12 tasks × 100 demos × 5 replays = 6000 perturbation episodes total.

**Acceptance test**:
- Perturbation distribution matches spec (verify with histogram).
- All perturbations satisfy magnitude constraints.

---

### Module 12: Failure Analysis

**File**: `artiresidual/evaluation/failure_analysis.py`

**Purpose**: Categorize failure modes when running ALL methods on the benchmark. Innovation 3.

**Failure categories**:
- (a) **Joint axis misestimation**: |estimated_ω − true_ω| > 30° at end of trajectory
- (b) **Inter-arm force conflict**: instantaneous wrench norm > threshold for > 1 second
- (c) **Handle slip**: gripper-object distance jumps > 3 cm in one step
- (d) **Coordination desync**: stabilizer and actor are out of phase

**API**:
```python
def categorize_failure(trajectory: Dict) -> Dict:
    """
    Returns:
        {
            'success': bool,
            'failure_type': str | None,
            'failure_step': int | None,
            'evidence': Dict,
        }
    """
```

**Implementation notes**:
- Apply post-hoc to every evaluation rollout.
- Each method gets a stacked bar chart: success% + breakdown of failure types.
- For ArtiResidual's reviewer defense: our self-correction addresses (a), wrench feedback addresses (b), comparable on (c) and (d).

**Acceptance test**:
- On a synthetic dataset with known failure modes injected, categorizer correctly identifies them with ≥ 90% accuracy.

---

## 4. Math Spec

### 4.1 Analytical flow (revolute)
For point x ∈ ℝ³, joint axis (ω, p), joint angle delta δθ:
```
f_ana(x) = (ω × (x − p)) · δθ
```

### 4.2 Analytical flow (prismatic)
```
f_ana(x) = ω · δθ
```

### 4.3 Hypothesis update rule (every N control steps)
```
ℓ_k = NeuralLogLikelihood(window, hypothesis_k)
w_k_new = w_k * exp(ℓ_k)
w_k_new = max(w_k_new, w_min)
w_k_new = w_k_new / sum(w_k_new)
```

### 4.4 Tangent-space residual application
For ω ∈ S² and tangent Δω ∈ ℝ³ (perpendicular to ω):
```
ω_new = ω · cos(‖Δω‖·η) + (Δω/‖Δω‖) · sin(‖Δω‖·η)
```
with `η = 0.5`, `‖Δω‖` clipped to `30° = 0.524 rad`.

For p ∈ ℝ³:
```
p_new = p + clip(η · Δp, max_norm=0.05)   # 5 cm
```

### 4.5 Belief-weighted flow
```
f_cond(x) = Σ_k w_k · f_ana_k(x)
```

### 4.6 Entropy
```
H(w) = −Σ_k w_k · log(w_k + ε)
```

### 4.7 v-prediction loss
For target `x_0` and noisy `x_t = α_t · x_0 + σ_t · ε`:
```
v_target = α_t · ε − σ_t · x_0
L_diffusion = ‖v_pred − v_target‖²
```

### 4.8 Total loss (joint fine-tune)
```
L = L_diffusion + λ_refiner · (L_NLL + L_mu_residual + λ_H · (−H(w))) + λ_belief · L_belief_consistency
```
with `λ_refiner = 0.1`, `λ_H = 0.01`, `λ_belief = 0.05`.

### 4.9 L_NLL (hypothesis selection)
Given ground-truth hypothesis index k*:
```
L_NLL = −log(w_k*)
```

### 4.10 L_mu_residual
```
L_mu_residual = Σ_k w_k · (1 − cos(ω_k, ω_true)) + λ_p · ‖p_k − p_true‖²
```

---

## 5. Configuration Schema (Hydra)

Use Hydra for config management. Example `configs/base.yaml`:
```yaml
defaults:
  - refiner: imm_k3
  - policy: dit_50m
  - task: multi_handle
  - _self_

global:
  seed: 42
  device: cuda
  dtype: float32

paths:
  data_root: /workspace/data
  ckpt_root: /workspace/ckpts
  output_root: /workspace/outputs

logging:
  use_wandb: true
  project: artiresidual
  
training:
  batch_size: 32
  num_workers: 8
  lr: 1e-4
  weight_decay: 1e-4
  grad_clip: 1.0
  warmup_steps: 1000
  total_steps: 200000
```

---

## 6. Implementation Order (CRITICAL)

Build modules in this order. **DO NOT skip ahead — dependencies are real**.

1. **Module 05** (analytical_flow) — pure math, easy win
2. **Module 06** (state_estimator) — geometry, no ML
3. **Module 09** (benchmark) — define tasks first
4. **Module 10** (demo_gen) — generate data for everything else
5. **Module 11** (perturb) — augment demos
6. **Module 01** (prior_estimator) — train on PartNet-Mobility GT
7. **Module 02** (flow_predictor) — train with sim GT
8. **Module 03** (selector) — train with demo role labels
9. **CHECKPOINT**: Now you can roll out a *baseline* (no refiner/policy yet) — open-loop scripted execution. Verify benchmark works.
10. **Module 04** (IMM refiner) ★ — Stage 1: refiner only, policy frozen
11. **Module 07** (DiT block) — implement and unit-test
12. **Module 08** (DiT policy) — Stage 2: policy only, refiner frozen
13. **Joint fine-tune** (Stage 3): unfreeze everything, low refiner-loss weight
14. **Module 12** (failure_analysis) — apply to all evaluation runs
15. **Baselines**: DP3, ACT, fine-tuned RDT-1B, ArticuBot-bimanual, Buchanan replication

---

## 7. Coding Standards

**For Claude Code, when generating any module**:

1. **Type hints everywhere**. Use PEP 604 syntax (`X | None`).
2. **Docstrings** in Google style for every public function/class.
3. **Tensor shape comments** at every line where shape changes:
   ```python
   x = self.encoder(pcd)  # [B, N, D]
   x = x.mean(dim=1)      # [B, D]
   ```
4. **Assert shapes** at function boundaries when in doubt.
5. **Hydra configs** for all hyperparameters. No magic numbers in code.
6. **Wandb logging** in training scripts. Log losses, gradient norms, learning rate, sample videos every N steps.
7. **Deterministic seeds** in all training entry points.
8. **`torch.compile`** on policy and refiner once stable.
9. **AMP / bf16** for training (saves ~40% memory, 2× speed on A800).
10. **DDP** for multi-GPU.

---

## 8. Critical Reference Repositories (Bookmark These)

| Module | Primary reference |
|---|---|
| Module 01 (Prior Estimator) | https://github.com/r-pad/flowbot3d |
| Module 02 (Flow Predictor) | https://github.com/r-pad/flowbot3d + https://github.com/YanjieZe/3D-Diffusion-Policy |
| Module 03 (Selector) | https://github.com/VoxAct-B/voxact-b |
| Module 04 (IMM Refiner) | Implement from scratch; reference https://github.com/akloss/differentiable_filters |
| Module 05 (Analytical Flow) | Student's prior PAct code (`affordance_utils.py`) |
| Module 06 (State Estimator) | Geometry textbook |
| Module 07 (DiT Block) | https://github.com/facebookresearch/DiT |
| Module 08 (DiT Policy) | https://github.com/thu-ml/RoboticsDiffusionTransformer + https://github.com/facebookresearch/DiT |
| Module 09 (Benchmark) | https://github.com/RoboTwin-Platform/RoboTwin |
| Module 10 (Demo Gen) | https://github.com/yufeiwang63/articubot/tree/gen_demo |
| Module 11 (Perturb) | From scratch (~150 lines) |
| Module 12 (Failure Analysis) | From scratch |

Baselines:
- RDT-1B: https://github.com/thu-ml/RoboticsDiffusionTransformer
- π0 (if open): https://github.com/Physical-Intelligence/openpi
- DP3: https://github.com/YanjieZe/3D-Diffusion-Policy
- ACT: https://github.com/tonyzhaozh/act
- ArticuBot: https://github.com/yufeiwang63/articubot

---

## 9. Definition of Done (per module)

A module is "done" when ALL true:
1. Code compiles, all imports resolve, no warnings.
2. Type hints complete.
3. Acceptance test passes.
4. Unit tests cover edge cases (empty input, batch size 1, NaN handling).
5. Wandb logging in place (training modules).
6. Hydra config exists for all hyperparameters.
7. Docstring includes example usage.
8. README in the module's directory explains purpose in 3 sentences.

---

## 10. Instructions to Claude Code

**When the user asks you to implement a module from this spec**:

1. Re-read the relevant section above.
2. Check the listed reference repository for code patterns.
3. Confirm the API contract (function signatures, tensor shapes) before writing.
4. Implement with type hints, docstrings, and shape comments.
5. Add unit tests in `tests/test_module_name.py`.
6. Verify the acceptance test passes (or describe what would need to be true if data isn't available yet).
7. Update the relevant Hydra config.
8. If you find ambiguity in this spec, STOP and ask the user before assuming.

**When you find a design decision that conflicts with this spec**:
- Default: follow this spec.
- Exception: if conflict is a bug in this spec (e.g., shape inconsistency), report and propose fix.

**When you produce code**:
- Match existing repo style (after first module is in place).
- Don't refactor existing code unless asked.
- Don't add dependencies without flagging it.

---

## End of Spec
