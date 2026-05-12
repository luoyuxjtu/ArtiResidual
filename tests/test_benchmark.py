"""Unit tests for artiresidual.data.benchmark (Module 09 skeleton).

The real RoboTwin 2.0 binding is not in place yet — see the TODO list at the
top of `artiresidual/data/benchmark.py`. These tests target the spec §3
Module 09 *acceptance test*:

    "All 12 tasks load and run a random policy for 100 steps without crash."

Plus the contract of the temporary ``MockArticulationEnv`` that lets
downstream modules be developed before the simulator is installed.

All tests are CPU-only and rely only on torch.
"""
from __future__ import annotations

import pytest
import torch

from artiresidual.data.benchmark import (
    ACTION_DIM,
    EE_POSE_SHAPE,
    MAX_STEPS_DEFAULT,
    PCD_N_RAW,
    Q_DIM,
    WRENCH_DIM,
    MockArticulationEnv,
    MultiHandleBenchmark,
    TASKS_V1,
)


# ---------------------------------------------------------------------------
# 1. Canonical task list.
# ---------------------------------------------------------------------------


def test_list_tasks_returns_exactly_the_canonical_12() -> None:
    """Order, count, and uniqueness of the canonical task list."""
    bench = MultiHandleBenchmark()
    tasks = bench.list_tasks()
    assert tasks == list(TASKS_V1)
    assert len(tasks) == 12
    assert len(set(tasks)) == 12

    # Spot-check a few specific entries by name to catch typos / reorderings.
    assert tasks[0] == "open_double_door_cabinet"
    assert tasks[8] == "unscrew_bottle"
    assert tasks[-1] == "open_cabinet_with_drawer"


# ---------------------------------------------------------------------------
# 2. make_env returns a usable mock for every task.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_name", TASKS_V1)
def test_make_env_constructs_a_mock_for_each_task(task_name: str) -> None:
    bench = MultiHandleBenchmark()
    env = bench.make_env(task_name, seed=0)
    assert isinstance(env, MockArticulationEnv)
    assert env.task_name == task_name


# ---------------------------------------------------------------------------
# 3. Mock env reset() obs / info shapes match spec §2.
# ---------------------------------------------------------------------------


def test_mock_env_reset_returns_spec_shaped_obs_and_info() -> None:
    bench = MultiHandleBenchmark()
    env = bench.make_env("open_double_door_cabinet", seed=42)
    obs, info = env.reset()

    assert obs["pcd"].shape == (PCD_N_RAW, 3)
    assert obs["q"].shape == (Q_DIM,)
    assert obs["wrench"].shape == (WRENCH_DIM,)
    assert obs["ee_pose"].shape == EE_POSE_SHAPE

    assert info["task_name"] == "open_double_door_cabinet"
    assert info["step"] == 0
    assert info["is_mock"] is True
    # Stub GT articulation keys must be present — refiner tests rely on them.
    assert "omega_gt" in info and "p_gt" in info
    assert "joint_type_gt" in info and "theta_t_gt" in info


# ---------------------------------------------------------------------------
# 4. Mock env step() returns the Gymnasium 5-tuple.
# ---------------------------------------------------------------------------


def test_mock_env_step_returns_gymnasium_5tuple_with_correct_shapes() -> None:
    bench = MultiHandleBenchmark()
    env = bench.make_env("open_double_drawer", seed=0)
    env.reset()

    action = torch.zeros(ACTION_DIM)
    result = env.step(action)
    assert isinstance(result, tuple) and len(result) == 5
    obs, reward, terminated, truncated, info = result

    assert obs["pcd"].shape == (PCD_N_RAW, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and terminated is False  # no physics
    assert isinstance(truncated, bool)
    assert info["step"] == 1


def test_mock_env_step_rejects_wrong_action_shape() -> None:
    bench = MultiHandleBenchmark()
    env = bench.make_env("open_microwave_with_tray", seed=0)
    env.reset()
    with pytest.raises(ValueError, match="action must have shape"):
        env.step(torch.zeros(ACTION_DIM + 1))


# ---------------------------------------------------------------------------
# 5. truncated flips at max_steps.
# ---------------------------------------------------------------------------


def test_mock_env_truncates_at_max_steps() -> None:
    env = MockArticulationEnv("open_double_door_cabinet", seed=0, max_steps=5)
    env.reset()
    action = torch.zeros(ACTION_DIM)
    for step_i in range(4):
        _obs, _r, _term, truncated, _info = env.step(action)
        assert truncated is False, f"truncated too early at step {step_i + 1}"
    _obs, _r, _term, truncated, _info = env.step(action)
    assert truncated is True


# ---------------------------------------------------------------------------
# 6. Seed determinism.
# ---------------------------------------------------------------------------


def test_same_seed_gives_identical_obs() -> None:
    """Reproducibility: same seed → identical pcd. Different seed → different pcd."""
    bench = MultiHandleBenchmark()
    env_a = bench.make_env("open_fridge_with_freezer", seed=7)
    env_b = bench.make_env("open_fridge_with_freezer", seed=7)
    env_c = bench.make_env("open_fridge_with_freezer", seed=8)

    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()
    obs_c, _ = env_c.reset()

    assert torch.equal(obs_a["pcd"], obs_b["pcd"])
    assert not torch.equal(obs_a["pcd"], obs_c["pcd"])


# ---------------------------------------------------------------------------
# 7. Spec §3 Module 09 acceptance: 12 tasks × 100 steps × random policy.
# ---------------------------------------------------------------------------


def test_all_12_tasks_run_random_policy_for_100_steps_without_crash() -> None:
    """Spec acceptance test (paraphrased): all 12 tasks load and run a random
    policy for 100 steps without crash."""
    bench = MultiHandleBenchmark()
    torch.manual_seed(0)

    def random_policy(obs: dict) -> torch.Tensor:
        del obs
        return torch.randn(ACTION_DIM) * 0.01

    for task_name in bench.list_tasks():
        env = bench.make_env(task_name, seed=0, max_steps=200)
        obs, _info = env.reset()
        for _step in range(100):
            action = random_policy(obs)
            obs, _reward, terminated, truncated, _info = env.step(action)
            assert not terminated  # mock never reports success
            assert obs["pcd"].shape == (PCD_N_RAW, 3)
        # Successful run of 100 steps reached this point — no crash.


# ---------------------------------------------------------------------------
# 8. evaluate() returns the expected dict layout.
# ---------------------------------------------------------------------------


def test_evaluate_returns_well_formed_dict() -> None:
    bench = MultiHandleBenchmark()

    def zero_policy(obs: dict) -> torch.Tensor:
        del obs
        return torch.zeros(ACTION_DIM)

    result = bench.evaluate(
        zero_policy, "open_laptop", n_trials=3, max_steps=10
    )
    assert result["task_name"] == "open_laptop"
    assert result["n_trials"] == 3
    # Under the mock, success_rate must be exactly 0 (no physics).
    assert result["n_success"] == 0
    assert result["success_rate"] == 0.0
    assert result["mean_rollout_length"] == 10.0  # truncated at max_steps
    assert result["is_mock"] is True
    assert result["backend"] == "robotwin2"
    assert "failure_modes" in result


# ---------------------------------------------------------------------------
# 9. Error cases on the facade.
# ---------------------------------------------------------------------------


def test_unknown_task_name_raises_in_make_env() -> None:
    bench = MultiHandleBenchmark()
    with pytest.raises(ValueError, match="unknown task"):
        bench.make_env("not_a_real_task")


def test_unknown_task_name_raises_in_evaluate() -> None:
    bench = MultiHandleBenchmark()
    with pytest.raises(ValueError, match="unknown task"):
        bench.evaluate(lambda obs: torch.zeros(ACTION_DIM), "not_a_real_task")


def test_unsupported_backend_raises() -> None:
    with pytest.raises(ValueError, match="unsupported sim"):
        MultiHandleBenchmark(sim="pybullet")


def test_supported_backends_construct_without_error() -> None:
    """Both 'robotwin2' and 'maniskill3' should accept; they currently fall
    through to the mock since neither binding is installed."""
    bench_rw = MultiHandleBenchmark(sim="robotwin2")
    bench_ms = MultiHandleBenchmark(sim="maniskill3")
    assert bench_rw.sim == "robotwin2"
    assert bench_ms.sim == "maniskill3"
    # Both should produce a usable env from the mock fallback.
    for bench in (bench_rw, bench_ms):
        env = bench.make_env("open_double_door_cabinet", seed=0)
        obs, info = env.reset()
        assert obs["pcd"].shape == (PCD_N_RAW, 3)
        assert info["is_mock"] is True
