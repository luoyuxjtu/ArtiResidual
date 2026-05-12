"""Module 09 — Multi-Handle Bimanual Benchmark.

See artiresidual_tech_spec.md §3 Module 09 for the authoritative API and the
canonical 12-task list. This module is *Innovation 1* of the paper: a
benchmark targeted at the **coordination** problem in multi-handle bimanual
articulated manipulation — most prior benchmarks focus on single-handle or
single-arm cases. See `artiresidual/data/README.md` for the per-task
selection rationale.

Skeleton status (RoboTwin 2.0 / ManiSkill 3 NOT yet installed):

    list_tasks()    DONE  — returns the canonical 12 task names in spec order.
    make_env()      MOCK  — always returns ``MockArticulationEnv``, regardless
                            of ``self.sim``. The mock supports ``reset()`` and
                            ``step()`` and returns plausibly-shaped tensors,
                            so downstream modules (policy, refiner, eval
                            harness) can be built against a stable interface
                            today, without waiting for the simulator install.
    evaluate()      MOCK  — runs a generic rollout loop that will keep
                            working once the real env replaces the mock; the
                            success metric is stubbed to 0 because the mock
                            has no physics.

What's still ``TODO`` (and why) is documented at the call sites.

Reference repos for the eventual real binding:
    RoboTwin 2.0   https://github.com/RoboTwin-Platform/RoboTwin/tree/main/envs
    ManiSkill 3    https://github.com/haosulab/ManiSkill/tree/main/mani_skill/envs
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

__all__ = [
    "MAX_STEPS_DEFAULT",
    "MockArticulationEnv",
    "MultiHandleBenchmark",
    "TASKS_V1",
]


# ---------------------------------------------------------------------------
# Canonical task list (spec §3 Module 09, in spec order).
# ---------------------------------------------------------------------------

TASKS_V1: tuple[str, ...] = (
    "open_double_door_cabinet",
    "open_double_drawer",
    "open_fridge_with_freezer",
    "open_microwave_with_tray",
    "lift_pot_by_two_handles",
    "open_suitcase_two_clasps",
    "open_oven_door_and_tray",
    "open_double_door_microwave",
    "unscrew_bottle",
    "open_laptop",
    "open_storage_box",
    "open_cabinet_with_drawer",
)


# ---------------------------------------------------------------------------
# Shape constants (per spec §2 and §3 Module 09 eval.max_steps).
# ---------------------------------------------------------------------------

PCD_N_RAW = 1024
Q_DIM = 14
WRENCH_DIM = 12
EE_POSE_SHAPE: tuple[int, int] = (2, 7)
ACTION_DIM = 7
MAX_STEPS_DEFAULT = 400


# ---------------------------------------------------------------------------
# Mock env.
# ---------------------------------------------------------------------------


class MockArticulationEnv:
    """Mock RoboTwin-shaped env supporting only ``reset()`` and ``step()``.

    Designed to unblock downstream development: every method returns tensors of
    the shapes the real env will eventually produce, so e.g.
    ``policy.forward(obs)`` and ``refiner.step(obs)`` can be unit-tested
    without launching a real physics simulator.

    Explicit non-features (intentional — replace when the real env lands):
        * No physics: ``step()`` validates the action shape, advances a
          counter, and returns fresh random pcd. The action is otherwise
          ignored.
        * No success: ``terminated`` is always ``False`` — success requires
          real articulation dynamics. ``truncated`` flips to ``True`` once
          ``max_steps`` elapses.
        * Stub GT articulation: ``info`` carries a constant ``(omega, p,
          joint_type, theta_t)`` so refiner / state-estimator unit tests can
          read it, but the values are not derived from any URDF.
        * No rendering, no ``observation_space`` / ``action_space``
          properties.

    Obs keys follow spec §2:
        ``pcd``       [N_raw, 3]    world-frame point cloud (random)
        ``q``         [14]          bimanual joint positions (zeros)
        ``wrench``    [12]          6+6 wrenches (zeros)
        ``ee_pose``   [2, 7]        per-arm pos + scalar-first quat (frozen)

    Info keys (subject to growth once real env is in):
        ``task_name``, ``step``, ``is_mock``,
        ``omega_gt``, ``p_gt``, ``joint_type_gt``, ``theta_t_gt``.
    """

    metadata: dict[str, Any] = {"is_mock": True}

    def __init__(
        self,
        task_name: str,
        seed: int = 0,
        max_steps: int = MAX_STEPS_DEFAULT,
    ) -> None:
        if task_name not in TASKS_V1:
            raise ValueError(
                f"unknown task {task_name!r}; "
                f"see MultiHandleBenchmark.list_tasks() for valid names"
            )
        self.task_name: str = task_name
        self.max_steps: int = max_steps
        self._seed: int = seed
        self._step_count: int = 0
        self._rng: torch.Generator = torch.Generator().manual_seed(seed)

    # -- Gymnasium-style API: reset returns (obs, info). ----------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """Reset to t=0. Returns ``(obs, info)`` (Gymnasium 5-tuple convention).

        Args:
            seed: if given, re-seeds the internal RNG and replaces the seed
                cached in the constructor.
            options: accepted for Gymnasium compatibility; ignored by the
                mock.
        """
        del options  # accepted for API compat; the mock has no options to set
        if seed is not None:
            self._seed = seed
            self._rng = torch.Generator().manual_seed(seed)
        self._step_count = 0
        return self._make_obs(), self._make_info()

    # -- Gymnasium-style API: step returns 5-tuple. --------------------------
    def step(
        self, action: Tensor | list | tuple
    ) -> tuple[dict, float, bool, bool, dict]:
        """Advance one control step.

        Args:
            action: [7] actor-arm joint-delta. Stabilizer pose is held by the
                env (real env: from Module 03 selector; mock: ignored).

        Returns:
            ``(obs, reward, terminated, truncated, info)`` following the
            Gymnasium 5-tuple convention.

        Raises:
            ValueError: on wrong action shape.
        """
        action_t = torch.as_tensor(action, dtype=torch.float32)
        if action_t.shape != (ACTION_DIM,):
            raise ValueError(
                f"action must have shape ({ACTION_DIM},); got "
                f"{tuple(action_t.shape)}"
            )

        self._step_count += 1
        reward = 0.0
        terminated = False  # mock never reports success — no physics
        truncated = self._step_count >= self.max_steps
        return self._make_obs(), reward, terminated, truncated, self._make_info()

    # -- internals -----------------------------------------------------------
    def _make_obs(self) -> dict:
        """Plausible obs at the current step. Deterministic given the seed."""
        return {
            "pcd": torch.randn(PCD_N_RAW, 3, generator=self._rng),  # [N, 3]
            "q": torch.zeros(Q_DIM),                                  # [14]
            "wrench": torch.zeros(WRENCH_DIM),                        # [12]
            "ee_pose": torch.tensor(                                  # [2, 7]
                [
                    [0.30,  0.20, 0.50, 1.0, 0.0, 0.0, 0.0],
                    [0.30, -0.20, 0.50, 1.0, 0.0, 0.0, 0.0],
                ]
            ),
        }

    def _make_info(self) -> dict:
        # TODO: replace stub GT with values parsed from the real URDF once
        # RoboTwin 2.0 is wired in. The IMM refiner unit tests already read
        # these keys, so we keep them populated even in mock mode.
        return {
            "task_name": self.task_name,
            "step": self._step_count,
            "is_mock": True,
            "omega_gt": torch.tensor([0.0, 0.0, 1.0]),
            "p_gt": torch.tensor([0.0, 0.0, 0.0]),
            "joint_type_gt": 0,    # revolute placeholder
            "theta_t_gt": 0.0,
        }


# ---------------------------------------------------------------------------
# Benchmark facade.
# ---------------------------------------------------------------------------


class MultiHandleBenchmark:
    """Multi-handle bimanual articulated-object benchmark (spec §3 Module 09).

    Innovation 1 of the paper: 12 tasks spanning revolute×revolute,
    revolute×prismatic, prismatic×prismatic, bimanual rigid grasp, and
    continuous-rotation coordination modes. See
    ``artiresidual/data/README.md`` for the per-task rationale.

    Backends:
        ``robotwin2`` (default)  — native bimanual + AgileX NERO support.
        ``maniskill3``           — secondary cross-sim ablation.
        Neither is hooked up yet; both fall through to ``MockArticulationEnv``.

    Usage:
        bench = MultiHandleBenchmark()
        env = bench.make_env("open_double_door_cabinet", seed=0)
        obs, info = env.reset()
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    """

    SUPPORTED_BACKENDS: tuple[str, ...] = ("robotwin2", "maniskill3")

    def __init__(
        self,
        sim: str = "robotwin2",
        config_path: str | None = None,
    ) -> None:
        """
        Args:
            sim: backend name. One of :attr:`SUPPORTED_BACKENDS`.
            config_path: optional path to a Hydra task config. Reserved for the
                real binding (RoboTwin 2.0 reads task definitions from disk);
                ignored by the mock.

        Raises:
            ValueError: on unknown ``sim``.
        """
        if sim not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"unsupported sim {sim!r}; "
                f"supported backends: {list(self.SUPPORTED_BACKENDS)}"
            )
        self.sim: str = sim
        self.config_path: str | None = config_path
        # TODO: when sim == "robotwin2", load the RoboTwin task registry from
        # config_path (or env-var). Until the installer is in, leave None.
        self._task_registry: dict | None = None

    def list_tasks(self) -> list[str]:
        """Return the canonical task names in spec order. See spec §3 Module 09."""
        return list(TASKS_V1)

    def make_env(
        self,
        task_name: str,
        seed: int = 0,
        **kwargs: Any,
    ) -> MockArticulationEnv:
        """Construct an env for ``task_name``.

        Args:
            task_name: one of :data:`TASKS_V1`.
            seed: passed to the env's RNG.
            **kwargs: forwarded to the env constructor. Currently only
                ``max_steps`` is honored by the mock.

        Returns:
            A mock env today; a real RoboTwin / ManiSkill env once the
            backends land. The return type is duck-compatible with
            ``MockArticulationEnv`` (reset / step), which is all the rest of
            the codebase requires.

        Raises:
            ValueError: if ``task_name`` is not in :data:`TASKS_V1`.
        """
        if task_name not in TASKS_V1:
            raise ValueError(
                f"unknown task {task_name!r}; "
                f"see MultiHandleBenchmark.list_tasks() for valid names"
            )
        # TODO: dispatch on ``self.sim``:
        #   if self.sim == "robotwin2":
        #       return _make_robotwin_env(task_name, seed=seed, **kwargs)
        #   if self.sim == "maniskill3":
        #       return _make_maniskill_env(task_name, seed=seed, **kwargs)
        # Each branch will import a task-specific BaseTask subclass; that
        # registry doesn't exist yet (see TODO list in benchmark.py header).
        return MockArticulationEnv(task_name, seed=seed, **kwargs)

    def evaluate(
        self,
        policy: Callable[[dict], Tensor],
        task_name: str,
        n_trials: int = 100,
        max_steps: int = MAX_STEPS_DEFAULT,
        seed_base: int = 0,
    ) -> dict[str, Any]:
        """Roll out ``policy`` over ``n_trials`` independent seeds and aggregate.

        The loop is backend-agnostic — it works against any env exposing
        ``reset()`` / ``step()`` in the Gymnasium 5-tuple form. So this method
        will continue to work unchanged once the mock is swapped for a real
        env.

        Args:
            policy: callable mapping a single-step obs dict to an action tensor
                of shape ``(ACTION_DIM,)``.
            task_name: one of :data:`TASKS_V1`.
            n_trials: number of independent episodes.
            max_steps: episode length cap per trial.
            seed_base: ``env`` seed for trial ``i`` is ``seed_base + i``.

        Returns:
            Dict with success rate, mean rollout length, and a (stubbed)
            failure-mode breakdown. ``success_rate`` is always 0 under the
            mock (no physics, no success criterion).

        Raises:
            ValueError: if ``task_name`` is unknown.
        """
        if task_name not in TASKS_V1:
            raise ValueError(f"unknown task {task_name!r}")

        n_success = 0
        rollout_lengths: list[int] = []
        for trial_idx in range(n_trials):
            env = self.make_env(
                task_name, seed=seed_base + trial_idx, max_steps=max_steps
            )
            obs, _info = env.reset()
            done = False
            steps_taken = 0
            while not done and steps_taken < max_steps:
                action = policy(obs)
                obs, _reward, terminated, truncated, _info = env.step(action)
                steps_taken += 1
                if terminated:
                    n_success += 1
                done = terminated or truncated
            rollout_lengths.append(steps_taken)

        # TODO: once Module 12 (failure_analysis) lands, pipe each trajectory
        # through ``failure_analysis.categorize_failure`` and aggregate the
        # breakdown here instead of the stub bucket.
        return {
            "task_name": task_name,
            "n_trials": n_trials,
            "n_success": n_success,
            "success_rate": n_success / n_trials if n_trials else 0.0,
            "mean_rollout_length": (
                sum(rollout_lengths) / len(rollout_lengths)
                if rollout_lengths
                else 0.0
            ),
            "failure_modes": {"unknown": n_trials - n_success},
            "backend": self.sim,
            "is_mock": True,
        }
