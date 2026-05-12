"""Data subpackage — Modules 09 (benchmark), 10 (demo gen), 11 (perturb).

Innovation 1 lives here: the 12-task multi-handle bimanual benchmark on top of
RoboTwin 2.0. Demos are scripted with role labels (stabilizer vs. actor) and
perturbed to provide the refiner's training signal.

See ``artiresidual/data/README.md`` for the per-task selection rationale.
"""
from artiresidual.data.benchmark import (
    MAX_STEPS_DEFAULT,
    MockArticulationEnv,
    MultiHandleBenchmark,
    TASKS_V1,
)

__all__ = [
    "MAX_STEPS_DEFAULT",
    "MockArticulationEnv",
    "MultiHandleBenchmark",
    "TASKS_V1",
]
