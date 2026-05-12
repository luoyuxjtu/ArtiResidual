"""pytest configuration shared by all tests.

In the codespace we don't have a GPU or a simulator. Tests that need them
are tagged with `@pytest.mark.gpu` / `@pytest.mark.sim` and are skipped
automatically here. On the server you can opt in with `pytest -m gpu`
or `pytest -m 'not gpu and not sim'` for a CPU-only sanity pass.
"""
from __future__ import annotations

import os
import shutil

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    has_gpu = _has_gpu()
    has_sim = _has_sim()

    skip_gpu = pytest.mark.skip(reason="no CUDA GPU available (codespace)")
    skip_sim = pytest.mark.skip(reason="no simulator available (codespace)")

    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)
        if "sim" in item.keywords and not has_sim:
            item.add_marker(skip_sim)


def _has_gpu() -> bool:
    if os.environ.get("ARTIRESIDUAL_FORCE_NO_GPU"):
        return False
    try:
        import torch  # noqa: PLC0415

        return torch.cuda.is_available()
    except Exception:
        return False


def _has_sim() -> bool:
    # Heuristic: RoboTwin 2.0 ships its CLI entrypoint as `robotwin`.
    return shutil.which("robotwin") is not None
