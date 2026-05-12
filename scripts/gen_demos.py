"""CLI wrapper to generate scripted expert demos for every benchmark task.

RUN ON THE UBUNTU SERVER ONLY (needs RoboTwin 2.0 + cuRobo + a sim renderer).

Intended invocation:
    python scripts/gen_demos.py --task open_double_door_cabinet --n 100

See artiresidual_tech_spec.md §3 Module 10 for the data schema.
"""
from __future__ import annotations

__all__: list[str] = []
