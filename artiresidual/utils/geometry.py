"""Geometry primitives used across modules.

Quaternion / rotation matrix conversions, tangent-space exp/log maps for S²,
SE(3) interpolation, point-cloud transformations. Pure functions, no learning.

Where the IMM refiner (Module 04) needs the tangent-space exp map for omega
updates (spec §4.4), prefer adding it here rather than inline in the filter.
"""
from __future__ import annotations

__all__: list[str] = []
