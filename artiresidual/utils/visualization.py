"""Visualization helpers — flow field overlays, hypothesis weight plots, etc.

Used for wandb sample logging during training and for paper figures. Heavy
deps (matplotlib, open3d) are imported lazily inside functions so importing
this module is cheap.
"""
from __future__ import annotations

__all__: list[str] = []
