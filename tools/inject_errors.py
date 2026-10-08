"""Thin wrapper over the corruption mechanisms of tracks/common/injection.py.

``inject(family, ...)`` dispatches to the canonical mechanism of a data family: vision labels,
forecasting windows, or process and tabular rows. It returns what the mechanism returns (corrupted
arrays and per-record tags). The mechanisms use default_rng(seed + 7) as the canonical runners did.
"""
from __future__ import annotations

from typing import Any

from tracks.common import injection


def inject(family: str, *args: Any, **kwargs: Any):
    """Apply the canonical corruption of ``family`` (vision, timeseries, process, tabular)."""
    if family == "vision":
        return injection.inject_vision_labels(*args, **kwargs)
    if family == "timeseries":
        return injection.inject_windows(*args, **kwargs)
    if family in ("process", "tabular"):
        return injection.inject_rows(*args, **kwargs)
    raise KeyError(f"unknown family {family!r}")
