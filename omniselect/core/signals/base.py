"""Signal interface and ``minmax``, which maps a vector to [0, 1] (a constant vector maps to 0.5).

A Signal maps a list of UnifiedRecords to one float per record, higher meaning more valuable to
keep. Non-text tracks compute their channels directly as arrays (core/signals/knn.py,
influence.probe_log_likelihood) and use only ``minmax`` from this module. The controller min-max
normalizes every channel before fusion.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from omniselect.core.datatypes import UnifiedRecord


def minmax(x) -> np.ndarray:
    """Min-max normalization to [0, 1]. A constant input maps to 0.5 everywhere."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


class Signal(ABC):
    """A modality-agnostic per-record utility signal."""

    name: str = "signal"

    @abstractmethod
    def score(self, records: Sequence[UnifiedRecord]) -> np.ndarray:
        """Length-N array in [0, 1], higher meaning more valuable to keep."""
        raise NotImplementedError
