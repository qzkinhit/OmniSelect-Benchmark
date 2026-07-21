"""Signal interface + the shared [0,1] normalization that makes signals comparable."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from ..datatypes import UnifiedRecord


def minmax(x) -> np.ndarray:
    """Min-max normalize to [0,1]; constant input maps to all-0.5 (no information)."""
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
        """Return a length-N array in [0,1]; higher = more valuable to keep."""
        raise NotImplementedError
