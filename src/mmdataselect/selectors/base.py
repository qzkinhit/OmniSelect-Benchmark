"""Selector interface + a plain importance Top-K reference selector."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

from ..datatypes import UnifiedRecord


class Selector(ABC):
    name: str = "selector"

    @abstractmethod
    def select(self, records: Sequence[UnifiedRecord], importance, k: int, **kwargs) -> List[int]:
        """Return the indices of the kept records (len <= k)."""
        raise NotImplementedError


class TopKSelector(Selector):
    """Keep the k highest-importance records (no diversity term). Reference baseline."""

    name = "topk"

    def select(self, records, importance, k, **kwargs):
        imp = np.asarray(importance, dtype=float)
        k = max(0, min(int(k), len(records)))
        if k == 0:
            return []
        return [int(i) for i in np.argsort(-imp)[:k]]
