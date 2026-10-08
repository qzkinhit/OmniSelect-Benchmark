"""Selector interface and TopKSelector, which keeps the k records with the highest importance.

A selector maps records, an importance vector and a budget k to at most k record indices.
TopKSelector sorts the importance in descending order and has no diversity term. BudgetSelector
in budget_select.py adds the diversity term. Both return plain lists of integers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

from omniselect.core.datatypes import UnifiedRecord


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
        return [int(i) for i in np.argsort(-imp, kind="stable")[:k]]
