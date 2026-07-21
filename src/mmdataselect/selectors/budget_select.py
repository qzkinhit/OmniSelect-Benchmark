"""Budget-aware selection: importance(influence) x diversity(coverage - redundancy).

Greedy facility-location: at each step keep the record maximizing

    gain_i = importance_i + lam * (1 - max_sim(i, already_selected))

so high-influence records are preferred while penalizing redundancy against the
already-selected set. Operates only on UnifiedRecord + a score vector, so the same
selector is reused across vision-language / math / code. A cheap stochastic variant
(:func:`gumbel_topk`) is provided for ablations.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..datatypes import UnifiedRecord
from ..signals.base import minmax
from ..signals.redundancy import hashed_features
from .base import Selector


def gumbel_topk(importance, k: int, rng: np.random.Generator) -> List[int]:
    """Perturb log-importance with Gumbel noise and take the top-k (stochastic)."""
    imp = np.asarray(importance, dtype=float)
    k = max(0, min(int(k), imp.shape[0]))
    if k == 0:
        return []
    keys = np.log(np.clip(imp, 1e-9, None)) + rng.gumbel(size=imp.shape)
    return [int(i) for i in np.argsort(-keys)[:k]]


class BudgetSelector(Selector):
    name = "budget_select"

    def __init__(self, lam: float = 0.5, method: str = "greedy", feat_dim: int = 256, seed: int = 0):
        self.lam = float(lam)
        self.method = method
        self.feat_dim = int(feat_dim)
        self.seed = int(seed)

    def select(
        self,
        records: Sequence[UnifiedRecord],
        importance,
        k: int,
        *,
        features: Optional[np.ndarray] = None,
        **kwargs,
    ) -> List[int]:
        n = len(records)
        k = max(0, min(int(k), n))
        if k == 0:
            return []
        imp = minmax(importance)
        if self.method == "gumbel":
            return gumbel_topk(imp, k, np.random.default_rng(self.seed))

        feats = features if features is not None else hashed_features(records, dim=self.feat_dim)
        selected: List[int] = []
        chosen = np.zeros(n, dtype=bool)
        max_sim = np.zeros(n, dtype=float)
        for _ in range(k):
            gain = imp + self.lam * (1.0 - max_sim)
            gain[chosen] = -np.inf
            j = int(np.argmax(gain))
            selected.append(j)
            chosen[j] = True
            sims = feats @ feats[j]  # cosine: rows are L2-normalized
            max_sim = np.maximum(max_sim, sims)
        return selected
