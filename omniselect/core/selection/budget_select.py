"""BudgetSelector: greedy selection of k records by importance plus a diversity bonus.

At each step the record maximizing minmax(importance)_i + lam * (1 - max_sim(i, selected)) is
added, where max_sim is the largest dot product of its (L2-normalized) feature row with a selected
row. With lam = 0 this is top-k by importance. A ``similarity`` object (core/selection/similarity.py)
replaces the feature dot product (gradient or concat coverage space). ``method="infomax"`` solves the
relaxed objective s^T x - alpha x^T K x with the kNN kernel of the same similarity and beta = lam
(core/selection/relaxation.py). ``gumbel_topk`` is a stochastic variant that adds Gumbel noise to
log-importance.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from omniselect.core.datatypes import UnifiedRecord
from omniselect.core.signals.base import minmax
from omniselect.core.signals.redundancy import hashed_features
from omniselect.core.selection.base import Selector


def gumbel_topk(importance, k: int, rng: np.random.Generator) -> List[int]:
    """Perturb log-importance with Gumbel noise and take the top-k (stochastic)."""
    imp = np.asarray(importance, dtype=float)
    k = max(0, min(int(k), imp.shape[0]))
    if k == 0:
        return []
    keys = np.log(np.clip(imp, 1e-9, None)) + rng.gumbel(size=imp.shape)
    return [int(i) for i in np.argsort(-keys, kind="stable")[:k]]


class BudgetSelector(Selector):
    """Budget selector of the fusion cells: greedy importance plus diversity, or its relaxation.

    ``lam`` weights the diversity bonus, ``method`` is ``greedy``, ``infomax`` or ``gumbel``,
    ``feat_dim`` is the width of the hashed features used when no feature matrix is given, and
    ``seed`` seeds the Gumbel variant.
    """

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
        similarity=None,
        **kwargs,
    ) -> List[int]:
        """Pool indices of the k selected records in selection order (ties go to the lower index)."""
        n = len(records)
        k = max(0, min(int(k), n))
        if k == 0:
            return []
        imp = minmax(importance)
        if self.method == "gumbel":
            return gumbel_topk(imp, k, np.random.default_rng(self.seed))

        feats = features if features is not None else hashed_features(records, dim=self.feat_dim)
        if self.method == "infomax":
            from omniselect.core.selection.relaxation import infomax_select

            sel, _ = infomax_select(imp, feats, k, beta=self.lam, similarity=similarity)
            return [int(i) for i in sel]
        selected: List[int] = []
        chosen = np.zeros(n, dtype=bool)
        max_sim = np.zeros(n, dtype=float)
        for _ in range(k):
            gain = imp + self.lam * (1.0 - max_sim)
            gain[chosen] = -np.inf
            j = int(np.argmax(gain))
            selected.append(j)
            chosen[j] = True
            sims = feats @ feats[j] if similarity is None else similarity.column(j)  # rows L2-normalized
            max_sim = np.maximum(max_sim, sims)
        return selected
