"""Uniform random selection — the simplest budget-respecting baseline.

A pure function with no I/O and no external deps beyond numpy: draw ``k`` ids
uniformly at random (without replacement) from ``ids``. Reproducible via ``seed``.
This is the canonical *lower bound* every smarter method should beat.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def select(ids: Sequence[str], k: int, seed: int = 0) -> List[str]:
    """Pick ``k`` ids uniformly at random, without replacement.

    Parameters
    ----------
    ids : sequence of str
        Candidate record ids (the pool).
    k : int
        Number of ids to keep; clamped into ``[0, len(ids)]``.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    list of str
        The selected ids, in their drawn order.
    """
    ids = list(ids)
    n = len(ids)
    k = max(0, min(int(k), n))
    if k == 0:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    return [ids[int(i)] for i in idx]
