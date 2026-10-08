"""Selections break score ties by ascending pool index, independent of the numpy sort implementation."""
from __future__ import annotations

import numpy as np

from benchmark.Methods._common import top_k
from benchmark.Methods.AuthenticityOnly.method import auth_bottom, auth_only
from omniselect.core.portfolio.registry import SelectionContext
from omniselect.core.selection.fusion_grid import fusion_select

TIES = np.array([0.2, 0.5, 0.5, 0.1, 0.5, 0.2, 0.9, 0.5, 0.2, 0.1] * 5, dtype=float)   # 50 scores, heavy ties


def test_top_k_pins_tied_ids():
    assert top_k(TIES, 8) == [6, 16, 26, 36, 46, 1, 2, 4]
    # a long vector of kNN agreement counts (multiples of 1/15), where quicksort would reorder ties
    counts = np.random.default_rng(0).integers(0, 16, 4000) / 15.0
    sel = top_k(counts, 1200)
    expected = [int(i) for i in np.lexsort((np.arange(4000), -counts))[:1200]]
    assert sel == expected


def test_authenticity_rows_use_the_stable_order():
    ctx = SelectionContext(n=len(TIES), seed=0, features=np.zeros((len(TIES), 2)), auth=TIES)
    assert auth_only(ctx, 6) == [6, 16, 26, 36, 46, 1]
    assert auth_bottom(ctx, 4) == [3, 9, 13, 19]


def test_fusion_ranking_ties():
    n = 30
    S = np.stack([np.repeat([1.0, 0.0], 15), np.zeros(n), np.zeros(n)])
    records = list(range(n))
    sel = fusion_select(S, S[0], records, np.eye(n), (1.0, 0.0, 0.0), 0.0, 0.0, 5)
    assert sel == [0, 1, 2, 3, 4]
    reverse = fusion_select(1.0 - S, S[0], records, np.eye(n), (1.0, 0.0, 0.0), 0.0, 0.0, 5)
    assert reverse == [15, 16, 17, 18, 19]
