"""End-to-end select_pool: n_selected == budget.resolve(n) + diagnostics complete.

Runs model-free (model_name=None) so the influence actor uses its CPU proxy and no
torch import is triggered.
"""
from __future__ import annotations

import numpy as np

from mmdataselect.api import select_pool
from mmdataselect.budget import Budget
from mmdataselect.datatypes import DOMAIN_GENERAL, Modality, UnifiedRecord


def _rec(i, text):
    return UnifiedRecord(id=f"r{i}", modality=Modality.TEXT, domain=DOMAIN_GENERAL, text=text)


def test_select_pool_respects_fraction_budget(small_pool):
    budget = Budget(kind="fraction", value=0.5)
    n = len(small_pool)
    expected_k = budget.resolve(n, token_counts=[len((r.text or "").split()) for r in small_pool])

    res = select_pool(small_pool, budget, model_name=None, method="greedy", seed=0)

    assert len(res.selected_idx) == expected_k
    assert len(res.selected_ids) == expected_k
    # ids correspond to the chosen indices.
    assert res.selected_ids == [small_pool[i].id for i in res.selected_idx]
    # all selected ids exist in the pool and are unique.
    pool_ids = {r.id for r in small_pool}
    assert set(res.selected_ids) <= pool_ids
    assert len(set(res.selected_ids)) == len(res.selected_ids)


def test_select_pool_respects_records_budget(small_pool):
    budget = Budget(kind="records", value=2)
    res = select_pool(small_pool, budget, model_name=None, method="greedy", seed=0)
    assert len(res.selected_idx) == budget.resolve(len(small_pool))
    assert len(res.selected_idx) == 2


def test_select_pool_importance_shape_and_weights(small_pool):
    res = select_pool(small_pool, Budget(kind="fraction", value=0.5), model_name=None, seed=0)
    assert isinstance(res.importance, np.ndarray)
    assert res.importance.shape == (len(small_pool),)
    # actor weights are returned per-name and sum to 1 (softmax over theta).
    assert set(res.weights.keys()) == {"redundancy", "influence"}
    assert np.isclose(sum(res.weights.values()), 1.0)


def test_select_pool_diagnostics_fields_present(small_pool):
    res = select_pool(small_pool, Budget(kind="fraction", value=0.5), model_name=None, seed=0)
    d = res.diagnostics
    for key in (
        "n_total",
        "n_selected",
        "keep_ratio",
        "set_redundancy_pool",
        "set_redundancy_selected",
        "mean_importance_pool",
        "mean_importance_selected",
    ):
        assert key in d, f"missing diagnostic: {key}"
    assert d["n_total"] == len(small_pool)
    assert d["n_selected"] == len(res.selected_idx)
    assert 0.0 <= d["keep_ratio"] <= 1.0
    assert 0.0 <= d["set_redundancy_pool"] <= 1.0
    assert 0.0 <= d["set_redundancy_selected"] <= 1.0


def test_select_pool_empty_returns_empty():
    res = select_pool([], Budget(kind="fraction", value=0.5), model_name=None)
    assert res.selected_idx == []
    assert res.selected_ids == []
    assert res.importance.shape == (0,)
    assert res.diagnostics["n_total"] == 0
    assert res.diagnostics["n_selected"] == 0


def test_select_pool_gumbel_reproducible(small_pool):
    a = select_pool(small_pool, Budget(kind="records", value=3), model_name=None, method="gumbel", seed=11)
    b = select_pool(small_pool, Budget(kind="records", value=3), model_name=None, method="gumbel", seed=11)
    assert a.selected_idx == b.selected_idx
    assert a.selected_ids == b.selected_ids


def test_select_pool_greedy_reduces_or_matches_pool_redundancy():
    # A pool with a heavy redundant cluster: the selected subset's set-redundancy
    # should not exceed the whole pool's (diversity-aware greedy avoids piling dups).
    dup = "the quick brown fox jumps over the lazy dog repeatedly and again"
    records = [_rec(i, dup) for i in range(5)]
    records += [
        _rec(5, "eigenvalues singular values and matrix factorization methods"),
        _rec(6, "volcanic subduction zones and tectonic plate boundaries"),
        _rec(7, "raft consensus leader election and replicated state machines"),
    ]
    res = select_pool(records, Budget(kind="fraction", value=0.5), model_name=None, method="greedy", seed=0)
    d = res.diagnostics
    assert d["set_redundancy_selected"] <= d["set_redundancy_pool"] + 1e-6
