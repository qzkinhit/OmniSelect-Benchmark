"""Selectors: TopK by importance, BudgetSelector greedy diversity, gumbel repro."""
from __future__ import annotations

import numpy as np

from mmdataselect.datatypes import DOMAIN_GENERAL, Modality, UnifiedRecord
from mmdataselect.selectors.base import TopKSelector
from mmdataselect.selectors.budget_select import BudgetSelector, gumbel_topk
from mmdataselect.signals.redundancy import hashed_features, set_redundancy


def _rec(i, text):
    return UnifiedRecord(id=f"r{i}", modality=Modality.TEXT, domain=DOMAIN_GENERAL, text=text)


def _redundant_then_diverse_pool():
    """A pool whose highest-importance records are a redundant cluster.

    Records 0..3 share identical text but carry the top importance; records 4..7
    are lexically distinct with slightly lower importance. A pure Top-K selector
    will pile up the redundant cluster, whereas greedy diversity should spread out.
    """
    cluster_text = "the quick brown fox jumps over the lazy dog every single day"
    distinct = [
        "volcanic island geology and plate tectonic subduction zones",
        "matrix eigenvalue decomposition and singular value analysis",
        "baroque counterpoint fugue structure and harmonic resolution",
        "distributed raft consensus leader election and log replication",
    ]
    records = [_rec(i, cluster_text) for i in range(4)]
    records += [_rec(4 + j, t) for j, t in enumerate(distinct)]
    # cluster gets the top importance; distinct records are just below it.
    importance = np.array([0.99, 0.98, 0.97, 0.96, 0.90, 0.88, 0.86, 0.84])
    return records, importance


def test_topk_selects_highest_importance():
    records = [_rec(i, f"text number {i}") for i in range(6)]
    importance = np.array([0.1, 0.9, 0.3, 0.8, 0.2, 0.7])
    idx = TopKSelector().select(records, importance, k=3)
    assert sorted(idx) == [1, 3, 5]  # the three largest scores


def test_topk_clamps_k_to_pool():
    records = [_rec(i, f"t{i}") for i in range(3)]
    idx = TopKSelector().select(records, np.array([0.2, 0.5, 0.1]), k=99)
    assert sorted(idx) == [0, 1, 2]


def test_topk_zero_k_is_empty():
    records = [_rec(i, f"t{i}") for i in range(3)]
    assert TopKSelector().select(records, np.array([0.1, 0.2, 0.3]), k=0) == []


def test_budget_selector_keeps_exactly_k():
    records, importance = _redundant_then_diverse_pool()
    feats = hashed_features(records)
    idx = BudgetSelector(lam=0.5, method="greedy", seed=0).select(
        records, importance, k=4, features=feats
    )
    assert len(idx) == 4
    assert len(set(idx)) == 4  # no duplicate picks


def test_budget_greedy_not_more_redundant_than_topk():
    records, importance = _redundant_then_diverse_pool()
    feats = hashed_features(records)
    k = 4

    topk_idx = TopKSelector().select(records, importance, k=k)
    greedy_idx = BudgetSelector(lam=0.7, method="greedy", seed=0).select(
        records, importance, k=k, features=feats
    )

    topk_sel = [records[i] for i in topk_idx]
    greedy_sel = [records[i] for i in greedy_idx]

    r_topk = set_redundancy(topk_sel)
    r_greedy = set_redundancy(greedy_sel)
    # diversity-aware greedy must not produce a MORE redundant subset than plain Top-K.
    assert r_greedy <= r_topk + 1e-9
    # and on this adversarial pool it should be strictly less redundant.
    assert r_greedy < r_topk


def test_budget_greedy_starts_from_top_importance():
    records, importance = _redundant_then_diverse_pool()
    feats = hashed_features(records)
    idx = BudgetSelector(lam=0.5, method="greedy", seed=0).select(
        records, importance, k=3, features=feats
    )
    # first pick maximizes importance + lam*(1-0) -> the global importance argmax.
    assert idx[0] == int(np.argmax(importance))


def test_budget_selector_zero_k_empty():
    records, importance = _redundant_then_diverse_pool()
    idx = BudgetSelector().select(records, importance, k=0)
    assert idx == []


def test_gumbel_reproducible_with_same_seed():
    records, importance = _redundant_then_diverse_pool()
    a = BudgetSelector(method="gumbel", seed=123).select(records, importance, k=4)
    b = BudgetSelector(method="gumbel", seed=123).select(records, importance, k=4)
    assert a == b
    assert len(a) == 4


def test_gumbel_topk_direct_reproducible_and_sized():
    importance = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    out1 = gumbel_topk(importance, k=3, rng=rng1)
    out2 = gumbel_topk(importance, k=3, rng=rng2)
    assert out1 == out2
    assert len(out1) == 3
    assert len(set(out1)) == 3


def test_gumbel_topk_zero_k_empty():
    assert gumbel_topk(np.array([0.1, 0.2]), k=0, rng=np.random.default_rng(0)) == []
