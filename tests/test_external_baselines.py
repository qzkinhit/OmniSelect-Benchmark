"""Smoke tests for the recognized external data-selection baselines."""
import numpy as np

from src.mmdataselect.selectors.external_baselines import (
    herding, kcenter_greedy, el2n, grand, dsir_select,
)


def _valid(idx, k, n):
    idx = list(idx)
    return len(idx) == k and len(set(idx)) == len(idx) and all(0 <= i < n for i in idx)


def test_geometric_coresets():
    rng = np.random.default_rng(0)
    n, d, k = 200, 16, 40
    X = rng.normal(size=(n, d))
    assert _valid(herding(X, k), k, n)
    assert _valid(kcenter_greedy(X, k), k, n)


def test_score_based_pruning():
    rng = np.random.default_rng(1)
    n, k = 200, 40
    logits = rng.normal(size=(n, 5))
    y = rng.integers(0, 5, n)
    X = rng.normal(size=(n, 16))
    assert _valid(el2n(logits, y, k), k, n)
    assert _valid(grand(logits, y, X, k), k, n)


def test_dsir_resamples_toward_target():
    rng = np.random.default_rng(2)
    n, V, k = 300, 64, 60
    counts = rng.poisson(0.5, size=(n, V)).astype(float)
    # plant a target-aligned cluster: 50 docs heavy on the first 8 features
    counts[:50, :8] += 5.0
    target = np.zeros(V)
    target[:8] = 10.0
    sel = dsir_select(counts, target, k, seed=0)
    assert _valid(sel, k, n)
    # the planted target-aligned docs should be over-represented vs a random 60/300 draw
    planted = sum(1 for i in sel if i < 50)
    assert planted > k * (50 / n)  # > chance share
