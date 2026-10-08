"""Smoke tests for the recognized external data-selection baselines."""
import numpy as np

from benchmark.Methods import (
    ccs,
    dsir_select,
    el2n,
    grand,
    herding,
    kcenter_greedy,
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


def test_kcenter_remains_unique_when_every_feature_is_identical():
    X = np.zeros((40, 8))
    assert _valid(kcenter_greedy(X, 25, seed=3), 25, len(X))


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


def test_ccs_tied_difficulties_still_return_exact_unique_budget():
    n, classes, k = 200, 4, 60
    probabilities = np.full((n, classes), 1.0 / classes)
    labels = np.arange(n) % classes
    selected = ccs(
        probabilities,
        labels,
        k,
        is_logits=False,
        bins=50,
        cutoff=0.1,
        seed=7,
    )
    assert _valid(selected, k, n)
    assert selected == ccs(
        probabilities,
        labels,
        k,
        is_logits=False,
        bins=50,
        cutoff=0.1,
        seed=7,
    )


def test_ccs_uses_equal_width_not_quantile_strata():
    # Nine easy records and one hard record. Equal-width binning puts the hard
    # record in its own stratum, whereas quantile binning would split tied easy
    # records across artificial rank buckets.
    probabilities = np.array(
        [[0.99, 0.01]] * 9 + [[0.01, 0.99]],
        dtype=float,
    )
    labels = np.zeros(10, dtype=int)
    selected = ccs(
        probabilities,
        labels,
        4,
        is_logits=False,
        bins=2,
        cutoff=0.0,
        seed=4,
    )
    assert 9 in selected
    assert len(selected) == len(set(selected)) == 4
