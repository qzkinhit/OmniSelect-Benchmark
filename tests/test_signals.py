"""Signal contracts: [0,1] range, length-N, determinism, set-redundancy monotonicity.

All signals here are exercised model-free (no model_name), so no torch is needed.
"""
from __future__ import annotations

import numpy as np

from mmdataselect.datatypes import DOMAIN_GENERAL, Modality, UnifiedRecord
from mmdataselect.signals.base import minmax
from mmdataselect.signals.influence import InfluenceSignal
from mmdataselect.signals.redundancy import (
    RedundancySignal,
    hashed_features,
    set_redundancy,
)


def _rec(i, text):
    return UnifiedRecord(id=f"r{i}", modality=Modality.TEXT, domain=DOMAIN_GENERAL, text=text)


def test_redundancy_signal_range_and_length(small_pool):
    s = RedundancySignal().score(small_pool)
    assert isinstance(s, np.ndarray)
    assert s.shape == (len(small_pool),)
    # per-record compression ratio is bounded in [0, 1].
    assert np.all(s >= 0.0 - 1e-9)
    assert np.all(s <= 1.0 + 1e-9)


def test_influence_cpu_proxy_range_and_length(small_pool):
    # No model_name -> deterministic CPU proxy path (no torch).
    s = InfluenceSignal(model_name=None).score(small_pool)
    assert s.shape == (len(small_pool),)
    # minmax-normalized into [0, 1].
    assert np.all(s >= 0.0 - 1e-9)
    assert np.all(s <= 1.0 + 1e-9)


def test_influence_cpu_proxy_deterministic(small_pool):
    a = InfluenceSignal(model_name=None).score(small_pool)
    b = InfluenceSignal(model_name=None).score(small_pool)
    assert np.allclose(a, b)


def test_influence_proxy_prefers_lexically_richer_text():
    # unique/total ratio: repeated tokens -> low, all-unique tokens -> high.
    poor = _rec(0, "a a a a a a")
    rich = _rec(1, "alpha beta gamma delta epsilon zeta")
    s = InfluenceSignal(model_name=None).score([poor, rich])
    assert s[1] > s[0]


def test_set_redundancy_in_unit_interval(small_pool):
    r = set_redundancy(small_pool)
    assert 0.0 <= r <= 1.0


def test_set_redundancy_higher_for_more_repetitive_set():
    line = "the quick brown fox jumps over the lazy dog"
    redundant = [_rec(i, line) for i in range(8)]  # same line repeated
    diverse = [
        _rec(0, "volcanic island geology and plate tectonics"),
        _rec(1, "matrix eigenvalues and singular value decomposition"),
        _rec(2, "baroque counterpoint and fugue structure"),
        _rec(3, "distributed consensus and raft leader election"),
        _rec(4, "photosynthesis light dependent reactions in chloroplasts"),
        _rec(5, "quantum entanglement and bell inequality violations"),
        _rec(6, "renaissance perspective and vanishing point geometry"),
        _rec(7, "thermodynamic entropy and carnot cycle efficiency"),
    ]
    assert set_redundancy(redundant) > set_redundancy(diverse)


def test_set_redundancy_empty_is_zero():
    assert set_redundancy([]) == 0.0


def test_hashed_features_shape_and_determinism(small_pool):
    dim = 256
    a = hashed_features(small_pool, dim=dim)
    b = hashed_features(small_pool, dim=dim)
    assert a.shape == (len(small_pool), dim)
    # crc32 hashing -> identical across calls/processes.
    assert np.array_equal(a, b)


def test_hashed_features_rows_are_l2_normalized(small_pool):
    X = hashed_features(small_pool)
    norms = np.linalg.norm(X, axis=1)
    # every non-empty-text row is a unit vector.
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_hashed_features_identical_text_identical_rows():
    recs = [_rec(0, "same identical content here"), _rec(1, "same identical content here")]
    X = hashed_features(recs)
    assert np.allclose(X[0], X[1])
    # cosine similarity of identical rows is 1.
    assert np.isclose(float(X[0] @ X[1]), 1.0, atol=1e-6)


def test_minmax_constant_maps_to_half():
    out = minmax([3.0, 3.0, 3.0])
    assert np.allclose(out, 0.5)


def test_minmax_normalizes_to_unit_interval():
    out = minmax([0.0, 5.0, 10.0])
    assert np.isclose(out.min(), 0.0)
    assert np.isclose(out.max(), 1.0)
    assert np.isclose(out[1], 0.5)
