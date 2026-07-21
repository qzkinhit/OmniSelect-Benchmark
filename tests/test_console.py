"""MultiActorConsole: importance shape/range + softmax-normalized adaptive update.

Uses only model-free actors (RedundancySignal + CPU-proxy InfluenceSignal), so the
console runs without torch.
"""
from __future__ import annotations

import numpy as np

from mmdataselect.fusion.console import MultiActorConsole
from mmdataselect.signals.influence import InfluenceSignal
from mmdataselect.signals.redundancy import RedundancySignal


def _console():
    actors = [
        ("redundancy", RedundancySignal()),
        ("influence", InfluenceSignal(model_name=None)),
    ]
    return MultiActorConsole(actors, lr=0.5)


def test_actor_scores_shape_and_range(small_pool):
    c = _console()
    S = c.actor_scores(small_pool)
    assert S.shape == (2, len(small_pool))  # (m actors, N records)
    assert np.all(S >= 0.0 - 1e-9)
    assert np.all(S <= 1.0 + 1e-9)


def test_importance_shape_and_range(small_pool):
    c = _console()
    imp = c.importance(small_pool)
    assert imp.shape == (len(small_pool),)
    # convex combination of [0,1] actor scores stays within [0,1].
    assert np.all(imp >= 0.0 - 1e-9)
    assert np.all(imp <= 1.0 + 1e-9)


def test_initial_weights_uniform_and_normalized():
    c = _console()
    w = c.weights()
    assert w.shape == (2,)
    assert np.isclose(w.sum(), 1.0)
    # zero-initialized theta -> uniform softmax.
    assert np.allclose(w, 0.5)


def test_importance_consistent_with_scores_arg(small_pool):
    c = _console()
    S = c.actor_scores(small_pool)
    imp_a = c.importance(small_pool, scores=S)
    imp_b = c.importance(small_pool)
    assert np.allclose(imp_a, imp_b)
    # explicit weighted combination matches the documented formula.
    w = c.weights()
    expected = (w[:, None] * S).sum(axis=0)
    assert np.allclose(imp_a, expected)


def test_update_keeps_softmax_normalized():
    c = _console()
    w = c.update({"redundancy": 1.0, "influence": 0.0})
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= 0.0)


def test_update_shifts_weight_toward_higher_reward_actor():
    c = _console()
    before = c.weights()
    after = c.update({"redundancy": 1.0, "influence": 0.0})
    # the higher-reward actor (redundancy) gains weight; the other loses.
    ri = c.names.index("redundancy")
    ii = c.names.index("influence")
    assert after[ri] > before[ri]
    assert after[ii] < before[ii]
    assert after[ri] > after[ii]


def test_repeated_updates_accumulate_preference():
    c = _console()
    ri = c.names.index("redundancy")
    w1 = c.update({"redundancy": 1.0, "influence": 0.0}).copy()
    w2 = c.update({"redundancy": 1.0, "influence": 0.0}).copy()
    # consistently rewarding the same actor monotonically increases its share.
    assert w2[ri] > w1[ri]
    assert np.isclose(w2.sum(), 1.0)


def test_console_requires_at_least_one_actor():
    import pytest

    with pytest.raises(ValueError):
        MultiActorConsole([])
