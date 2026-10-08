"""Unseen corruption mechanisms and validation contamination (tracks/common/injection.py)."""
from __future__ import annotations

import numpy as np
import pytest

from tracks.common import injection as inj


def test_unseen_vision_picks_the_canonical_records_and_flips_within_superclasses():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 100, 600)
    superclass = np.repeat(np.arange(20), 5)                # fine classes 5s..5s+4 form superclass s
    obs, tags, plan = inj.unseen_vision_labels(labels, seed=3, noise_frac=0.4, n_classes=100, superclass=superclass)
    _, canon_tags, _ = inj.inject_vision_labels(labels, 3, 0.4)
    assert set(np.flatnonzero(tags != "high")) == set(np.flatnonzero(canon_tags != "high"))
    assert {k: len(v) for k, v in plan.groups.items()} == {"blur": 80, "jpeg": 80, "class_flip": 80}
    flipped = plan.groups["class_flip"]
    assert np.all(obs[flipped] != labels[flipped]) and np.all(superclass[obs[flipped]] == superclass[labels[flipped]])
    assert np.array_equal(obs[tags != "class_flip"], labels[tags != "class_flip"])
    # next fine class of the same superclass, cyclic
    assert inj.class_flip_target(np.array([0, 4, 7]), 100, superclass).tolist() == [1, 0, 8]


def test_cifar10_class_map_and_binary_flips():
    target = inj.class_flip_target(np.arange(10), 10, class_map=inj.CIFAR10_CLASS_MAP)
    assert target[9] == 1 and target[2] == 0 and target[3] == 5 and target[5] == 3 and target[4] == 7
    X = np.random.default_rng(1).normal(size=(200, 8))
    y = np.array([0, 1] * 100)
    Xn, obs, tags = inj.unseen_rows(X, y, seed=0, n_classes=2, noise_frac=0.4)
    flips = np.flatnonzero(tags == "class_flip")
    assert len(flips) == 40 and np.all(y[flips] == 1) and np.all(obs[flips] == 0)
    stuck = np.flatnonzero(tags == "stuck_feature")
    assert len(stuck) == 40
    median = np.median(X, axis=0)
    for i in stuck:
        changed = np.flatnonzero(Xn[i] != X[i])
        assert len(changed) <= 2 and np.allclose(Xn[i, changed], median[changed])
    assert np.array_equal(Xn[tags == "high"], X[tags == "high"])


def test_unseen_windows_faults():
    rng = np.random.default_rng(2)
    X, Y = rng.normal(size=(90, 12)), rng.normal(size=(90, 4))
    Xn, Yn, tags = inj.unseen_windows(X, Y, seed=0, noise_frac=0.3, scale=1.3, bias=1.0)
    assert sorted(set(tags)) == ["bias_drift", "high", "scale_drift", "stuck"]
    i = int(np.flatnonzero(tags == "scale_drift")[0])
    ramp = np.linspace(0, 1, 16)
    assert np.allclose(np.concatenate([Xn[i], Yn[i]]), np.concatenate([X[i], Y[i]]) * (1 + 0.3 * ramp))
    j = int(np.flatnonzero(tags == "bias_drift")[0])
    assert np.allclose(np.concatenate([Xn[j], Yn[j]]) - np.concatenate([X[j], Y[j]]), X.std() * ramp)
    k = int(np.flatnonzero(tags == "stuck")[0])
    seg = np.concatenate([Xn[k], Yn[k]])
    first = min(t for t in range(16) if np.all(seg[t:] == seg[t]))
    assert 6 <= first < 12 and np.array_equal(seg[:first], np.concatenate([X[k], Y[k]])[:first])
    assert np.array_equal(Xn[tags == "high"], X[tags == "high"])


def test_validation_contamination_kinds():
    y = np.repeat(np.arange(10), 30)
    same, pos, info = inj.contaminate_validation("none", y, 0.3, 0, modulus=10, max_offset=10)
    assert np.array_equal(same, y) and np.array_equal(pos, np.arange(300)) and info["changed"] == 0
    sym, pos, info = inj.contaminate_validation("symmetric", y, 0.3, 0, modulus=10, max_offset=10)
    assert np.array_equal(sym, inj.flip_validation_labels(y, 0.3, 10, 10)) and 50 < info["changed"] < 130
    human = y.copy()
    human[:40] = (human[:40] + 1) % 10
    nat, _, info = inj.contaminate_validation("natural", y, 0.0, 0, modulus=10, max_offset=10, natural=human)
    assert np.array_equal(nat, human) and info["changed"] == 40
    with pytest.raises(ValueError):
        inj.contaminate_validation("natural", y, 0.0, 0, modulus=10, max_offset=10)
    shifted, pos, info = inj.contaminate_validation("prior_shift", y, 0.5, 0, modulus=10, max_offset=10)
    assert len(shifted) == 300 and np.array_equal(shifted, y[pos])
    minority = np.isin(shifted, info["minority_classes"])
    assert len(info["minority_classes"]) == 5 and 0.15 < minority.mean() < 0.35
    assert inj.normalize_val_noise("class_prior_shift") == "prior_shift"
