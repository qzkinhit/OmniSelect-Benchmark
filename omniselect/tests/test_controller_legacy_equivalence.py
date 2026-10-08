"""adjudicate() under the canonical preset reproduces the frozen 2026-07 AdaptiveController."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from omniselect.core.adjudication.controller import AdaptiveController
from omniselect.core.datatypes import Modality, UnifiedRecord
from omniselect.tests.fixtures.adaptive_legacy import AdaptiveController as LegacyController


def _gain_factory(target: np.ndarray, salt: str):
    def gain(rows):
        rows = [int(i) for i in rows]
        noise = int(hashlib.sha256((salt + str(sorted(rows))).encode()).hexdigest()[:8], 16) / 16 ** 8
        return float(target[rows].mean() + 0.02 * noise)
    return gain


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "grid",
    [
        dict(lam_grid=(0.0, 0.25, 0.6), prefilter_grid=(0.0, 0.25)),
        dict(weight_grid=[(1, 0, 0), (0, .5, .5), (.34, .33, .33)], lam_grid=(0.0, 0.5), prefilter_grid=(0.0, 0.25)),
    ],
)
def test_canonical_adjudication_matches_legacy_controller(seed, grid):
    rng = np.random.default_rng(seed)
    n, k = 60, 20
    records = [UnifiedRecord(id=str(i), modality=Modality.TEXT, domain="x", text="") for i in range(n)]
    scores = rng.random((3, n))
    feats = rng.normal(size=(n, 8))
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    target = rng.random(n)
    order = rng.permutation(n)
    extras = [
        ("random", lambda kk, o=order: [int(i) for i in o[:kk]]),
        ("auth_only", lambda kk: [int(i) for i in np.argsort(-scores[0])[:kk]]),
        ("influence_only", lambda kk: [int(i) for i in np.argsort(-scores[1])[:kk]]),
        ("auth_bottom", lambda kk: [int(i) for i in np.argsort(scores[0])[:kk]]),
    ]
    kwargs = dict(
        features=feats,
        held_out_gain=_gain_factory(target, "v2"),
        construct_gain=_gain_factory(target, "v1"),
        extra_strategies=extras,
    )
    legacy = LegacyController(seed=seed, **grid)
    current = AdaptiveController(seed=seed, **grid)
    sel_legacy = legacy.select(records, scores, k, **kwargs)
    sel_current = current.select(records, scores, k, **kwargs)
    assert sel_current == sel_legacy
    assert current.chosen_ == legacy.chosen_
    assert current.leaderboard_ == legacy.leaderboard_
    assert current.sh_stats_ == legacy.sh_stats_


def test_canonical_adjudication_matches_legacy_without_construction_split():
    rng = np.random.default_rng(7)
    n, k = 40, 12
    records = [UnifiedRecord(id=str(i), modality=Modality.TEXT, domain="x", text="") for i in range(n)]
    scores = rng.random((3, n))
    feats = rng.normal(size=(n, 6))
    target = rng.random(n)
    kwargs = dict(features=feats, held_out_gain=_gain_factory(target, "v2"))
    legacy = LegacyController(seed=0, lam_grid=(0.0,), prefilter_grid=(0.0,))
    current = AdaptiveController(seed=0, lam_grid=(0.0,), prefilter_grid=(0.0,))
    assert current.select(records, scores, k, **kwargs) == legacy.select(records, scores, k, **kwargs)
    assert current.leaderboard_ == legacy.leaderboard_
