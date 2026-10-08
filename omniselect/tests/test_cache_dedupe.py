"""Train-once cache counts one fit per (subset, fidelity) and grid cells dedupe by subset hash."""
from __future__ import annotations

import numpy as np

from omniselect.config.config import OmniSelectConfig
from omniselect.core.adjudication.cache import FitCache
from omniselect.core.adjudication.controller import adjudicate
from omniselect.core.datatypes import Modality, UnifiedRecord
from omniselect.core.selection.fusion_grid import FusionGrid, build_cells


class CountingLearner:
    keep_models = True

    def __init__(self, stage_fidelity=None):
        self.fits = 0
        self.scores = 0
        self.stage_fidelity = stage_fidelity or {}

    def fidelity(self, stage):
        return {"schedule": self.stage_fidelity.get(stage, "full"), "seed": 0}

    def fit(self, subset, stage):
        self.fits += 1
        return np.asarray(subset)

    def score(self, model, split):
        self.scores += 1
        return {"target": np.zeros(3, dtype=int), "prediction": np.zeros(3, dtype=int) + len(model) % 2}


def test_enabled_cache_fits_once_per_subset_and_scores_lazily():
    learner = CountingLearner()
    cache = FitCache(learner, enabled=True)
    cache.evaluate([3, 1, 2], "con", ["con"])
    cache.evaluate([1, 2, 3], "rank", ["rank"])       # same sorted subset, same fidelity key
    cache.evaluate([1, 2, 3], "report", ["con", "rank", "conf", "test"])
    assert learner.fits == 1
    assert learner.scores == 4
    stats = cache.stats()
    assert stats["fits"] == 1 and stats["cache_hits"] == 2 and stats["distinct_subsets"] == 1


def test_distinct_fidelity_keys_do_not_share_fits():
    learner = CountingLearner(stage_fidelity={"con": "short", "rank": "short", "report": "full"})
    cache = FitCache(learner, enabled=True)
    cache.evaluate([1, 2], "con", ["con"])
    cache.evaluate([1, 2], "rank", ["rank"])
    cache.evaluate([1, 2], "report", ["test"])
    assert learner.fits == 2


def test_disabled_cache_refits_every_call():
    learner = CountingLearner()
    cache = FitCache(learner, enabled=False)
    for _ in range(3):
        cache.evaluate([1, 2], "con", ["con"])
    entry = cache.evaluate([2, 1], "report", ["con", "rank", "test"])
    assert learner.fits == 4
    assert set(entry.per_unit) == {"con", "rank", "test"}


def test_grid_dedupe_merges_identical_subsets():
    rng = np.random.default_rng(0)
    n, k = 30, 10
    records = [UnifiedRecord(id=str(i), modality=Modality.TEXT, domain="x", text="") for i in range(n)]
    S = rng.random((3, n))
    feats = rng.normal(size=(n, 4))
    grid = FusionGrid(q_grid=(0.0, 0.25), lam_grid=(0.0,))
    plain = build_cells(S, records, feats, k, FusionGrid(q_grid=(0.0, 0.25), lam_grid=(0.0,)))
    deduped = build_cells(S, records, feats, k, grid, dedupe=True)
    shas = {tuple(sorted(sel)) for _, sel in plain}
    assert len(deduped) == len(shas) < len(plain)
    assert sum(len(v) for v in grid.aliases.values()) == len(plain) - len(deduped)


def test_v2_adjudication_trains_each_distinct_subset_once():
    rng = np.random.default_rng(1)
    n, k = 40, 12
    records = [UnifiedRecord(id=str(i), modality=Modality.TEXT, domain="x", text="") for i in range(n)]
    S = rng.random((3, n))
    feats = rng.normal(size=(n, 5))
    target = rng.random(n)
    learner = CountingLearner()
    cache = FitCache(learner, enabled=True)

    def gain(split):
        def g(sel):
            cache.evaluate(sel, split, [split])
            return float(target[list(sel)].mean())
        return g

    cfg = OmniSelectConfig.preset("v2", **{"gate.kind": "margin"})   # no per-unit scores in this test
    refs = [("random", list(rng.permutation(n)[:k])), ("auth_only", list(np.argsort(-S[0])[:k]))]
    election = adjudicate(records=records, scores=S, features=feats, k=k, cfg=cfg,
                          grid=FusionGrid(lam_grid=(0.0, 0.5)), references=refs,
                          gain_rank=gain("rank"), gain_con=gain("con"), seed=0)
    distinct = {tuple(sorted(c.selection)) for c in election.candidates}
    # one fit per distinct subset ever evaluated. Coordinate-ascent probes add subsets that are
    # not candidates themselves
    assert learner.fits == cache.stats()["distinct_subsets"] >= len(distinct)
    assert cache.stats()["cache_hits"] > 0
