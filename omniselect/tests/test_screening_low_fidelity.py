"""Low-fidelity screening: halving on a cheap utility, full-fidelity finalists, Kendall tau, screen stage."""
from __future__ import annotations

import numpy as np
import pytest

from omniselect.config.config import OmniSelectConfig
from omniselect.core.adjudication.controller import adjudicate
from omniselect.core.adjudication.screening import kendall_tau, low_fidelity_halving, successive_halving
from omniselect.core.datatypes import Modality, UnifiedRecord
from omniselect.core.selection.fusion_grid import FusionGrid
from tracks.common.downstream import stage_plan
from tracks.common.experiment import screening_subsample


def test_low_fidelity_halving_scores_finalists_at_full_fidelity():
    cells = [(f"c{i}", [i]) for i in range(12)]
    full_calls = []

    def low(sel):
        return float(sel[0]) + (0.5 if sel[0] == 3 else 0.0)

    def full(sel):
        full_calls.append(sel[0])
        return 10.0 * sel[0]

    pool, trace, scores = low_fidelity_halving(cells, low, full, keep=4)
    reference_pool, _ = successive_halving(cells, low, keep=4)
    assert [c[0] for c in pool] == [c[0] for c in reference_pool]
    assert sorted(full_calls) == sorted(c[1][0] for c in pool)
    assert scores == {c[0]: 10.0 * c[1][0] for c in pool}
    assert trace.evaluations == 12 + 6


def test_kendall_tau_edge_cases():
    assert kendall_tau([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert kendall_tau([1.0], [2.0]) is None and kendall_tau([1, 1, 1], [1, 2, 3]) is None


def _toy():
    rng = np.random.default_rng(2)
    n, k = 60, 20
    records = [UnifiedRecord(id=str(i), modality=Modality.TEXT, domain="x", text="") for i in range(n)]
    S = rng.random((3, n))
    feats = rng.normal(size=(n, 4))
    quality = S[0] + 0.2 * rng.random(n)
    refs = [("random", list(rng.permutation(n)[:k]))]
    return records, S, feats, quality, refs, k


def test_adjudicate_low_fidelity_records_tau_and_needs_callback():
    records, S, feats, quality, refs, k = _toy()

    def gain(sel):
        return float(quality[list(sel)].mean())

    def cheap(sel):
        return float(quality[list(sel)[: len(sel) // 2]].mean())

    cfg = OmniSelectConfig.preset("v2", **{"gate.kind": "margin", "screening.kind": "low_fidelity",
                                           "precheck.enabled": False})
    grid = FusionGrid(lam_grid=(0.0, 0.5))
    with pytest.raises(ValueError):
        adjudicate(records=records, scores=S, features=feats, k=k, cfg=cfg, grid=grid, references=refs,
                   gain_rank=gain, gain_con=gain, seed=0)
    election = adjudicate(records=records, scores=S, features=feats, k=k, cfg=cfg, grid=grid, references=refs,
                          gain_rank=gain, gain_con=gain, screen_gain=cheap, seed=0)
    screening = election.screening
    assert screening["applied"] and screening["fidelity"] == "low" and screening["subsample"] == 0.4
    finalists = [c for c in election.candidates if c.stage == "finalist"]
    assert set(screening["finalists_full_u_con"]) == {c.name for c in finalists}
    for c in finalists:
        assert c.u_con == pytest.approx(gain(c.selection))
    assert screening["kendall_tau_finalists"] is None or -1.0 <= screening["kendall_tau_finalists"] <= 1.0


def test_screen_stage_schedule_and_subsample():
    cfg = OmniSelectConfig.preset("v2")
    plan = stage_plan(cfg, scoring={"max_iter": 150}, reported={"max_iter": 300}, seeds={"con": 1, "rank": 2,
                      "report": 3}, schedule_key="max_iter", low={"max_iter": 60})
    assert plan.fidelity("screen") == {"max_iter": 60, "seed": 3, "subsample": 0.4}
    assert plan.cost_factor("screen") == pytest.approx(0.2)
    canonical = stage_plan(OmniSelectConfig.preset("canonical"), scoring={"max_iter": 150},
                           reported={"max_iter": 300}, seeds={"con": 1, "rank": 2, "report": 3},
                           schedule_key="max_iter")
    assert canonical.fidelity("con") == {"max_iter": 150, "seed": 1}
    assert canonical.fidelity("screen")["max_iter"] == 150
    sel = list(range(100, 150))
    sub = screening_subsample(sel, 0.4, seed=0)
    assert len(sub) == 20 and sub == sorted(sub) and set(sub) <= set(sel)
    assert screening_subsample(sel, 0.4, seed=0) == sub and screening_subsample(sel, 0.4, seed=1) != sub
