"""Held-half coordinate ascent: steps fitted on one half of V_con are accepted only if the other half does not drop."""
from __future__ import annotations

import json

import numpy as np

from omniselect.core.adjudication.synthesis import coordinate_ascent
from omniselect.core.datatypes import Modality, UnifiedRecord
from tracks.common.experiment import held_halves, load_track, resolve_configs, run_cell, slice_units


def _setup(n=60, k=20, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.random((3, n))
    records = [UnifiedRecord(id=str(i), modality=Modality.TEXT, domain="x", text="") for i in range(n)]
    feats = rng.normal(size=(n, 4))
    cells = [(f"fuse w=({a}, {b}, {c}) q=0.0 lam=0.0", list(np.argsort(-(a * S[0] + b * S[1] + c * S[2]))[:k]), g,
              False) for (a, b, c), g in [((1, 0, 0), 0.5), ((0, 1, 0), 0.4), ((0.34, 0.33, 0.33), 0.45)]]
    return S, records, feats, cells, k


def test_default_path_is_unchanged_and_the_check_can_refuse_every_step():
    S, records, feats, cells, k = _setup()

    def fit_half(sel):                     # prefers channels 1 and 2
        return float((S[1] + S[2])[sel].mean())

    plain = coordinate_ascent(cells, S, records, feats, k, S[0], fit_half)
    stats: dict = {}
    same = coordinate_ascent(cells, S, records, feats, k, S[0], fit_half, stats=stats)
    assert plain == same and stats["held_half"] is False and stats["accepted_steps"] == stats["proposed_steps"]

    def opposite(sel):                     # every step that raises the fitting half lowers this half
        return -fit_half(sel)

    held: dict = {}
    result = coordinate_ascent(cells, S, records, feats, k, S[0], fit_half, check=opposite, stats=held)
    assert held["held_half"] and held["proposed_steps"] > 0 and held["accepted_steps"] == 0
    assert held["start_cell"] == max(cells, key=lambda t: fit_half(t[1]))[0]   # chosen on the fitting half
    start_sel = next(t[1] for t in cells if t[0] == held["start_cell"])
    assert sorted(result[1]) == sorted(start_sel) and held["u_check_half"] == -fit_half(start_sel)

    agree: dict = {}
    coordinate_ascent(cells, S, records, feats, k, S[0], fit_half, check=fit_half, stats=agree)
    assert agree["accepted_steps"] == agree["proposed_steps"] > 0


def test_halves_are_stratified_or_contiguous():
    target = np.repeat([0, 1, 2], [10, 7, 5])
    a, b = held_halves({"target": target, "prediction": target}, seed=0)
    assert len(set(a) & set(b)) == 0 and len(a) + len(b) == 22
    for c in (0, 1, 2):
        assert abs(np.sum(target[a] == c) - np.sum(target[b] == c)) <= 1
    starts = np.array([50, 10, 30, 20, 40, 0])
    a, b = held_halves({"start": starts, "prediction": np.zeros((6, 2))}, seed=0)
    assert set(starts[a]) == {0, 10, 20} and set(starts[b]) == {30, 40, 50}
    domain = np.array(["a"] * 6 + ["b"] * 4)
    a, b = held_halves({"domain": domain, "nll": np.ones(10)}, seed=1)
    assert np.sum(domain[a] == "a") == 3 and np.sum(domain[a] == "b") == 2
    sliced = slice_units({"target": target, "classes": np.arange(3)}, a)
    assert len(sliced["target"]) == len(a) and len(sliced["classes"]) == 3


def test_driver_records_held_half_steps(tmp_path):
    track = load_track("process")
    tcfg, ocfg = resolve_configs(track, "tep21", learner="mlp", seed=1, protocol="v2", smoke=True,
                                 track_overrides={"pool_n": 200, "val_n": 200, "test_n": 150, "mlp_max_iter": 20},
                                 omni_overrides={"synthesis.held_half": "true", "precheck.enabled": "false"})
    run_cell(track, tcfg, ocfg, out_root=tmp_path, batch="hh", cli=["test"])
    cell = tmp_path / "hh" / "process" / "tep21" / "mlp" / "seed_1"
    learned = [d for d in (cell / "candidates").iterdir() if d.name.startswith("fuse_learned")]
    assert learned
    info = json.loads((learned[0] / "scores.json").read_text())["selection_info"]
    assert info["held_half"] is True and info["accepted_steps"] <= info["proposed_steps"] <= info["trials"]
    assert info["halves"] and sum(info["halves"]) == json.loads((cell / "splits.json").read_text())["counts"]["con"]
