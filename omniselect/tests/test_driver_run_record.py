"""A driver cell writes every file of the run record, and recompute and replay verify it.

A small TEP cell runs under both protocols. The test checks the file layout of
docs/RUN_RECORD.md, that every candidate has per-unit files for each non-empty split, that
recompute.py reproduces the stored macro-F1 from the per-unit files, that replay.py verifies
every selection, and that a second invocation skips the complete cell.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from omniselect.store.recompute import main as recompute_main
from omniselect.store.replay import main as replay_main
from tracks.common.experiment import load_track, resolve_configs, run_cell

TOP_LEVEL = ("config.json", "splits.json", "leaderboard.json", "decision.json", "timings.json", "metrics.json",
             "log.txt")


@pytest.mark.parametrize("protocol", ["canonical", "v2"])
def test_process_cell_writes_complete_record(tmp_path, protocol):
    track = load_track("process")
    tcfg, ocfg = resolve_configs(track, "tep21", learner="mlp", seed=1, protocol=protocol, smoke=True,
                                 track_overrides={"pool_n": 200, "val_n": 200, "test_n": 150, "mlp_max_iter": 20},
                                 omni_overrides={})
    summary = run_cell(track, tcfg, ocfg, out_root=tmp_path, batch="rr", cli=["test"])
    cell = tmp_path / "rr" / "process" / "tep21" / "mlp" / "seed_1"
    for name in TOP_LEVEL:
        assert (cell / name).is_file(), name
    for name in ("authenticity.npy", "influence.npy", "redundancy.npy", "reference_sample_ids.json"):
        assert (cell / "signals" / name).is_file(), name
    splits = json.loads((cell / "splits.json").read_text())
    expected_splits = [s for s in ("con", "rank", "conf", "test") if splits["counts"][s] > 0]
    assert ("conf" in expected_splits) == (protocol == "v2")
    candidates = sorted((cell / "candidates").iterdir())
    assert len(candidates) > 10
    for cdir in candidates:
        with np.load(cdir / "selection.npz") as z:
            assert str(z["role"]) in ("reference", "challenger", "screened_out", "baseline_row")
        for split in expected_splits:
            assert (cdir / "per_unit" / f"{split}.npz").is_file(), (cdir.name, split)
        scores = json.loads((cdir / "scores.json").read_text())
        assert set(scores["utility"]) == set(expected_splits)
    reference = json.loads((cell / "signals" / "reference_sample_ids.json").read_text())
    assert reference["source"] == ("v_con" if protocol == "v2" else "pool_clean_tag")
    decision = json.loads((cell / "decision.json").read_text())
    assert decision["elected"] == summary["elected"]
    timings = json.loads((cell / "timings.json").read_text())
    assert timings["cache"]["enabled"] == (protocol == "v2")
    assert timings["fde"]["total"] > 0
    assert recompute_main([str(cell), "--metric", "macro_f1", "--check"]) == 0
    assert recompute_main([str(cell), "--metric", "accuracy", "--split", "rank", "--check"]) == 0
    assert replay_main([str(cell)]) == 0
    again = run_cell(track, tcfg, ocfg, out_root=tmp_path, batch="rr", cli=["test"])
    assert again["status"] == "skipped_complete"
    index = (tmp_path / "rr" / "index.jsonl").read_text().splitlines()
    assert len(index) == 1 and json.loads(index[0])["elected"] == summary["elected"]


def test_standalone_method_run(tmp_path):
    track = load_track("process")
    tcfg, ocfg = resolve_configs(track, "tep21", learner="mlp", seed=0, protocol="v2", smoke=True,
                                 track_overrides={"pool_n": 200, "val_n": 200, "test_n": 150, "mlp_max_iter": 20,
                                                  "methods": ("herding",)},
                                 omni_overrides={})
    summary = run_cell(track, tcfg, ocfg, out_root=tmp_path, batch="solo", cli=["test"], standalone=True)
    cell = tmp_path / "solo" / "process" / "tep21" / "mlp" / "seed_0"
    decision = json.loads((cell / "decision.json").read_text())
    assert decision["controller"] == "off" and summary["elected"] is None
    assert [p.name for p in (cell / "candidates").iterdir()] == ["herding"]


def test_per_unit_scope_finalists_keeps_scalars_for_screened_out_cells(tmp_path):
    track = load_track("process")
    tcfg, ocfg = resolve_configs(track, "tep21", learner="mlp", seed=1, protocol="v2", smoke=True,
                                 track_overrides={"pool_n": 200, "val_n": 200, "test_n": 150, "mlp_max_iter": 20},
                                 omni_overrides={"store.per_unit_scope": "finalists", "screening.sh_keep": 2})
    run_cell(track, tcfg, ocfg, out_root=tmp_path, batch="fin", cli=["test"])
    cell = tmp_path / "fin" / "process" / "tep21" / "mlp" / "seed_1"
    screened, kept = 0, 0
    for cdir in sorted((cell / "candidates").iterdir()):
        scores = json.loads((cdir / "scores.json").read_text())
        with np.load(cdir / "selection.npz") as z:
            stage = str(z["stage"])
        if stage == "screened_out":
            screened += 1
            assert not (cdir / "per_unit").exists() and scores["per_unit_written"] is False
            assert "macro_f1" in scores["metrics"]["test"] and scores["test_utility"] is not None
        else:
            kept += 1
            assert (cdir / "per_unit" / "conf.npz").is_file() and scores["per_unit_written"] is True
    assert screened > 0 and kept > 0
    assert recompute_main([str(cell), "--metric", "macro_f1", "--check"]) == 0
    assert replay_main([str(cell)]) == 0
