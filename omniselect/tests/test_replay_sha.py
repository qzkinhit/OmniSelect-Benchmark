"""replay.py verifies stored selections and detects a tampered index file."""
from __future__ import annotations

import numpy as np

from omniselect.store.replay import main as replay_main
from omniselect.store.replay import subset_ids, verify_cell
from omniselect.store.run_record import RunRecord
from omniselect.utils.hashing import sel_sha12


def _cell(tmp_path):
    rec = RunRecord(tmp_path / "cell")
    rec.prepare()
    rec.write_splits(ids={"pool": ["a", "b", "c", "d"], "con": [], "rank": [], "conf": [], "test": []},
                     pool_tags=["high"] * 4)
    rec.write_candidate("random", selection=[3, 1], n_pool=4, budget=2, role="reference", stage="reference",
                        selection_secs=0.0, per_unit={}, scores={})
    rec.write_json("decision.json", {"elected": "random", "elected_sel_sha12": sel_sha12([1, 3])})
    return rec.dir


def test_replay_accepts_a_clean_cell(tmp_path):
    cell = _cell(tmp_path)
    assert verify_cell(cell) == []
    assert replay_main([str(cell)]) == 0
    assert subset_ids(cell, "random") == ["b", "d"]


def test_replay_detects_tampering(tmp_path):
    cell = _cell(tmp_path)
    path = cell / "candidates" / "random" / "selection.npz"
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    data["idx"] = np.array([0, 1], dtype=np.int64)
    np.savez(path, **data)
    problems = verify_cell(cell)
    assert any("sel_sha12" in p for p in problems)
    assert replay_main([str(cell)]) == 1
