"""Run record writer round trip: files, atomic JSON, index line, and resume marker."""
from __future__ import annotations

import json

import numpy as np
import pytest

from omniselect.store.index import read_index
from omniselect.store.run_record import RunRecord, safe_name


def _classification_pu(rng, n=50, c=3):
    target = rng.integers(0, c, n)
    proba = rng.dirichlet(np.ones(c), n)
    return {"target": target, "prediction": proba.argmax(1), "proba": proba.astype(np.float32),
            "classes": np.arange(c), "correct": (proba.argmax(1) == target).astype(np.float32)}


def test_write_and_read_back_a_cell(tmp_path):
    rng = np.random.default_rng(0)
    cell = RunRecord.cell_path(tmp_path, "B", "vision", "uoft-cs/cifar100", "clip_vitb32", 0)
    rec = RunRecord(cell)
    rec.prepare()
    rec.write_config(omniselect={"protocol": "v2"}, track={"dataset": "x"}, cli=["--x"])
    rec.write_splits(ids={"pool": [0, 1, 2], "con": [5], "rank": [6], "conf": [], "test": [9]},
                     pool_tags=["high", "flip", "high"])
    rec.write_signals({"authenticity": np.arange(3.0)}, reference_ids=[5], reference_source="v_con")
    pu = {split: _classification_pu(rng) for split in ("con", "rank", "conf", "test")}
    cdir = rec.write_candidate("fuse w=(0.5, 0.5, 0) q=0.25 lam=0.6", selection=[2, 0], n_pool=3, budget=2,
                               role="challenger", stage="finalist", selection_secs=0.1, per_unit=pu,
                               scores={"utility": {"rank": 0.5}})
    assert cdir.name == safe_name("fuse w=(0.5, 0.5, 0) q=0.25 lam=0.6")
    with np.load(cdir / "selection.npz") as z:
        assert z["idx"].tolist() == [0, 2] and str(z["role"]) == "challenger"
    scores = json.loads((cdir / "scores.json").read_text())
    assert set(scores["metrics"]["test"]) >= {"accuracy", "macro_f1", "auc", "ece"}
    with np.load(cdir / "per_unit" / "test.npz") as z:
        assert np.array_equal(z["target"], pu["test"]["target"])
    assert not rec.is_complete()
    rec.finish(decision={"elected": "x"}, index_root=tmp_path / "B", status="ok", elected="x", test=0.5)
    assert rec.is_complete()
    assert read_index(tmp_path / "B")[0]["elected"] == "x"
    with pytest.raises(FileExistsError):
        RunRecord(cell).prepare()


def test_selection_outside_pool_is_rejected(tmp_path):
    rec = RunRecord(tmp_path / "cell")
    rec.prepare()
    with pytest.raises(ValueError):
        rec.write_candidate("bad", selection=[0, 5], n_pool=3, budget=2, role="reference", stage="reference",
                            selection_secs=0.0, per_unit={}, scores={})


def test_non_finite_json_is_refused(tmp_path):
    rec = RunRecord(tmp_path / "cell")
    rec.prepare()
    with pytest.raises(ValueError):
        rec.write_json("timings.json", {"x": float("nan")})


def test_environment_records_thread_settings(monkeypatch):
    from omniselect.store.run_record import environment

    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    env = environment()
    threads = env["threads"]
    assert set(threads) == {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "torch_num_threads",
                            "cpu_count", "load_avg_1min"}
    assert threads["OMP_NUM_THREADS"] == "8" and threads["MKL_NUM_THREADS"] is None
    assert threads["cpu_count"] is None or threads["cpu_count"] >= 1
    assert threads["torch_num_threads"] is None or threads["torch_num_threads"] >= 1
    json.dumps(env)
