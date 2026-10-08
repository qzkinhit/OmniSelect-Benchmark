"""The tabular track runs with the rf learner end to end without TabPFN and without network.

``sys.modules['tabpfn'] = None`` makes any TabPFN import fail, and fetch_openml returns a small
synthetic table, so the test checks that the rf path never touches TabPFN and that the run
record carries finite AUC values computed from predict_proba.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import pytest

from tracks.common.experiment import load_track, resolve_configs, run_cell


class _FakeOpenML:
    def __init__(self, n_rows=240, n_feat=5, seed=1234):
        import pandas as pd

        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_rows, n_feat))
        y = (X.sum(axis=1) > 0).astype(int)
        self.data = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_feat)])
        self.target = pd.Series(y.astype(str))
        self.details = {"id": "synthetic", "name": "synthetic", "version": "1", "file_id": "none",
                        "md5_checksum": "synthetic"}


@pytest.mark.parametrize("protocol", ["canonical", "v2"])
def test_rf_learner_runs_without_tabpfn(monkeypatch, tmp_path, protocol):
    monkeypatch.setitem(sys.modules, "tabpfn", None)
    with pytest.raises(ImportError):
        import tabpfn  # noqa: F401
    import sklearn.datasets as skd

    monkeypatch.setattr(skd, "fetch_openml", lambda *a, **k: _FakeOpenML())
    track = load_track("tabular")
    tcfg, ocfg = resolve_configs(
        track, "electricity", learner="rf", seed=0, protocol=protocol, smoke=True,
        track_overrides={"openml_name": "synthetic", "pool_n": 80, "val_n": 80, "test_n": 60, "knn": 5,
                         "methods": ("full", "random", "influence_only", "mmds_adapt"), "dsdm_runs": 2,
                         "dmf_rounds": 1, "influence_ref_n": 30},
        omni_overrides={},
    )
    summary = run_cell(track, tcfg, ocfg, out_root=tmp_path, batch="rf", cli=["test"])
    assert summary["status"] == "ok"
    cell = tmp_path / "rf" / "tabular" / "electricity" / "rf" / "seed_0"
    metrics = json.loads((cell / "metrics.json").read_text())["rows"]
    assert math.isfinite(metrics["random"]["test_metrics"]["auc"])
    assert math.isfinite(metrics["mmds_adapt"]["u_test"])
    assert sys.modules.get("tabpfn") is None
