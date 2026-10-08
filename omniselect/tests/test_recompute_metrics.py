"""recompute.py reproduces the stored metrics from per-unit files for every family."""
from __future__ import annotations

import math

import numpy as np

from omniselect.store import metrics
from omniselect.store.recompute import main as recompute_main
from omniselect.store.recompute import recompute_cell
from omniselect.store.run_record import RunRecord


def _pu_sets(rng):
    target = rng.integers(0, 2, 120)
    proba = np.clip(rng.random((120, 1)) * 0.6 + 0.4 * target[:, None], 0, 1)
    proba = np.hstack([1 - proba, proba])
    cls = {"target": target, "prediction": proba.argmax(1), "proba": proba, "classes": np.array([0, 1])}
    y = rng.normal(size=(60, 24))
    fc = {"target": y, "prediction": y + rng.normal(scale=0.3, size=y.shape), "last_value": y[:, 0] + 0.1,
          "start": np.arange(60)}
    tx = {"nll": rng.random(30) + 1.0, "n_tokens": rng.integers(5, 50, 30),
          "domain": np.array(["math", "code", "general"] * 10)}
    return cls, fc, tx


def test_metric_values_from_arrays():
    rng = np.random.default_rng(0)
    cls, fc, tx = _pu_sets(rng)
    from sklearn.metrics import f1_score, roc_auc_score

    assert math.isclose(metrics.macro_f1(cls), f1_score(cls["target"], cls["prediction"], average="macro"))
    assert math.isclose(metrics.auc(cls), roc_auc_score(cls["target"], cls["proba"][:, 1]))
    naive = np.abs(fc["target"] - fc["last_value"][:, None]).mean()
    assert math.isclose(metrics.mase(fc), np.abs(fc["prediction"] - fc["target"]).mean() / (naive + 1e-9))
    ppl = metrics.per_domain_ppl(tx)
    assert set(ppl) == {"math", "code", "general"}
    assert math.isclose(metrics.gmean_ppl(tx), float(np.exp(np.mean(np.log(list(ppl.values()))))))
    assert 0.0 <= metrics.ece(cls) <= 1.0


def test_recompute_matches_stored_values(tmp_path):
    rng = np.random.default_rng(1)
    cls, fc, tx = _pu_sets(rng)
    rec = RunRecord(tmp_path / "cell")
    rec.prepare()
    for name, pu in (("cls", cls), ("fc", fc), ("tx", tx)):
        rec.write_candidate(name, selection=[0, 1], n_pool=5, budget=2, role="reference", stage="reference",
                            selection_secs=0.0, per_unit={"test": pu}, scores={})
    for metric, cand in (("macro_f1", "cls"), ("auc", "cls"), ("mase", "fc"), ("gmean_ppl", "tx")):
        rows = [r for r in recompute_cell(rec.dir, metric, "test") if r["candidate"] == cand]
        assert rows and math.isclose(rows[0]["value"], rows[0]["stored"], rel_tol=0, abs_tol=1e-12)
    assert recompute_main([str(rec.dir), "--metric", "macro_f1", "--check"]) == 0


def test_multiclass_auc_matches_sklearn_and_handles_a_missing_class():
    import pytest
    from sklearn.metrics import roc_auc_score

    from omniselect.store.metrics import auc

    rng = np.random.default_rng(3)
    proba = rng.dirichlet(np.ones(4), 200)
    target = rng.integers(0, 4, 200)
    full = {"proba": proba, "target": target, "classes": np.arange(4)}
    assert auc(full) == pytest.approx(roc_auc_score(target, proba, multi_class="ovr"))
    partial = {"proba": proba[:, :3] / proba[:, :3].sum(1, keepdims=True), "target": target,
               "classes": np.arange(3)}
    value = auc(partial)
    assert 0.0 <= value <= 1.0
