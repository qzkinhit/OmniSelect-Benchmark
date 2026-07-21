"""Smoke test for the MODEL=rf dispatch path of scripts/run_tabular_experiment.py
(audit 1910): rf is in _VALID_MODELS but the old dispatch fell through to the
TabPFN import, crashing in any env without tabpfn.

The test runs WITHOUT network and WITHOUT tabpfn: ``sys.modules["tabpfn"] = None``
makes any ``import tabpfn`` raise ImportError (so the test FAILS on the old
fall-through code), and ``sklearn.datasets.fetch_openml`` is replaced by a tiny
in-memory synthetic table (the runner has no synthetic-dataset fixture of its
own, so the dispatch is exercised by importing the runner module and calling its
``main()`` directly). ``_trial_dump`` is stubbed so nothing is written under
``outputs/``.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUNNER = os.path.join(_REPO, "scripts", "run_tabular_experiment.py")


class _FakeOpenML:
    """Minimal stand-in for the fetch_openml return value: .data / .target."""

    def __init__(self, n_rows=200, n_feat=5, seed=1234):
        import pandas as pd

        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_rows, n_feat))
        y = (X.sum(axis=1) > 0).astype(int)
        self.data = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_feat)])
        self.target = pd.Series(y.astype(str))


def test_rf_dispatch_runs_without_tabpfn(monkeypatch):
    # --- config must be in the environment BEFORE the runner module is imported ---
    monkeypatch.setenv("MODEL", "rf")
    monkeypatch.setenv("TAB_DATASET", "synthetic-rf-smoke")
    monkeypatch.setenv("POOL_N", "60")
    monkeypatch.setenv("VAL_N", "24")
    monkeypatch.setenv("TEST_N", "40")
    monkeypatch.setenv("NOISE_FRAC", "0.40")
    monkeypatch.setenv("BUDGET_FRAC", "0.5")
    monkeypatch.setenv("KNN", "3")
    monkeypatch.setenv("SEED", "0")
    monkeypatch.setenv("METHODS", "random")
    monkeypatch.setenv("RUN_ID", "rf-smoke-test")  # skips the legacy results_*.json dump
    for var in ("SPLIT_EXPORT_DIR", "SELECT_ONLY", "PAIRED_RNG", "ROBUST_VAL", "FIDELITY_MODE"):
        monkeypatch.delenv(var, raising=False)

    # --- guard: importing tabpfn must raise ImportError (clean env, no tabpfn) ---
    monkeypatch.setitem(sys.modules, "tabpfn", None)
    with pytest.raises(ImportError):
        import tabpfn  # noqa: F401

    # --- import the runner as a module (main() only fires under __main__) ---
    spec = importlib.util.spec_from_file_location("run_tabular_experiment_rf_smoke", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.MODEL == "rf"

    # --- no network: fetch_openml -> tiny synthetic table ---
    import sklearn.datasets as _skd

    monkeypatch.setattr(_skd, "fetch_openml", lambda *a, **k: _FakeOpenML())

    # --- spy on RandomForestClassifier to prove the rf branch is actually taken ---
    import sklearn.ensemble as _ens

    real_rf = _ens.RandomForestClassifier
    calls = {"n": 0, "kwargs": None}

    def _spy_rf(*args, **kwargs):
        calls["n"] += 1
        calls["kwargs"] = dict(kwargs)
        return real_rf(*args, **kwargs)

    monkeypatch.setattr(_ens, "RandomForestClassifier", _spy_rf)

    # --- keep the repo clean: no per-trial artifact under outputs/ ---
    recorded = {}

    def _capture(results, *a, **k):
        recorded["results"] = results
        return "<not-written>"

    monkeypatch.setattr(mod, "_trial_dump", _capture)

    # --- exercise the full rf dispatch path end to end ---
    rc = mod.main()

    assert rc == 0
    assert calls["n"] >= 1, "rf branch never instantiated RandomForestClassifier"
    assert calls["kwargs"].get("random_state") is not None, "rf branch must fix a seed"
    rows = [r for r in recorded.get("results", []) if r.get("method") == "random"]
    assert len(rows) == 1
    assert math.isfinite(rows[0]["auc"]), "predict_proba output must feed ROC-AUC"
    assert "tabpfn" not in sys.modules or sys.modules["tabpfn"] is None
