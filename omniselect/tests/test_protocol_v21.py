"""Protocol v2.1: alignment channel, gradient coverage space, solver challengers, four-channel grid."""
from __future__ import annotations

import json

import numpy as np
import pytest

from benchmark.Methods._gradients import LastLayer, append_bias
from benchmark.Methods.GLISTER.method import glister
from omniselect.config.config import OmniSelectConfig
from omniselect.core.datatypes import Modality, UnifiedRecord
from omniselect.core.portfolio.membership import reference_members
from omniselect.core.selection.budget_select import BudgetSelector
from omniselect.core.selection.fusion_grid import W3, cell_name, extend_grid, fusion_select, parse_cell_name
from omniselect.core.selection.similarity import Similarity
from omniselect.core.signals.alignment import gradient_alignment, mean_gradient


def _layer(seed=0, n=120, d=5, m=3):
    rng = np.random.default_rng(seed)
    phi = append_bias(rng.normal(size=(n, d)))
    y = rng.integers(0, m, n)
    phi_v = append_bias(rng.normal(size=(40, d)))
    yv = rng.integers(0, m, 40)
    return LastLayer("softmax", phi, y, 0.3 * rng.normal(size=(d + 1, m)), phi_v, yv, labels=y)


def test_alignment_is_glisters_round0_gain_normalized():
    model = _layer()
    err = model.errors(model.phi, model.target)
    err_v = model.errors(model.phi_val, model.target_val)
    align = gradient_alignment(model.phi, err, model.phi_val, err_v)
    g_val = mean_gradient(model.phi_val, err_v)
    gain = np.sum((model.phi @ g_val) * err, axis=1)         # GLISTER's gain at Theta_0 (empty subset)
    norm = np.linalg.norm(model.phi, axis=1) * np.linalg.norm(err, axis=1) * np.linalg.norm(g_val)
    assert np.allclose(align * norm, gain) and np.all(np.abs(align) <= 1 + 1e-12)
    explicit = np.array([np.sum(np.outer(p, e) * g_val) / (np.linalg.norm(np.outer(p, e)) * np.linalg.norm(g_val))
                         for p, e in zip(model.phi, err)])
    assert np.allclose(align, explicit)
    # GLISTER's first pick is the record with the largest un-normalized gain
    assert glister(model, 1, rounds=1)[0] == int(np.argmax(gain))


def test_gradient_similarity_equals_outer_product_cosine():
    rng = np.random.default_rng(1)
    phi, err = rng.normal(size=(7, 4)), rng.normal(size=(7, 3))
    feats = rng.normal(size=(7, 6))
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    g = np.stack([np.outer(p, e).ravel() for p, e in zip(phi, err)])
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    grad = Similarity.build("gradient", feats, phi, err)
    assert np.allclose(grad.block(np.arange(7)), g @ g.T) and np.allclose(grad.column(2), g @ g[2])
    concat = Similarity.build("concat", feats, phi, err)
    assert np.allclose(concat.block(np.arange(7)), 0.5 * (feats @ feats.T + g @ g.T))
    assert np.allclose(grad.subset(np.array([1, 3])).block(np.array([0])), (g[[1, 3]] @ g[[1, 3]].T)[:1])
    with pytest.raises(ValueError):
        Similarity.build("gradient", feats)


def test_infomax_solver_and_mmr():
    rng = np.random.default_rng(2)
    n, k = 80, 15
    records = [UnifiedRecord(id=str(i), modality=Modality.TEXT, domain="x", text="") for i in range(n)]
    feats = rng.normal(size=(n, 6))
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    imp = rng.random(n)
    mmr0 = BudgetSelector(lam=0.0).select(records, imp, k, features=feats)
    solver0 = BudgetSelector(lam=0.0, method="infomax").select(records, imp, k, features=feats)
    assert solver0 == mmr0
    solver = BudgetSelector(lam=0.5, method="infomax").select(records, imp, k, features=feats)
    assert len(solver) == k == len(set(solver))
    S = np.stack([imp, rng.random(n), rng.random(n), rng.random(n)])
    for q, lam in ((0.0, 0.5), (0.25, 0.6)):
        sel = fusion_select(S, S[0], records, feats, (0.25, 0.25, 0.25, 0.25), q, lam, k, method="infomax")
        assert len(sel) == k == len(set(sel))
    assert fusion_select(S, S[0], records, feats, (1, 0, 0, 0), 0.0, 0.0, k, method="infomax") == \
        fusion_select(S, S[0], records, feats, (1, 0, 0, 0), 0.0, 0.0, k)


def test_grid_extension_and_names():
    grid = extend_grid(W3)
    assert len(grid) == 7 + 1 + 7 + 1 and grid[0] == (1.0, 0.0, 0.0, 0.0) and grid[7] == (0.0, 0.0, 0.0, 1.0)
    assert grid[8] == (0.5, 0.0, 0.0, 0.5) and grid[-1] == (0.25, 0.25, 0.25, 0.25)
    name = cell_name(grid[8], 0.25, 0.6)
    w, q, lam = parse_cell_name(name + " solver=infomax")
    assert name == "fuse w=(0.5, 0.0, 0.0, 0.5) q=0.25 lam=0.6" and list(w) == [0.5, 0, 0, 0.5] and lam == 0.6
    assert parse_cell_name("fuse w=(0.5, 0, 0.5) q=0.0 lam=0.0")[0].shape == (3,)


def test_v21_preset_and_membership():
    v21 = OmniSelectConfig.preset("v2_1")
    v2 = OmniSelectConfig.preset("v2")
    assert v21.signals.alignment and v21.synthesis.infomax_solver and v21.coverage.space == "feature"
    assert v21.portfolio.membership == "unified_v21" and v21.gate == v2.gate and v21.splits == v2.splits
    assert not v2.signals.alignment and not v2.synthesis.infomax_solver and v2.portfolio.membership == "unified"
    assert "alignment_only" in reference_members("process", "unified_v21")
    assert "alignment_only" not in reference_members("process", "unified")


def test_v21_driver_cell(tmp_path):
    from tracks.common.experiment import load_track, resolve_configs, run_cell

    track = load_track("process")
    tcfg, ocfg = resolve_configs(track, "tep21", learner="mlp", seed=1, protocol="v2_1", smoke=True,
                                 track_overrides={"pool_n": 200, "val_n": 200, "test_n": 150, "mlp_max_iter": 20},
                                 omni_overrides={"precheck.enabled": "false"})
    run_cell(track, tcfg, ocfg, out_root=tmp_path, batch="v21", cli=["test"])
    cell = tmp_path / "v21" / "process" / "tep21" / "mlp" / "seed_1"
    align = np.load(cell / "signals" / "alignment.npy")
    assert align.shape == (200,) and np.all(np.abs(align) <= 1 + 1e-9)
    timing = json.loads((cell / "signals" / "timings.json").read_text())
    assert timing["alignment_source"] == "MLP (128, 64) output layer" and timing["seconds"]["alignment"] >= 0
    board = json.loads((cell / "leaderboard.json").read_text())
    names = [r["name"] for r in board["rows"]] + board["screened_out"]
    assert "alignment_only" in names
    cells = [x for x in names if x.startswith("fuse w=")]
    assert cells and all(len(parse_cell_name(x)[0]) == 4 for x in cells)
    assert any("solver=infomax" in x for x in names) or all(
        "lam=0.0" in r["name"] for r in board["rows"] if r["stage"] == "finalist")
