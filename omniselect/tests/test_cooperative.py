"""Cooperative candidates of protocol v2.2: gate, vetoes, refill, selectors, learned cleanliness, preset and driver."""
from __future__ import annotations

import json

import numpy as np
import pytest

from omniselect.config.config import CleanlinessConfig, OmniSelectConfig, parse_overrides
from omniselect.core.portfolio.membership import TRACKS, challenger_members, reference_members
from omniselect.core.selection.cooperative import (
    CooperativeFamily,
    apply_vetoes,
    cell_name,
    duplicate_groups,
    gate,
    herding_order,
    kcenter_order,
    neighbour_graph,
    outlier_flags,
    parse_rho,
    refill,
)
from omniselect.core.signals.cleanliness import classification_features, learned_cleanliness


def _family(X: np.ndarray, score: np.ndarray, seed: int = 0) -> CooperativeFamily:
    return CooperativeFamily(scores={"authenticity": score}, graph=neighbour_graph(X, "euclidean"),
                             selector_features=X, seed=seed)


def test_gate_keeps_ceil_rho_k_highest_scores_ties_by_index():
    score = np.array([0.5, 0.9, 0.9, 0.1, 0.7, 0.9, 0.3, 0.2, 0.8, 0.6, 0.4, 0.0])
    assert list(gate(score, 4, 1.3)) == [1, 2, 5, 8, 4, 9]          # ceil(5.2) = 6, the three ties in index order
    assert len(gate(score, 10, 1.15)) == 12                         # ceil(11.5) capped at the pool size
    assert len(gate(np.arange(5000.0), 2000, 1.15)) == 2300         # exact product, no floating-point overshoot
    assert len(gate(score, 4, None)) == len(score)
    assert parse_rho("none") is None and parse_rho("1.3") == 1.3
    with pytest.raises(ValueError):
        parse_rho("0.9")
    assert cell_name("cleanliness", 1.3, "kcenter") == "coop gate=cleanliness rho=1.3 within=kcenter"
    assert cell_name(None, None, "herding") == "coop rho=none within=herding"


def test_outlier_veto_drops_planted_outliers():
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(size=(300, 5)), rng.normal(size=(6, 5)) + 12.0 * rng.choice([-1, 1], size=(6, 5))]
    flags, info = outlier_flags(neighbour_graph(X, "euclidean"), 3.0)
    assert flags[300:].all() and flags[:300].mean() < 0.05 and info["scale"] > 0
    kept, out_drop, dup_drop = apply_vetoes(np.arange(306), flags, None)
    assert set(range(300, 306)) <= set(out_drop) and not dup_drop and len(kept) + len(out_drop) == 306


def test_duplicate_veto_keeps_the_cleanest_member_of_each_group():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(200, 8))
    copies = [base[s] + 1e-4 * rng.normal(size=(3, 8)) for s in (3, 50, 120)]
    X = np.vstack([base, *copies])                                 # records 200..208 copy records 3, 50, 120
    labels, info = duplicate_groups(neighbour_graph(X, "euclidean"), 0.1)
    for src, copies in ((3, range(200, 203)), (50, range(203, 206)), (120, range(206, 209))):
        assert all(labels[c] == labels[src] for c in copies)
    assert len(np.unique(labels)) == 200 and info["threshold"] > 0
    score = rng.random(len(X))
    order = np.argsort(-score, kind="stable")
    kept, out_drop, dup_drop = apply_vetoes(order, None, labels)
    for group in ([3, 200, 201, 202], [50, 203, 204, 205], [120, 206, 207, 208]):
        best = max(group, key=lambda i: (score[i], -i))
        assert best in kept and all(i in dup_drop for i in group if i != best)
    assert len(dup_drop) == 9


def test_refill_restores_dropped_records_in_score_order():
    kept, extra = refill([4, 7], [9, 1, 3], 4)
    assert kept == [4, 7, 9, 1] and extra == 2
    rng = np.random.default_rng(2)
    base = rng.normal(size=(60, 4))
    X = np.r_[base, base[:10] + 1e-5]                              # ten duplicate pairs
    score = np.r_[np.linspace(1.0, 0.5, 60), np.linspace(0.99, 0.95, 10)]
    fam = _family(X, score)
    sel, info = fam.select("authenticity", 1.0, "herding", 20, name="c")
    assert len(sel) == len(set(sel)) == 20 and info["refill"] == info["outlier_drops"] + info["duplicate_drops"]
    assert info["admissible_gate"] == 20 and info["admissible_size"] == 20 and info["duplicate_drops"] > 0


def test_selectors_inside_the_gate_return_k_distinct_admissible_ids():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 6))
    score = rng.random(400)
    fam = _family(X, score, seed=5)
    admissible = set(gate(score, 100, 1.3).tolist())
    for selector in ("herding", "kcenter"):
        sel, info = fam.select("authenticity", 1.3, selector, 100, name=selector)
        assert len(sel) == len(set(sel)) == 100 and set(sel) <= admissible
        assert set(sel) <= set(fam.admissible_sets[selector].tolist())
        assert info["selector"] == selector and info["rho"] == 1.3 and info["admissible_gate"] == 130
        assert info["admissible_size"] == 130 - info["outlier_drops"] - info["duplicate_drops"] + info["refill"]
    sel, info = fam.select("authenticity", None, "herding", 100, name="none")
    assert info["admissible_gate"] == 400 and info["gate_score"] is None and len(set(sel)) == 100


def test_herding_and_kcenter_follow_the_benchmark_rules():
    from benchmark.Methods.Herding.method import herding
    from benchmark.Methods.KCenter.method import kcenter_greedy

    X = np.random.default_rng(4).normal(size=(300, 16))
    assert herding_order(X, 80) == herding(X, 80)
    assert kcenter_order(X, 80, seed=7) == kcenter_greedy(X, 80, seed=7)


def test_neighbour_graph_ties_and_queries():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    g = neighbour_graph(X, "euclidean", knn=3)
    assert list(g.index[0]) == [1, 4, 2] and g.distance[0, 0] == 0.0 and 0 not in g.index[0]
    q = neighbour_graph(X, "euclidean", knn=2, query=np.array([[0.0, 0.0]]))
    assert list(q.index[0]) == [0, 1]
    c = neighbour_graph(np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]), "cosine", knn=1)
    assert list(c.index[:, 0]) == [1, 0, 0] and c.distance[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_token_budget_family_returns_a_full_order_and_counts_the_refill():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(40, 3))
    X = np.r_[base, np.repeat(base[:1], 3, axis=0)]                # records 40, 41, 42 duplicate record 0
    score = np.r_[rng.random(40), np.full(3, 2.0)]                  # ... and are the cleanest by score

    def cut(order, budget=10):
        return [int(i) for i in order[:budget]]

    fam = CooperativeFamily(scores={"authenticity": score}, graph=neighbour_graph(X, "euclidean"),
                            selector_features=X, seed=0, budget_gate=lambda order, rho: cut(order, int(10 * rho)),
                            budget_cut=cut, token_orders={"herding": lambda subset: [int(i) for i in subset]})
    order, info = fam.select("authenticity", 1.3, "herding", 10, name="t")
    admissible = cut(list(np.argsort(-score, kind="stable")), 13)
    assert sorted(order) == list(range(43)) and info["admissible_gate"] == 13 and order[0] == 40
    assert info["duplicate_drops"] == sum(i in admissible for i in (0, 41, 42))
    assert info["admissible_size"] == 13 - info["duplicate_drops"] - info["outlier_drops"]
    assert info["refill"] == max(0, 10 - info["admissible_size"])


def test_learned_cleanliness_scores_clean_records_higher():
    rng = np.random.default_rng(6)
    centers = rng.normal(size=(4, 6)) * 3
    y = rng.integers(4, size=500)
    X = centers[y] + rng.normal(size=(500, 6))
    obs = y.copy()
    flipped = rng.choice(500, 150, replace=False)
    obs[flipped] = (y[flipped] + 1 + rng.integers(3, size=150)) % 4
    yc = rng.integers(4, size=150)
    Xc = centers[yc] + rng.normal(size=(150, 6))
    cfg = CleanlinessConfig()
    g, gc = neighbour_graph(X, "euclidean"), neighbour_graph(X, "euclidean", query=Xc)
    Fp, Fc, names = classification_features(g, gc, X, obs, Xc, yc, 0, cfg)
    assert Fp.shape == (500, 5) and Fc.shape == (150, 5) and names[0] == "knn_label_agreement"
    score, auc = learned_cleanliness(Fp, Fc, 0, cfg)
    from sklearn.metrics import roc_auc_score

    clean = np.ones(500, dtype=bool)
    clean[flipped] = False
    assert score.min() >= 0.0 and score.max() <= 1.0 and roc_auc_score(clean, score) > 0.85
    assert auc is not None and 0.5 < auc <= 1.0
    again, _ = learned_cleanliness(Fp, Fc, 0, cfg)
    assert np.array_equal(score, again)


def test_v22_preset_membership_and_track_grid():
    v2, v22 = OmniSelectConfig.preset("v2"), OmniSelectConfig.preset("v2_2")
    assert v22.cooperative.enabled and not v2.cooperative.enabled and v22.portfolio.membership == "unified_v22"
    same = {k: v for k, v in v22.to_dict().items() if k not in ("protocol", "cooperative", "portfolio")}
    assert same == {k: v for k, v in v2.to_dict().items() if k not in ("protocol", "cooperative", "portfolio")}
    assert OmniSelectConfig.from_dict(v22.to_dict()) == v22
    cfg = OmniSelectConfig.preset("v2_2", **parse_overrides(["cooperative.screen_keep=4", "cleanliness.folds=3",
                                                             "cooperative.outlier_veto=false"]))
    assert cfg.cooperative.screen_keep == 4 and cfg.cleanliness.folds == 3 and not cfg.cooperative.outlier_veto
    for bad in ({"cooperative.duplicate_ratio": 1.5}, {"cooperative.reference_rho": 0.5}, {"cleanliness.folds": 1}):
        with pytest.raises(ValueError):
            OmniSelectConfig.preset("v2_2", **bad)
    for track in TRACKS:
        assert "coop_herding" in reference_members(track, "unified_v22")
        assert "clean_top" in challenger_members(track, "unified_v22")
        assert "clean_top" not in reference_members(track, "unified_v22")
        for mode in ("unified", "unified_v21"):
            assert "coop_herding" not in reference_members(track, mode)
            assert "clean_top" not in challenger_members(track, mode)
    from tracks.common.experiment import load_track

    grid = load_track("process").config("tep21").cooperative_grid()
    assert len(grid) == 14 and grid[0] == ("authenticity", 1.15, "herding") and grid[-1] == (None, None, "kcenter")
    text = load_track("text").config("five_domain").cooperative_grid()
    assert text == [("authenticity", 1.3, "herding"), ("authenticity", 1.3, "kcenter"),
                    ("cleanliness", 1.3, "herding"), ("cleanliness", 1.3, "kcenter"), (None, None, "herding")]
    with pytest.raises(ValueError):
        load_track("process").config("tep21", coop_selectors=("random",)).cooperative_grid()


def test_v22_driver_cell(tmp_path):
    from tracks.common.experiment import load_track, resolve_configs, run_cell

    track = load_track("process")
    tcfg, ocfg = resolve_configs(track, "tep21", learner="mlp", seed=1, protocol="v2_2", smoke=True,
                                 track_overrides={"pool_n": 240, "val_n": 200, "test_n": 150, "mlp_max_iter": 20},
                                 omni_overrides={"precheck.enabled": "false"})
    run_cell(track, tcfg, ocfg, out_root=tmp_path, batch="v22", cli=["test"])
    cell = tmp_path / "v22" / "process" / "tep21" / "mlp" / "seed_1"
    clean = np.load(cell / "signals" / "cleanliness.npy")
    assert clean.shape == (240,) and clean.min() >= 0 and clean.max() <= 1
    timing = json.loads((cell / "signals" / "timings.json").read_text())["cleanliness"]
    assert timing["features"][0] == "knn_label_agreement" and timing["n_pool"] == 240 and timing["seconds"] >= 0
    decision = json.loads((cell / "decision.json").read_text())
    coop = decision["screening"]["cooperative"]
    assert coop["applied"] and coop["finalists"] == 3
    board = json.loads((cell / "leaderboard.json").read_text())
    rows = {r["name"]: r for r in board["rows"]}
    assert rows["coop_herding"]["role"] == "reference" and rows["clean_top"]["role"] == "challenger"
    finalists = [n for n, r in rows.items() if n.startswith("coop ") and r["stage"] == "finalist"]
    assert len(finalists) == 3 and all(rows[n]["role"] == "challenger" for n in finalists)
    names = set(rows) | set(board["screened_out"])
    assert sum(n.startswith("coop ") for n in names) + sum(
        len(json.loads((cell / "candidates" / board["candidate_dirs"][n] / "scores.json").read_text())
            ["scoring"].get("aliases", [])) for n in names if n.startswith("coop ")) == 14
    info = json.loads((cell / "candidates" / board["candidate_dirs"]["coop_herding"] / "scores.json").read_text())
    info = info["selection_info"]
    for key in ("rho", "selector", "admissible_gate", "admissible_size", "outlier_drops", "duplicate_drops",
                "refill", "seconds", "admissible_purity"):
        assert key in info
    assert info["rho"] == 1.3 and info["selector"] == "herding" and info["gate_score"] == "authenticity"
