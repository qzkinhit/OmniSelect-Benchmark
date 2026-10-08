"""Analytic cases for the gates, including the K / delta threshold of the statistical gates."""
from __future__ import annotations

import math

import numpy as np
import pytest

from omniselect.core.gates import (
    argmax_gate,
    bootstrap_gate,
    eprocess_gate,
    hoeffding_radius,
    lcb_gate,
    margin_gate,
    run_family,
)
from omniselect.core.gates.surrogate import (
    auc_units,
    balanced_units,
    forecast_units,
    paired_differences,
    text_units,
)


def test_margin_gate_uses_relative_margin():
    assert not margin_gate("r", 0.40, "c", 0.405, 0.015).adopted          # 0.405 <= 0.406
    result = margin_gate("r", 0.40, "c", 0.4061, 0.015)
    assert result.adopted and math.isclose(result.statistics["tau"], 0.006)
    assert margin_gate("r", -1.0, "c", -0.98, 0.015).adopted              # |u| for negative utilities


def test_argmax_gate_is_strict():
    assert not argmax_gate("r", 0.5, "c", 0.5).adopted
    assert argmax_gate("r", 0.5, "c", 0.5001).adopted


def test_hoeffding_radius_closed_form():
    w = np.full(400, 1 / 400)
    expected = math.sqrt(2 * math.log(3 / 0.05) / 400)
    assert math.isclose(hoeffding_radius(w, 0.05, 3), expected, rel_tol=1e-12)


def test_lcb_gate_adopts_exactly_when_mean_exceeds_radius():
    n = 400
    w = np.full(n, 1 / n)
    radius = hoeffding_radius(w, 0.05, 1)
    d_pass = np.full(n, radius + 1e-6)
    d_fail = np.full(n, radius - 1e-6)
    assert lcb_gate("r", "c", d_pass, w, 0.05, 1).adopted
    assert not lcb_gate("r", "c", d_fail, w, 0.05, 1).adopted
    # the same difference fails once the family grows (K / delta threshold)
    k = 10
    assert not lcb_gate("r", "c", d_pass, w, 0.05, k).adopted
    assert lcb_gate("r", "c", np.full(n, hoeffding_radius(w, 0.05, k) + 1e-6), w, 0.05, k).adopted


def test_lcb_gate_rejects_unbounded_differences():
    with pytest.raises(ValueError):
        lcb_gate("r", "c", np.array([2.0, 0.0]), np.array([0.5, 0.5]), 0.05)


def test_run_family_adopts_first_passing_challenger_in_order():
    passes = {"a": False, "b": True, "c": True}
    seen = []

    def test_one(name, k):
        seen.append((name, k))
        return lcb_gate("ref", name, np.full(4, 1.0 if passes[name] else -1.0), np.full(4, 0.25), 0.9, 1)

    decision = run_family("lcb", "ref", ["a", "b", "c"], test_one)
    assert decision.elected == "b" and decision.adopted
    assert seen == [("a", 3), ("b", 3)]
    empty = run_family("lcb", "ref", [], test_one)
    assert empty.elected == "ref" and not empty.adopted


def test_bootstrap_gate_threshold_depends_on_k():
    n = 200
    w = np.full(n, 1 / n)
    strong = np.full(n, 0.5)
    result = bootstrap_gate("r", "c", strong, w, n_boot=1000, p_beat_min=0.9, k=3, seed=0)
    assert result.adopted and result.statistics["p_win"] == 1.0
    assert math.isclose(result.statistics["threshold"], 1 - 0.1 / 3)
    assert not bootstrap_gate("r", "c", -strong, w, n_boot=1000, p_beat_min=0.9, k=3, seed=0).adopted


def test_bootstrap_on_a_shifted_sample_and_bonferroni():
    n = 400
    w = np.full(n, 1 / n)
    base = np.random.default_rng(3).normal(0.0, 0.3, n)
    base -= base.mean()
    shift = 1.5 * base.std() / math.sqrt(n)             # p_win near Phi(1.5) = 0.933
    d = np.clip(base + shift, -1, 1)
    one = bootstrap_gate("r", "c", d, w, k=1, seed=0)
    three = bootstrap_gate("r", "c", d, w, k=3, seed=0)
    assert 0.9 <= one.statistics["p_win"] < 1 - 0.1 / 3
    assert one.adopted and not three.adopted
    assert bootstrap_gate("r", "c", np.clip(base + 0.1, -1, 1), w, k=3, seed=0).statistics["p_win"] == 1.0
    assert not bootstrap_gate("r", "c", np.clip(base - 0.1, -1, 1), w, k=1, seed=0).adopted
    # same seed, same resamples
    assert bootstrap_gate("r", "c", d, w, k=1, seed=0).statistics == one.statistics


def test_bootstrap_respects_unit_weights():
    d = np.array([1.0, -1.0, -1.0, -1.0])
    heavy = np.array([0.91, 0.03, 0.03, 0.03])
    assert bootstrap_gate("r", "c", d, heavy, k=1, seed=0).statistics["mean"] > 0.8
    assert not bootstrap_gate("r", "c", d, np.full(4, 0.25), k=1, seed=0).adopted


@pytest.mark.parametrize("k", [1, 3, 10])
def test_zero_differences_never_adopt(k):
    n = 500
    w = np.full(n, 1 / n)
    ep = eprocess_gate("r", "c", np.zeros(n), w, delta=0.05, eps=0.0, k=k, seed=0)
    assert not ep.adopted and ep.statistics["e_value"] == 1.0 and ep.statistics["stop_index"] is None
    assert not bootstrap_gate("r", "c", np.zeros(n), w, k=k, seed=0).adopted
    # eps > 0 makes the wealth grow on zero differences, the positive-mean condition still refuses
    shifted = eprocess_gate("r", "c", np.zeros(n), w, delta=0.05, eps=0.1, k=k, seed=0)
    assert shifted.statistics["e_value"] > k / 0.05 and not shifted.adopted


def test_eprocess_constant_gain_stops_after_a_handful_of_units():
    # lambda_1 = 0, then lambda_t = 1/2, so W_t = 1.5^(t-1): 1.5^8 = 25.6 >= 20 > 1.5^7
    n = 10
    result = eprocess_gate("r", "c", np.ones(n), np.full(n, 1 / n), delta=0.05, k=1, seed=0)
    assert result.adopted and result.statistics["stop_index"] == 9
    assert math.isclose(result.statistics["e_value"], 1.5 ** 9)
    assert math.isclose(result.statistics["max_stake"], 0.5)


def test_eprocess_uses_k_over_delta():
    n = 10
    w = np.full(n, 1 / n)
    one = eprocess_gate("r", "c", np.ones(n), w, delta=0.05, k=1, seed=0)
    three = eprocess_gate("r", "c", np.ones(n), w, delta=0.05, k=3, seed=0)
    assert math.isclose(three.statistics["threshold"], 60.0)
    assert one.adopted and not three.adopted                   # 20 <= 1.5^9 = 38.4 < 60
    longer = eprocess_gate("r", "c", np.ones(40), np.full(40, 1 / 40), delta=0.05, k=3, seed=0)
    assert longer.adopted and longer.statistics["stop_index"] == 12   # 1.5^11 = 86.5 >= 60 > 1.5^10


def test_eprocess_stakes_are_predictable():
    from omniselect.core.gates.eprocess import agrapa_wealth

    y = np.array([0.5, -0.2, 0.9, 0.1])
    log_wealth, stakes = agrapa_wealth(y)
    assert stakes[0] == 0.0
    mu, second = y[:2].mean(), (y[:2] ** 2).mean()
    assert math.isclose(stakes[2], min(0.5, max(0.0, mu / second)))
    changed = y.copy()
    changed[3] = -0.9                                          # the last value cannot change earlier stakes
    assert np.array_equal(agrapa_wealth(changed)[1], stakes)


def test_eprocess_crossing_rate_under_the_null_is_below_delta():
    rng = np.random.default_rng(1)
    crossings = 0
    for s in range(400):
        d = rng.choice([-0.5, 0.5], 200)
        crossings += eprocess_gate("r", "c", d, np.full(200, 1 / 200), delta=0.05, k=1, seed=s).statistics[
            "log_e_value"] >= math.log(20)
    assert crossings / 400 <= 0.05


def _classification_units(target, prediction):
    return balanced_units({"target": np.asarray(target), "prediction": np.asarray(prediction)})


def test_class_stratified_reading_draws_classes_uniformly():
    from omniselect.core.gates.paired import paired_sample

    rng = np.random.default_rng(0)
    target = np.repeat([0, 1, 2, 3], [200, 50, 30, 20])            # imbalanced classes
    ref = _classification_units(target, np.where(rng.random(300) < 0.5, target, -1))
    good = np.where(target == 3, target, np.where(rng.random(300) < 0.5, target, -1))   # class 3 always right
    sample = paired_sample(ref, _classification_units(target, good))
    assert sample.design("stratified") == "class" and sample.design("uniform") == "uniform"
    seq = sample.sequence("stratified", seed=1)
    assert 0 < len(seq) <= 300 and np.all(np.abs(seq) <= 1)
    assert np.array_equal(seq, sample.sequence("stratified", seed=1))
    # the sequence mean estimates the balanced (class-averaged) gain, not the unit average
    means = [sample.sequence("stratified", seed=s).mean() for s in range(200)]
    assert abs(np.mean(means) - sample.mean()) < 0.03
    boot = sample.bootstrap_means("stratified", 2000, seed=0)
    assert abs(boot.mean() - sample.mean()) < 0.01
    assert len(sample.sequence("uniform", seed=1)) == 300


def test_auc_pair_reading_uses_kernel_differences():
    from omniselect.core.gates.paired import paired_sample

    rng = np.random.default_rng(2)
    y = np.array([1] * 120 + [0] * 200)
    s_ref = np.clip(0.2 * y + rng.random(320) * 0.8, 0, 1)
    s_ch = np.clip(0.4 * y + rng.random(320) * 0.6, 0, 1)
    classes = np.array([0, 1])

    def units(score):
        return auc_units({"target": y, "proba": np.stack([1 - score, score], 1), "classes": classes})

    sample = paired_sample(units(s_ref), units(s_ch))
    assert sample.design("stratified") == "auc_pairs"
    seq = sample.sequence("stratified", seed=0)
    assert len(seq) == 120 and set(np.unique(seq)) <= {-1.0, -0.5, 0.0, 0.5, 1.0}
    from sklearn.metrics import roc_auc_score

    gain = roc_auc_score(y, s_ch) - roc_auc_score(y, s_ref)
    assert math.isclose(sample.mean(), gain, rel_tol=1e-9)
    assert abs(np.mean([sample.sequence("stratified", seed=s).mean() for s in range(300)]) - gain) < 0.02
    assert abs(sample.bootstrap_means("stratified", 500, seed=0).mean() - gain) < 0.02


def test_domain_reading_for_text_and_audit_entries():
    from omniselect.core.gates.audit import audit_family, test_paired
    from omniselect.core.gates.paired import paired_sample

    rng = np.random.default_rng(3)
    domain = np.repeat(np.array(["a", "b", "c"]), [300, 60, 40])
    tokens = rng.integers(20, 200, 400)
    ref = text_units({"nll": np.full(400, 2.0), "n_tokens": tokens, "domain": domain})
    better = text_units({"nll": np.full(400, 2.0) - 0.3 * (domain != "a"), "n_tokens": tokens, "domain": domain})
    sample = paired_sample(ref, better, clip=1.0)
    assert sample.design("stratified") == "domain"
    seq = sample.sequence("stratified", seed=0)
    assert set(np.round(np.unique(seq), 6)) <= {0.0, 0.3}
    ep = test_paired("eprocess", "r", "c", sample, k=3, seed=0, delta=0.05, eps=0.0, n_boot=200, p_beat_min=0.9,
                     order="stratified")
    assert ep.statistics["reading"] == "domain" and ep.adopted and ep.statistics["stop_index"] is not None
    bs = test_paired("bootstrap", "r", "c", sample, k=3, seed=0, delta=0.05, eps=0.0, n_boot=200, p_beat_min=0.9,
                     order="stratified")
    assert bs.statistics["resampling"] == "domain" and bs.adopted
    zero = paired_sample(ref, ref, clip=1.0)
    audit = audit_family("r", [("c", sample), ("z", zero)], k=3, seed=0, delta=0.05, eps=0.0, n_boot=200,
                         order="stratified", split="conf")
    first, second = audit["entries"]
    assert first["certified"] and first["boot_interval90"][0] > 0 and math.isclose(first["mean_gain"], 0.2)
    assert not second["certified"] and second["stop_index"] is None and second["mean_gain"] == 0.0
    assert audit["K"] == 3 and audit["reading_order"] == "stratified"


def test_run_family_test_all_records_every_challenger():
    outcome = {"a": False, "b": True, "c": True}

    def test_one(name, k):
        d = np.ones(40) if outcome[name] else np.zeros(40)
        return eprocess_gate("ref", name, d, np.full(40, 1 / 40), k=k)

    decision = run_family("eprocess", "ref", ["a", "b", "c"], test_one, test_all=True)
    assert decision.elected == "b" and decision.k_tested == 3
    assert [t.challenger for t in decision.tests] == ["a", "b", "c"]
    assert [t.adopted for t in decision.tests] == [False, True, True]


def test_surrogates_reproduce_their_metrics():
    rng = np.random.default_rng(0)
    target = rng.integers(0, 3, 300)
    pred = np.where(rng.random(300) < 0.7, target, rng.integers(0, 3, 300))
    bal = balanced_units({"target": target, "prediction": pred})
    expected = np.mean([(pred[target == c] == c).mean() for c in range(3)])
    assert math.isclose(bal.mean(), expected, rel_tol=1e-12)

    from sklearn.metrics import roc_auc_score

    y = rng.integers(0, 2, 200)
    p1 = np.clip(0.3 * y + rng.random(200) * 0.7, 0, 1)
    proba = np.stack([1 - p1, p1], axis=1)
    au = auc_units({"target": y, "proba": proba, "classes": np.array([0, 1])})
    assert math.isclose(au.mean(), roc_auc_score(y, p1), rel_tol=1e-12)


def test_forecast_blocks_and_text_weights():
    rng = np.random.default_rng(1)
    target = rng.normal(size=(40, 4))
    pred = target + 0.1
    units = forecast_units({"target": target, "prediction": pred, "start": np.arange(40) * 3}, s0=1.0,
                           block_steps=30)
    assert len(units.values) == 4 and math.isclose(units.weights.sum(), 1.0)
    tu = text_units({"nll": np.array([1.0, 2.0, 3.0]), "n_tokens": np.array([1, 3, 2]),
                     "domain": np.array(["a", "a", "b"])})
    assert np.allclose(tu.weights, [0.125, 0.375, 0.5])


def test_paired_differences_clip_and_alignment():
    a = text_units({"nll": np.array([1.0, 1.0]), "n_tokens": np.array([1, 1]), "domain": np.array(["x", "y"])})
    b = text_units({"nll": np.array([0.0, 4.0]), "n_tokens": np.array([1, 1]), "domain": np.array(["x", "y"])})
    d, w = paired_differences(a, b, clip=1.0)
    assert np.allclose(d, [1.0, -1.0]) and np.allclose(w, [0.5, 0.5])


def test_margin_family_records_top_k_when_the_reference_leads():
    from omniselect.config.config import OmniSelectConfig
    from omniselect.core.adjudication.controller import CandidateRecord, _decide

    cands = [CandidateRecord("ref", [0], True, "reference", u_rank=0.50),
             CandidateRecord("other_ref", [1], True, "reference", u_rank=0.40),
             CandidateRecord("a", [2], False, "finalist", u_rank=0.49),
             CandidateRecord("b", [3], False, "finalist", u_rank=0.48),
             CandidateRecord("c", [4], False, "finalist", u_rank=0.47),
             CandidateRecord("d", [5], False, "finalist", u_rank=0.46)]
    cfg = OmniSelectConfig.preset("v2")
    decision = _decide(cfg, cands, cands[0], cands[0], None, 0)
    assert decision.elected == "ref" and not decision.adopted and decision.k_tested == 3
    assert [t.challenger for t in decision.tests] == ["a", "b", "c"]
    assert all(not t.adopted and t.statistics["difference"] < 0 for t in decision.tests)
    cands[3].u_rank = 0.60                      # b now leads by more than the margin
    decision = _decide(cfg, cands, cands[0], cands[3], None, 0)
    assert decision.elected == "b" and decision.adopted and [t.challenger for t in decision.tests][0] == "b"
