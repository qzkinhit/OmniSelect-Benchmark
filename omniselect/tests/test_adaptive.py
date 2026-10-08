"""AdaptiveController compares an explicit finite portfolio on validation."""
import numpy as np
import pytest

from omniselect.core.datatypes import Modality, UnifiedRecord
from omniselect.core.adjudication.controller import AdaptiveController
from omniselect.core.signals import hashed_features


def _pool(n=60):
    return [UnifiedRecord(id=str(i), modality=Modality.TEXT, domain="x", text=f"sample number {i} text")
            for i in range(n)]


def test_adapt_recovers_rewarded_channel():
    n = 60
    recs = _pool(n)
    rng = np.random.default_rng(0)
    ch_good = rng.random(n)          # channel 0 = the "useful" signal
    scores = np.stack([ch_good, rng.random(n), rng.random(n)], axis=0)
    feats = hashed_features(recs, dim=64)

    def gain(sel):                   # held-out gain rewards high-ch_good selections
        return float(ch_good[sel].mean())

    ctrl = AdaptiveController(lam_grid=(0.0, 0.25), prefilter_grid=(0.0,))
    sel = ctrl.select(recs, scores, 20, features=feats, held_out_gain=gain)
    assert len(sel) == 20 and len(set(sel)) == 20
    # validation-best selection concentrates on the rewarded channel
    assert ch_good[sel].mean() > ch_good.mean()
    # With no switch margin against an external reference here, the chosen
    # validation gain equals the largest evaluated validation gain.
    assert abs(ctrl.leaderboard_[0][1] - ctrl.chosen_["val_gain"]) < 1e-9


def test_adapt_ge_every_extra_strategy_on_the_supplied_validation_callback():
    n = 50
    recs = _pool(n)
    rng = np.random.default_rng(1)
    scores = rng.random((3, n))
    feats = hashed_features(recs, dim=64)
    target = rng.random(n)            # the true objective the gain measures
    best_extra = list(np.argsort(-target)[:25])   # an oracle-ish strong baseline

    def gain(sel):
        return float(target[sel].mean())

    ctrl = AdaptiveController(prefilter_grid=(0.0,))
    sel = ctrl.select(recs, scores, 25, features=feats, held_out_gain=gain,
                      extra_strategies=[("oracle", lambda k: best_extra)])
    assert len(sel) == 25
    # This statement is scoped to the validation callback, not unseen test data.
    assert ctrl.chosen_["val_gain"] >= gain(best_extra) - 1e-9


def test_adapt_rejects_an_invalid_reference_before_scoring():
    n = 20
    recs = _pool(n)
    rng = np.random.default_rng(2)
    scores = rng.random((3, n))
    feats = hashed_features(recs, dim=16)
    ctrl = AdaptiveController(prefilter_grid=(0.0,))

    with pytest.raises(ValueError, match="duplicate selection ID"):
        ctrl.select(
            recs,
            scores,
            5,
            features=feats,
            held_out_gain=lambda selected: float(len(selected)),
            extra_strategies=[("invalid", lambda k: [0] * k)],
        )


def test_optional_construction_failures_are_recorded(monkeypatch):
    n = 30
    recs = _pool(n)
    rng = np.random.default_rng(3)
    scores = rng.random((3, n))
    feats = hashed_features(recs, dim=16)
    target = rng.random(n)
    random_selection = list(range(10))

    import omniselect.core.adjudication.controller as controller_module

    def broken_ascent(*args, **kwargs):
        raise ValueError("injected construction failure")

    monkeypatch.setattr(controller_module, "coordinate_ascent", broken_ascent)
    ctrl = AdaptiveController(prefilter_grid=(0.0,), lam_grid=(0.0,))
    selected = ctrl.select(
        recs,
        scores,
        10,
        features=feats,
        held_out_gain=lambda rows: float(target[rows].mean()),
        construct_gain=lambda rows: float(target[rows].mean()),
        extra_strategies=[("random", lambda k: random_selection)],
    )

    assert len(selected) == 10
    assert ctrl.chosen_["construction_error_count"] == 1
    assert ctrl.construction_errors_[0]["stage"] == "learned_weight_fusion"
    assert ctrl.construction_errors_[0]["error_type"] == "ValueError"


def test_adapt_rejects_malformed_or_nonfinite_inputs():
    records = _pool(8)
    features = hashed_features(records, dim=8)
    controller = AdaptiveController(prefilter_grid=(0.0,))

    with pytest.raises(ValueError, match="scores must have shape"):
        controller.select(
            records,
            np.ones((3, 7)),
            3,
            features=features,
            held_out_gain=lambda rows: 0.0,
        )
    bad_scores = np.ones((3, 8))
    bad_scores[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite-valued"):
        controller.select(
            records,
            bad_scores,
            3,
            features=features,
            held_out_gain=lambda rows: 0.0,
        )
    with pytest.raises(ValueError, match="one row per record"):
        controller.select(
            records,
            np.ones((3, 8)),
            3,
            features=np.ones((7, 4)),
            held_out_gain=lambda rows: 0.0,
        )


def test_adapt_rejects_nonfinite_validation_gain():
    records = _pool(8)
    controller = AdaptiveController(prefilter_grid=(0.0,))
    with pytest.raises(ValueError, match="held_out_gain"):
        controller.select(
            records,
            np.ones((3, 8)),
            3,
            features=hashed_features(records, dim=8),
            held_out_gain=lambda rows: float("nan"),
        )
