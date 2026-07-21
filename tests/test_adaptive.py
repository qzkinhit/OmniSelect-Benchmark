"""AdaptiveController is a portfolio meta-selector: it returns the validation-best
candidate, so it is >= every candidate (every baseline) by construction."""
import numpy as np

from mmdataselect.datatypes import Modality, UnifiedRecord
from mmdataselect.fusion.adaptive import AdaptiveController
from mmdataselect.signals import hashed_features


def _pool(n=60):
    return [UnifiedRecord(id=str(i), modality=Modality.TEXT, domain="x", text=f"sample number {i} text") for i in range(n)]


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
    # first-by-construction: chosen gain is the max over the whole portfolio
    assert abs(ctrl.leaderboard_[0][1] - ctrl.chosen_["val_gain"]) < 1e-9


def test_adapt_ge_every_extra_strategy():
    """Controller >= each candidate, including a strong extra strategy, by construction."""
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
    # the controller's chosen gain is >= the strong extra strategy's gain
    assert ctrl.chosen_["val_gain"] >= gain(best_extra) - 1e-9
