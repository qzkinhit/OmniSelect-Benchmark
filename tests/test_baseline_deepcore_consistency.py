"""The standalone DeepCore baseline and the integrated portfolio copy must agree.

`baselines/deepcore/method/coreset_select.py` is the self-contained reproduction (kept
independent per the baselines/ convention). `src/mmdataselect/selectors/external_baselines.py`
is the copy wired into the AdaptiveController portfolio + the cross-modal runners. They must
return identical selections so the reported numbers and the baseline directory never drift.
"""
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)

from baselines.deepcore.method import coreset_select as B  # noqa: E402
from mmdataselect.selectors import external_baselines as S  # noqa: E402


def _data(seed=0):
    rng = np.random.default_rng(seed)
    n, d, c = 240, 32, 8
    X = rng.normal(size=(n, d))
    probs = rng.dirichlet(np.ones(c), size=n)
    y = rng.integers(0, c, n)
    return X, probs, y


def test_geometric_match():
    X, _, _ = _data()
    for k in (30, 60, 120):
        assert B.herding(X, k) == S.herding(X, k)
        assert B.kcenter_greedy(X, k, seed=1) == S.kcenter_greedy(X, k, seed=1)


def test_score_based_match():
    X, probs, y = _data(3)
    for k in (30, 60, 120):
        assert B.el2n(probs, y, k, is_logits=False) == S.el2n(probs, y, k, is_logits=False)
        assert B.grand(probs, y, X, k, is_logits=False) == S.grand(probs, y, X, k, is_logits=False)


def test_dispatch():
    X, probs, y = _data(5)
    assert B.select("herding", 40, features=X) == B.herding(X, 40)
    assert B.select("el2n", 40, probs=probs, labels=y) == B.el2n(probs, y, 40, is_logits=False)
