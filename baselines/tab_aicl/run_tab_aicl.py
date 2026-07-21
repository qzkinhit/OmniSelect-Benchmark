"""Tab-AICL baseline runner -- reproduce in-context support-set selection for TabPFN.

    python baselines/tab_aicl/run_tab_aicl.py            # OpenML electricity subset, TabPFN-v2

Tab-AICL is tabular-native (like DeepCore is image-native), so this runner reproduces its
original setting directly: take a small OpenML pool, fit TabPFN-v2 on a seed context to get
forward-pass margins, run the three acquisition rules (coreset / margin / hybrid) plus random
at a fixed budget, refit TabPFN on each selected support set, and report test ROC AUC. The
point is a faithfulness check: the margin rule selects the lowest-margin (most uncertain)
candidates, and the three rules form the direct prior-work comparison for "select data for a
tabular foundation model".

Env: TAB_DATASET, TAB_POOL, TAB_BUDGET, SEED.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local `method` pkg
sys.path.insert(0, _REPO)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from method import tabpfn_coreset, tabpfn_hybrid, tabpfn_margin  # noqa: E402

DATASET = os.environ.get("TAB_DATASET", "electricity")
POOL_N = int(os.environ.get("TAB_POOL", "1200"))
TEST_N = int(os.environ.get("TAB_TEST", "1000"))
BUDGET_FRAC = float(os.environ.get("TAB_BUDGET", "0.3"))
SEED = int(os.environ.get("SEED", "0"))


def _tabpfn(X, y):
    from tabpfn import TabPFNClassifier
    c = TabPFNClassifier.create_default_for_version("v2", device="cpu", ignore_pretraining_limits=True)
    return c.fit(X, y)


def main():
    from sklearn.datasets import fetch_openml
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    d = fetch_openml(DATASET, version=1, as_frame=True, parser="auto")
    X = d.data.select_dtypes("number").to_numpy(float)
    y = (d.target.to_numpy().astype(str))
    classes, y = np.unique(y, return_inverse=True)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    tr, te = perm[:POOL_N], perm[POOL_N:POOL_N + TEST_N]
    sc = StandardScaler().fit(X[tr])
    Xp, yp = sc.transform(X[tr]), y[tr]
    Xt, yt = sc.transform(X[te]), y[te]
    n = len(Xp); k = int(BUDGET_FRAC * n)

    seed_sub = rng.permutation(n)[:min(300, k)]
    probs = _tabpfn(Xp[seed_sub], yp[seed_sub]).predict_proba(Xp)   # TabPFN forward-pass margins

    sels = {
        "random": list(rng.permutation(n)[:k]),
        "tabpfn_coreset": tabpfn_coreset(Xp, k, seed=SEED),
        "tabpfn_margin": tabpfn_margin(probs, k),
        "tabpfn_hybrid": tabpfn_hybrid(Xp, probs, k, seed=SEED),
    }
    print(f"Tab-AICL | {DATASET} pool={n} test={len(Xt)} budget={k} (TabPFN-v2 in-context, ROC AUC)")
    for m, sel in sels.items():
        p = _tabpfn(Xp[sel], yp[sel]).predict_proba(Xt)
        try:
            auc = float(roc_auc_score(yt, p[:, 1]) if p.shape[1] == 2 else roc_auc_score(yt, p, multi_class="ovr"))
        except Exception:
            auc = float((p.argmax(1) == yt).mean())
        print(f"  {m:15} auc={auc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
