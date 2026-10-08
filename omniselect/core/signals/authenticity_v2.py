"""Mechanism-matched authenticity: conjunctions of detector ranks, one detector per corruption mechanism.

``oof_label_agreement`` is the out-of-fold logistic-probe probability of the observed label.
``knn_label_agreement`` is the share of k nearest neighbours with the same label. ``knn_inlier`` is the
mean similarity to the k nearest neighbours. ``auth_label`` is min(rank(oof), rank(knn agreement))
and ``auth_full`` additionally takes the minimum with rank(inlier). kNN runs in row chunks.
"""
from __future__ import annotations

import numpy as np

__all__ = ["oof_label_agreement", "knn_label_agreement", "knn_inlier", "auth_label", "auth_full"]


def _rank01(v: np.ndarray) -> np.ndarray:
    r = np.argsort(np.argsort(v, kind="stable"), kind="stable")
    return r / (len(v) - 1 + 1e-9)


def oof_label_agreement(features: np.ndarray, labels: np.ndarray, *, folds: int = 5,
                        seed: int = 0, probe: str = "logreg") -> np.ndarray:
    """K-fold OUT-OF-FOLD predicted probability of each sample's observed label.

    The fold that scores a sample never saw it, so mislabelled samples cannot certify
    themselves (the failure of naive self-training refinement, which our probe showed makes
    things worse). This is the confident-learning counting principle applied as a score.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, StratifiedKFold

    X = np.asarray(features, float)
    y = np.asarray(labels)
    n = len(y)
    oof = np.zeros(n)
    try:
        splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, y))
    except Exception:
        splits = list(KFold(n_splits=folds, shuffle=True, random_state=seed).split(X))
    for tr, va in splits:
        clf = LogisticRegression(max_iter=200, C=1.0).fit(X[tr], y[tr])
        proba = clf.predict_proba(X[va])
        cls = {c: k for k, c in enumerate(clf.classes_)}
        oof[va] = np.array([proba[j, cls[y[i]]] if y[i] in cls else 0.0
                            for j, i in enumerate(va)])
    return oof


def knn_label_agreement(features: np.ndarray, labels: np.ndarray, *, knn: int = 15,
                        chunk_size: int = 2048) -> np.ndarray:
    """Fraction of a sample's k nearest neighbours sharing its observed label (v1 detector).

    Computed in row chunks so only a (chunk, n) similarity slab is ever materialised, never the
    full n x n matrix (at pool scale, n x n float32 is tens of GB and OOMs). Uses a per-chunk
    argsort, not argpartition: each row's label lookup reads specific neighbour INDICES (not just
    an order-invariant mean), so under similarity ties (the near-duplicate noise this
    detector targets) argpartition's unordered top-k can pick a different tied index
    than the original full-matrix argsort would have, changing the agreement score by up to 0.4 on
    duplicate-heavy pools (caught by adversarial re-verification). A per-row argsort depends only
    on that row's own similarities, not on which chunk it's grouped into, so this is bit-identical
    to the original unchunked `np.argsort(-S, axis=1)[:, :k]` row by row.
    """
    X = np.asarray(features, float)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    y = np.asarray(labels)
    n = len(y)
    k = min(knn, n)
    out = np.empty(n)
    for s0 in range(0, n, chunk_size):
        s1 = min(s0 + chunk_size, n)
        sims = X[s0:s1] @ X.T
        for r in range(s1 - s0):
            sims[r, s0 + r] = -np.inf
        idx = np.argsort(-sims, axis=1, kind="stable")[:, :k]
        for r in range(s1 - s0):
            out[s0 + r] = np.mean(y[idx[r]] == y[s0 + r])
    return out


def knn_inlier(features: np.ndarray, *, knn: int = 15, chunk_size: int = 2048) -> np.ndarray:
    """Mean cosine similarity to the k nearest neighbours (higher = inlier).

    Catches feature corruption (corrupted rows sit far from everything). Deliberately a
    separate arm: near-duplicates score HIGH here, so folding it in where duplication is the
    dominant mechanism promotes duplicates and hurts.

    Computed in row chunks (see knn_label_agreement) so only a (chunk, n) similarity slab is
    ever materialised, never the full n x n matrix.
    """
    X = np.asarray(features, float)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    n = X.shape[0]
    k = min(knn, n)
    out = np.empty(n)
    for s0 in range(0, n, chunk_size):
        s1 = min(s0 + chunk_size, n)
        sims = X[s0:s1] @ X.T
        for r in range(s1 - s0):
            sims[r, s0 + r] = -np.inf
        out[s0:s1] = np.sort(sims, axis=1)[:, -k:].mean(axis=1)
    return out


def auth_label(features: np.ndarray, labels: np.ndarray, *, knn: int = 15, folds: int = 5,
               seed: int = 0) -> np.ndarray:
    """Label-noise arm: min(rank(oof agreement), rank(kNN agreement)). Conjunctive."""
    oof = oof_label_agreement(features, labels, folds=folds, seed=seed)
    ka = knn_label_agreement(features, labels, knn=knn)
    return np.minimum(_rank01(oof), _rank01(ka))


def auth_full(features: np.ndarray, labels: np.ndarray, *, knn: int = 15, folds: int = 5,
              seed: int = 0) -> np.ndarray:
    """Label-noise arm AND corruption arm: additionally require inlierness. For modalities
    whose noise surface includes feature/sensor corruption (process, tabular, timeseries)."""
    base = auth_label(features, labels, knn=knn, folds=folds, seed=seed)
    inl = knn_inlier(features, knn=knn)
    return np.minimum(base, _rank01(inl))
