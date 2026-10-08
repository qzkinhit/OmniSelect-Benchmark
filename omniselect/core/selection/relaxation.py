"""Relaxed quadratic selection: max s^T x - alpha x^T K x over x in [0, 1]^n with sum(x) = k.

K keeps the similarities of the ``knn`` nearest neighbours of each record (sparse, symmetrized,
negative values set to 0, zero diagonal), from L2-normalized features or from a Similarity object.
alpha = beta * mean(s) / (rho * mean row sum of K) with rho = k / n and s the min-max scaled
importance. Projected gradient ascent with step 1 / (2 alpha ||K||_2) runs ``iters`` steps from
x = rho, and the k largest coordinates are selected (ties by index). With beta = 0 the solution is
the top-k of s. This is the solver of the InfoMax baseline and of the v2.1 solver challengers.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def knn_matrix(n: int, block, knn: int = 10, chunk: int = 2048):
    """Sparse kNN similarity matrix from ``block(rows) -> (len(rows), n)`` similarities."""
    from scipy import sparse

    knn = max(1, min(int(knn), n - 1))
    rows, cols, vals = [], [], []
    for start in range(0, n, chunk):
        idx = np.arange(start, min(start + chunk, n))
        sims = np.array(block(idx), dtype=float)
        sims[np.arange(len(idx)), idx] = -np.inf
        nbr = np.argsort(-sims, axis=1, kind="stable")[:, :knn]
        rows.append(np.repeat(idx, knn))
        cols.append(nbr.reshape(-1))
        vals.append(np.take_along_axis(sims, nbr, axis=1).reshape(-1))
    K = sparse.csr_matrix((np.clip(np.concatenate(vals), 0.0, None), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n))
    K = K.maximum(K.T).tocsr()
    K.setdiag(0.0)
    K.eliminate_zeros()
    return K


def knn_similarity(features: np.ndarray, knn: int = 10, chunk: int = 2048):
    """Sparse kNN cosine similarity of the L2-normalized feature rows."""
    X = np.asarray(features, dtype=np.float64)
    X = X.reshape(X.shape[0], -1)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return knn_matrix(len(X), lambda idx: X[idx] @ X.T, knn, chunk)


def project_capped_simplex(v: np.ndarray, k: float, iters: int = 60) -> np.ndarray:
    """Euclidean projection onto {x in [0, 1]^n, sum(x) = k} by bisection on the shift."""
    lo, hi = float(v.min()) - 1.0, float(v.max())
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if np.clip(v - mid, 0.0, 1.0).sum() > k:
            lo = mid
        else:
            hi = mid
    return np.clip(v - 0.5 * (lo + hi), 0.0, 1.0)


def relaxed_select(importance: np.ndarray, k: int, K, *, beta: float = 1.0, iters: int = 300
                   ) -> tuple[list[int], np.ndarray]:
    """Solve the relaxation with kernel K. Returns (the k indices ordered by x, x)."""
    s = np.asarray(importance, dtype=float)
    s = (s - s.min()) / (np.ptp(s) + 1e-12)
    n = len(s)
    k = min(int(k), n)
    if float(beta) <= 0.0:
        order = [int(i) for i in np.argsort(-s, kind="stable")]
        x = np.zeros(n)
        x[order[:k]] = 1.0
        return order[:k], x
    rho = k / n
    alpha = float(beta) * max(float(s.mean()), 1e-12) / (rho * max(float(np.asarray(K.sum(axis=1)).mean()), 1e-12))
    v = np.random.default_rng(0).standard_normal(n)
    for _ in range(30):                                      # power iteration for ||K||_2
        v = K @ v
        v /= np.linalg.norm(v) + 1e-12
    lipschitz = 2.0 * alpha * max(float(v @ (K @ v)), 1e-12)
    x = np.full(n, rho)
    for _ in range(int(iters)):
        x = project_capped_simplex(x + (s - 2.0 * alpha * (K @ x)) / lipschitz, k)
    order = [int(i) for i in np.argsort(-x, kind="stable")]
    return order[:k], x


def infomax_select(importance: np.ndarray, features: np.ndarray, k: int, *, beta: float = 1.0, knn: int = 10,
                   iters: int = 300, similarity: Optional[object] = None) -> tuple[list[int], np.ndarray]:
    """Relaxed selection with the kNN kernel of ``similarity`` (a Similarity) or of the features."""
    n = len(importance)
    if float(beta) > 0.0:
        K = knn_similarity(features, knn) if similarity is None else knn_matrix(n, similarity.block, knn)
    else:
        K = None
    return relaxed_select(importance, k, K, beta=beta, iters=iters)
