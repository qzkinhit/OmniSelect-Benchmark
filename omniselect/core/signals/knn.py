"""Chunked k-nearest-neighbour signals on a feature matrix: label agreement and novelty.

For each row the k most similar other rows (dot product, the diagonal set to ``self_value``)
are found with argpartition in row chunks, so only a (chunk, n) slab is materialized.
Authenticity is the fraction of neighbours sharing the row's observed label. Redundancy is
1 - mean similarity to the neighbours. This is the rule of the 2026-07 vision, TEP and
tabular runners (self_value -1.0) and of the native protocol (self_value -2.0).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def knn_agreement_and_novelty(
    features: np.ndarray,
    labels: Optional[np.ndarray],
    k: int,
    *,
    chunk: int = 2048,
    self_value: float = -1.0,
    dtype: type = np.float64,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """Return (label agreement or None when ``labels`` is None, novelty) per row."""
    X = np.asarray(features)
    n = X.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k must be in [1, {n - 1}], got {k}")
    auth = np.zeros(n, dtype=dtype) if labels is not None else None
    novelty = np.zeros(n, dtype=dtype)
    for s0 in range(0, n, chunk):
        sims = X[s0:s0 + chunk] @ X.T
        for r in range(sims.shape[0]):
            sims[r, s0 + r] = self_value
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        rows = np.arange(sims.shape[0])[:, None]
        if auth is not None:
            auth[s0:s0 + chunk] = (labels[idx] == labels[s0:s0 + chunk, None]).mean(axis=1)
        novelty[s0:s0 + chunk] = 1.0 - sims[rows, idx].mean(axis=1)
    return auth, novelty


def l2_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Rows divided by their L2 norm plus ``eps`` (the runners' normalization)."""
    X = np.asarray(X)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
