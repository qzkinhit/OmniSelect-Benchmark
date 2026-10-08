"""SemDeDup (Abbas et al., 2023): within-cluster semantic deduplication.

k-means on unit-normalized embeddings (min(64, n // 20) clusters, at least 2). Inside a cluster,
records are visited from lowest to highest centroid similarity and a record whose cosine
similarity to a kept record exceeds 1 - eps is marked duplicate. Survivors are shuffled with
default_rng(seed). The removed duplicates fill any budget shortfall.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from benchmark.Methods._common import as2d
from omniselect.core.portfolio.registry import SelectionContext, register


def semdedup(features: np.ndarray, k: int, *, seed: int = 0, n_clusters: Optional[int] = None,
             eps: float = 0.05) -> list:
    """SemDeDup rule: survivors first, then duplicates, truncated to k."""
    from sklearn.cluster import KMeans
    X = as2d(features)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    n = X.shape[0]
    k = min(k, n)
    K = n_clusters or max(2, min(64, n // 20))
    km = KMeans(n_clusters=K, n_init=3, random_state=seed).fit(X)
    thr = 1.0 - eps
    keep, dropped = [], []
    for c in range(K):
        mem = np.where(km.labels_ == c)[0]
        if not len(mem):
            continue
        cen = km.cluster_centers_[c]
        cen = cen / (np.linalg.norm(cen) + 1e-12)
        order = mem[np.argsort(X[mem] @ cen, kind="stable")]
        kept_c = []
        for i in order:
            sims = X[kept_c] @ X[i] if kept_c else np.array([])
            if len(sims) and sims.max() > thr:
                dropped.append(int(i))
            else:
                kept_c.append(int(i))
        keep.extend(kept_c)
    rng = np.random.default_rng(seed)
    keep = list(rng.permutation(np.array(keep, dtype=int)))
    if len(keep) < k:
        keep.extend(dropped[: k - len(keep)])
    return [int(i) for i in keep[:k]]


@register("semdedup", method_dir="SemDeDup", family="builtin", fidelity="local implementation",
          display="SemDeDup")
def semdedup_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """SemDeDup on the track representation with the run seed."""
    return semdedup(ctx.features, k, seed=ctx.seed)
