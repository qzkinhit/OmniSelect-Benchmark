"""D4 (Tirumala et al., NeurIPS 2023): SemDeDup, then drop the most prototypical records.

Stage 1 keeps max(k, (1 - dedup_frac) n) records by the SemDeDup rule. Stage 2 clusters the
survivors (about sqrt of their number) and keeps the k records farthest from their centroid.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from benchmark.Methods._common import as2d
from benchmark.Methods.SemDeDup.method import semdedup
from omniselect.core.portfolio.registry import SelectionContext, register


def d4(features: np.ndarray, k: int, *, seed: int = 0, dedup_frac: float = 0.25,
       n_clusters: Optional[int] = None) -> list:
    """D4 selection of k records."""
    from sklearn.cluster import KMeans
    f = as2d(features)
    n = len(f)
    keep = semdedup(f, max(k, int(round((1.0 - dedup_frac) * n))), seed=seed)
    keep = np.asarray(keep)
    if len(keep) <= k:
        return [int(i) for i in keep[:k]]
    g = f[keep]
    nc = n_clusters or max(2, int(np.sqrt(len(g))))
    km = KMeans(n_clusters=nc, n_init=3, random_state=seed).fit(g)
    d_cent = np.linalg.norm(g - km.cluster_centers_[km.labels_], axis=1)
    order = np.argsort(-d_cent, kind="stable")
    return [int(keep[i]) for i in order[:k]]


@register("d4", method_dir="D4", family="external", fidelity="local implementation", display="D4")
def d4_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """D4 on the track representation with the run seed."""
    return d4(ctx.features, k, seed=ctx.seed)
