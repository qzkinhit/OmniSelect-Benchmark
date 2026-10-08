"""Density sampling (Sachdeva et al., 2024): inverse-density sampling without replacement.

Local sparsity is the mean distance to the 10 nearest neighbours. Sampling logits are its log,
and Gumbel keys from default_rng(seed) turn the weights into a top-k without replacement. The
kNN estimator replaces the source method's LSH density estimator. The canonical vision,
forecasting, TEP and tabular rows used seed 0 on every run seed (``extras['density_seed']``).
"""
from __future__ import annotations

import numpy as np

from benchmark.Methods._common import as2d
from omniselect.core.portfolio.registry import SelectionContext, register


def density_select(features: np.ndarray, k: int, *, knn: int = 10, seed: int = 0) -> list:
    """Inverse-density Gumbel top-k."""
    X = as2d(features)
    n = X.shape[0]
    k = min(k, n)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(knn + 1, n)).fit(X)
    dist, _ = nn.kneighbors(X)
    sparsity = dist[:, 1:].mean(axis=1)
    logits = np.log(sparsity + 1e-12)
    rng = np.random.default_rng(seed)
    g = rng.gumbel(size=n)
    return [int(i) for i in np.argsort(-(logits + g), kind="stable")[:k]]


@register("density", method_dir="Density", family="external", fidelity="local implementation",
          display="Density")
def density_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """Density on the track representation. Seed 0 unless extras['density_seed'] == 'run_seed'."""
    seed = ctx.seed if ctx.extras.get("density_seed") == "run_seed" else 0
    return density_select(ctx.features, k, seed=seed)
