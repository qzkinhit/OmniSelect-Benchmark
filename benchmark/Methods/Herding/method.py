"""Herding (Welling, ICML 2009): greedily match the running selected mean to the pool mean.

At step t the unchosen record i minimizing ||(sum of selected + x_i) / t - mean|| is added.
Local implementation on the track representation (frozen CLIP embeddings, normalized windows,
standardized rows, or ResNet penultimate features).
"""
from __future__ import annotations

import numpy as np

from benchmark.Methods._common import as2d
from omniselect.core.portfolio.registry import SelectionContext, register


def herding(features: np.ndarray, k: int) -> list:
    """Herding coreset (Welling 2009). Greedily pick the point that pulls the running
    selected-mean closest to the full-set mean in feature space. Standard DeepCore baseline."""
    X = as2d(features)
    n = X.shape[0]
    k = min(k, n)
    target = X.mean(axis=0)
    chosen: list = []
    running = np.zeros_like(target)
    mask = np.ones(n, dtype=bool)
    for t in range(k):
        cand = (running[None, :] + X) / (t + 1)
        d = np.linalg.norm(cand - target[None, :], axis=1)
        d[~mask] = np.inf
        i = int(np.argmin(d))
        chosen.append(i)
        running = running + X[i]
        mask[i] = False
    return chosen


@register("herding", method_dir="Herding", family="external", fidelity="local implementation",
          display="Herding")
def herding_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """Herding on the track representation (torch on ``extras['selection_device']`` unless it is cpu)."""
    from benchmark.Methods._torch_select import herding_torch, record, torch_device

    device = torch_device(ctx.extras.get("selection_device", "cpu"))
    record(ctx, "herding", device)
    return herding(ctx.features, k) if device is None else herding_torch(ctx.features, k, device)
