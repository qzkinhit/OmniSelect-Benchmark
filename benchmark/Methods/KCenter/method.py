"""k-center greedy (Sener and Savarese, ICLR 2018): farthest-point traversal.

The first record is drawn from default_rng(seed). Each further step adds the record farthest
from the current selection. One-shot budgeted use of the active-learning coreset rule. In the
controller it is a portfolio-internal coverage candidate.
"""
from __future__ import annotations

import numpy as np

from benchmark.Methods._common import as2d
from omniselect.core.portfolio.registry import SelectionContext, register


def kcenter_greedy(features: np.ndarray, k: int, seed: int = 0) -> list:
    """k-center greedy / coreset (Sener & Savarese 2018). Farthest-point traversal: each step
    add the point farthest from the current selected set. Standard DeepCore baseline."""
    X = as2d(features)
    n = X.shape[0]
    k = min(k, n)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(n))
    chosen = [start]
    available = np.ones(n, dtype=bool)
    available[start] = False
    dist = np.linalg.norm(X - X[start][None, :], axis=1)
    for _ in range(1, k):
        i = int(np.argmax(np.where(available, dist, -np.inf)))
        chosen.append(i)
        available[i] = False
        dist = np.minimum(dist, np.linalg.norm(X - X[i][None, :], axis=1))
    return chosen


@register("kcenter", method_dir="KCenter", family="builtin", fidelity="local implementation",
          display="k-center")
def kcenter_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """k-center greedy on the track representation with the run seed (torch on a non-cpu selection device)."""
    from benchmark.Methods._torch_select import kcenter_torch, record, torch_device

    device = torch_device(ctx.extras.get("selection_device", "cpu"))
    record(ctx, "kcenter", device)
    if device is None:
        return kcenter_greedy(ctx.features, k, seed=ctx.seed)
    return kcenter_torch(ctx.features, k, seed=ctx.seed, device=device)
