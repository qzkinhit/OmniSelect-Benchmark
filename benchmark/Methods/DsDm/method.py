"""DsDm (Engstrom et al., 2024): linear datamodel by subset regression, proxy-scale reduction.

``k_runs`` random subsets of half the pool are scored by the construction-split utility, a ridge
regression of the utility on centred inclusion indicators gives each record a coefficient, and
the k largest coefficients are kept. The subsets come from default_rng(seed).
"""
from __future__ import annotations

import numpy as np

from benchmark.Methods._common import top_k
from omniselect.core.portfolio.registry import SelectionContext, register


def dsdm_scores(train_eval_fn, n: int, *, k_runs: int = 24, subset_frac: float = 0.5,
                seed: int = 0, ridge: float = 1.0) -> np.ndarray:
    """Datamodel coefficients of the n records."""
    rng = np.random.default_rng(seed)
    m = int(round(subset_frac * n))
    Xind = np.zeros((k_runs, n))
    y = np.zeros(k_runs)
    for r in range(k_runs):
        idx = rng.permutation(n)[:m]
        Xind[r, idx] = 1.0
        y[r] = float(train_eval_fn([int(i) for i in idx]))
    Xc = Xind - Xind.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    A = Xc.T @ Xc + ridge * np.eye(n)
    return np.linalg.solve(A, Xc.T @ yc)


@register("dsdm", method_dir="DsDm", family="external", fidelity="proxy-scale reduction",
          requires=("construction_gain",), display="DsDm")
def dsdm_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """Top k datamodel coefficients with ``extras['dsdm_runs']`` subset fits."""
    w = dsdm_scores(ctx.construction_gain, ctx.n, k_runs=int(ctx.extras.get("dsdm_runs", 12)), seed=ctx.seed)
    return top_k(w, k)
