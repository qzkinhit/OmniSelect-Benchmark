"""InfoMax (Tan et al., ICLR 2025): importance minus pairwise redundancy, relaxed and solved by projected gradient.

Written from the objective in the paper, without the authors' code (their repository has no
license). Maximize f(x) = s^T x - alpha x^T K x over x in [0, 1]^n with sum(x) = k, where s is the
min-max scaled importance channel and K is the cosine similarity of L2-normalized features kept
for the ``knn`` nearest neighbours of each record (sparse, symmetrized, negative values set to 0,
zero diagonal). alpha = beta * mean(s) / (rho * mean row sum of K) with rho = k / n, so beta = 1 gives both
terms the same size at x = rho. Projected gradient ascent with step 1 / (2 alpha ||K||_2) runs
``iters`` steps from x = rho, and the k largest coordinates are selected (ties by pool index).
"""
from __future__ import annotations

import numpy as np

from benchmark.Methods._common import as2d
from omniselect.core.portfolio.registry import SelectionContext, register
from omniselect.core.selection.relaxation import (  # noqa: F401
    infomax_select,
    knn_similarity,
    project_capped_simplex,
)


def infomax(importance: np.ndarray, features: np.ndarray, k: int, *, beta: float = 1.0, knn: int = 10,
            iters: int = 300) -> tuple[list[int], np.ndarray]:
    """InfoMax selection of k records. Returns (indices ordered by x, relaxed solution x).

    The solver is omniselect.core.selection.relaxation (shared with the v2.1 solver challengers).
    """
    return infomax_select(importance, as2d(features), k, beta=beta, knn=knn, iters=iters)


@register("infomax", method_dir="InfoMax", family="new", fidelity="local implementation",
          requires=("influence",), display="InfoMax")
def infomax_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """InfoMax with the influence channel as importance and the track representation as features."""
    sel, _ = infomax(ctx.influence, ctx.features, k)
    return sel
