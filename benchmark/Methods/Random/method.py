"""Uniform random selection of k ids without replacement from a seeded numpy generator."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def select(ids: Sequence[str], k: int, seed: int = 0) -> List[str]:
    """Pick ``k`` ids uniformly at random, without replacement.

    Parameters
    ----------
    ids : sequence of str
        Candidate record ids (the pool).
    k : int
        Number of ids to keep; clamped into ``[0, len(ids)]``.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    list of str
        The selected ids, in their drawn order.
    """
    ids = list(ids)
    n = len(ids)
    k = max(0, min(int(k), n))
    if k == 0:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    return [ids[int(i)] for i in idx]


# ---- strategy adapter -------------------------------------------------------------------------
from omniselect.core.portfolio.registry import SelectionContext, register  # noqa: E402
from omniselect.tools.pairing import stable_seed  # noqa: E402


@register("random", method_dir="Random", family="control", fidelity="exact", display="Random")
def random_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """k indices of a permutation from default_rng(stable_seed(seed, 'select', 'random')).

    Without the paired RNG discipline the canonical runners drew from the shared per-run
    generator; ``ctx.extras['shared_rng']`` supplies it in that mode.
    """
    shared = ctx.extras.get("shared_rng")
    rng = shared if shared is not None else np.random.default_rng(stable_seed(ctx.seed, "select", "random"))
    return [int(i) for i in rng.permutation(ctx.n)[:k]]
