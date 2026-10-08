"""Successive halving of the fusion grid on the construction split.

Each round ranks the remaining cells by the construction utility (stable sort, higher first)
and keeps max(keep, half of the pool) cells until at most ``keep`` remain, with keep at least 2.
This is the canonical screening loop. ``low_fidelity_halving`` runs the same loop with a
low-fidelity utility (a subsample of each selection and a short schedule, supplied by the driver)
and scores the finalists with the full-fidelity utility. ``kendall_tau`` compares two orderings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class ScreeningTrace:
    """Pool sizes per round, the number of utility evaluations, and the screened-out names."""

    total: int
    finalists: int
    evaluations: int
    rounds: list[list[tuple[str, float]]] = field(default_factory=list)
    screened_out: list[str] = field(default_factory=list)


def successive_halving(
    cells: Sequence[tuple],
    utility: Callable[[list[int]], float],
    keep: int,
) -> tuple[list[tuple], ScreeningTrace]:
    """Halve ``cells`` (tuples whose first two fields are name and selection) down to ``keep``."""
    floor = max(2, int(keep))
    pool = list(cells)
    trace = ScreeningTrace(total=len(pool), finalists=len(pool), evaluations=0)
    while len(pool) > floor:
        scored = [(cell, float(utility(cell[1]))) for cell in pool]
        trace.evaluations += len(pool)
        ranked = [cell for cell, _ in sorted(scored, key=lambda t: -t[1])]
        trace.rounds.append([(cell[0], value) for cell, value in scored])
        pool = ranked[: max(floor, len(pool) // 2)]
        if len(pool) == len(ranked):
            break
    kept = {cell[0] for cell in pool}
    trace.screened_out = [cell[0] for cell in cells if cell[0] not in kept]
    trace.finalists = len(pool)
    return pool, trace


def low_fidelity_halving(
    cells: Sequence[tuple],
    low_utility: Callable[[list[int]], float],
    full_utility: Callable[[list[int]], float],
    keep: int,
) -> tuple[list[tuple], ScreeningTrace, dict[str, float]]:
    """Successive halving on ``low_utility``, then ``full_utility`` of every finalist.

    Returns the finalists, the trace of the low-fidelity rounds, and the full-fidelity utility of
    each finalist by name.
    """
    pool, trace = successive_halving(cells, low_utility, keep)
    full = {cell[0]: float(full_utility(cell[1])) for cell in pool}
    return pool, trace, full


def kendall_tau(first: Sequence[float], second: Sequence[float]) -> Optional[float]:
    """Kendall tau-b of two score vectors over the same items. None for fewer than 2 items or a constant vector."""
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if len(a) != len(b):
        raise ValueError("kendall_tau needs vectors of equal length")
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    from scipy.stats import kendalltau

    value = kendalltau(a, b).statistic
    return None if not np.isfinite(value) else float(value)
