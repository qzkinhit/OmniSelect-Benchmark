"""full_data selection logic — the identity selector.

This baseline trains on the *entire* pool, so "selection" is a no-op that returns
all ids unchanged. It is the upper-bound reference: any budget-constrained method is
judged by how close it gets to full-data quality while keeping far fewer records.

Kept independent of the core package (pure stdlib) so the baseline stays
self-contained, mirroring the other ``baselines/<name>/method`` modules.
"""
from __future__ import annotations

from typing import List, Sequence


def select(ids: Sequence[str]) -> List[str]:
    """Return all ids, order preserved (no records are dropped).

    Parameters
    ----------
    ids : sequence of str
        The pool's record ids, in their on-disk order.

    Returns
    -------
    list of str
        A copy of ``ids`` — the full pool is always kept.
    """
    return list(ids)
