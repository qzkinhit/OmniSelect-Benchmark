"""Full-data selector: returns every pool id in its original order (the full-data reference row)."""
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
        A copy of ``ids``. The full pool is always kept.
    """
    return list(ids)
