"""Alignment-only: the single-signal rule of the alignment channel of protocol v2.1.

Keeps the k records with the largest alignment (the cosine of the record's last-layer gradient with
the mean V_con gradient, or the LESS score on text), ties by pool index. On token-budgeted tracks the
whole order is returned and the track cuts it to the budget.
"""
from __future__ import annotations

import numpy as np

from omniselect.core.portfolio.registry import SelectionContext, register


@register("alignment_only", method_dir="AlignmentOnly", family="builtin", fidelity="exact",
          requires=("alignment",), display="Alignment-only")
def alignment_only(ctx: SelectionContext, k: int) -> list[int]:
    """Top k of the alignment channel (full order when the track cuts by tokens)."""
    order = [int(i) for i in np.argsort(-np.asarray(ctx.extras["alignment"], dtype=float), kind="stable")]
    return order if ctx.extras.get("budget_cut") is not None else order[: int(k)]
