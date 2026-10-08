"""Fixed-weight fusion: authenticity gate, preregistered influence and coverage weights, budget selector.

The importance of a record is the console blend of the redundancy and influence channels with
weights (1 - w_infl, w_infl) computed by the track (``extras['imp_dyn']``). Records below the
``auth_q`` quantile of authenticity are gated. ``sentinel`` (canonical) writes -1e9 into gated
records before BudgetSelector min-max normalizes the vector. ``finite_subset`` runs the selector
on the passing records and fills any shortfall by ungated importance.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from omniselect.core.portfolio.registry import SelectionContext, register
from omniselect.core.selection.budget_select import BudgetSelector


def fixed_fusion(
    records: Sequence,
    importance: np.ndarray,
    auth: np.ndarray,
    k: int,
    *,
    auth_q: float,
    lam: float,
    features: np.ndarray,
    gate: str = "sentinel",
) -> list[int]:
    """k indices (or a full order when k = n) of the gated budget selection."""
    thr = float(np.quantile(auth, auth_q))
    if gate == "sentinel":
        imp = np.asarray(importance, dtype=float).copy()
        imp[auth < thr] = -1e9
        return [int(i) for i in BudgetSelector(lam=lam).select(records, imp, k, features=features)]
    if gate != "finite_subset":
        raise ValueError(f"unknown gate {gate!r}")
    imp = np.asarray(importance, dtype=float)
    keep = np.where(auth >= thr)[0]
    rest = np.where(auth < thr)[0]
    fill = [int(i) for i in rest[np.argsort(-imp[rest], kind="stable")]]
    if len(keep) <= k:
        head = [int(i) for i in keep]
        if len(keep) > 1:
            sub = BudgetSelector(lam=lam).select([records[int(i)] for i in keep], imp[keep], len(keep),
                                                 features=features[keep])
            head = [int(keep[int(j)]) for j in sub]
        return (head + fill)[:k]
    sub = BudgetSelector(lam=lam).select([records[int(i)] for i in keep], imp[keep], k, features=features[keep])
    return [int(keep[int(j)]) for j in sub]


@register("mmdataselect", method_dir="FixedFusion", family="builtin", fidelity="exact",
          requires=("auth", "imp_dyn"), display="Fixed fusion")
def mmdataselect(ctx: SelectionContext, k: int) -> list[int]:
    """Fixed fusion on the track's console importance and representation."""
    return fixed_fusion(ctx.records, ctx.extras["imp_dyn"], ctx.auth, k, auth_q=ctx.extras.get("auth_q", 0.25),
                        lam=ctx.extras.get("lam", 0.5), features=ctx.features,
                        gate=ctx.extras.get("fixed_fusion_gate", "sentinel"))
