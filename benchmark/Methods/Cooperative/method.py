"""Cooperative herding and cleanliness top-k, the two portfolio members of protocol v2.2.

``coop_herding`` (reference-eligible, reads the pool only): the gate keeps the ceil(rho k) records with
the highest authenticity (rho = cooperative.reference_rho, 1.3 by default), the outlier and duplicate
vetoes remove records in the selector's feature space, and herding picks k records inside
(omniselect/core/selection/cooperative.py). ``clean_top`` (challenger-only, the learned cleanliness
score reads V_con): the k records with the highest cleanliness. Token-budgeted tracks get the whole order.
"""
from __future__ import annotations

import numpy as np

from omniselect.core.portfolio.registry import SelectionContext, register


@register("coop_herding", method_dir="Cooperative", family="new", fidelity="local implementation",
          requires=("cooperative",), display="Cooperative herding")
def coop_herding(ctx: SelectionContext, k: int) -> list[int]:
    """Herding inside the authenticity gate of width ``reference_rho`` after both vetoes."""
    family = ctx.extras["cooperative"]
    selection, info = family.select("authenticity", family.reference_rho, "herding", k, name="coop_herding")
    ctx.extras.setdefault("strategy_info", {})["coop_herding"] = info
    return selection


@register("clean_top", method_dir="Cooperative", family="new", fidelity="local implementation",
          requires=("cleanliness",), display="Cleanliness top-k")
def clean_top(ctx: SelectionContext, k: int) -> list[int]:
    """Top k of the learned cleanliness score, ties by pool index (full order when the track cuts by tokens)."""
    order = [int(i) for i in np.argsort(-np.asarray(ctx.extras["cleanliness"], dtype=float), kind="stable")]
    return order if ctx.extras.get("budget_cut") is not None else order[: int(k)]
