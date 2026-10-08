"""Influence Top-K: the k records with the largest influence score.

The influence channel is computed by the track (log-probability of the observed label under a
probe fit on the influence reference sample, the negative forecasting error of a reference
model, or the negative reference-model loss on text). Larger values mean a larger estimated
contribution.
"""
from __future__ import annotations

from benchmark.Methods._common import top_k
from omniselect.core.portfolio.registry import SelectionContext, register


@register("influence_only", method_dir="InfluenceOnly", family="builtin", fidelity="exact",
          requires=("influence",), display="Influence-only")
def influence_only(ctx: SelectionContext, k: int) -> list[int]:
    """Top k of the influence channel."""
    return top_k(ctx.influence, k)
