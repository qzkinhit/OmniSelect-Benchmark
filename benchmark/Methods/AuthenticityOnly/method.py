"""Authenticity-only strategies and the rank-inverting control.

auth_only keeps the k records with the highest composite authenticity (kNN label agreement on
the track representation). auth2_only uses the mechanism-matched label arm min(rank(out-of-fold
agreement), rank(kNN agreement)). Auth3_only adds the kNN inlier arm. auth_bottom keeps the k
lowest-authenticity records (the control whose ranking is inverted).
"""
from __future__ import annotations

import numpy as np

from benchmark.Methods._common import top_k
from omniselect.core.portfolio.registry import SelectionContext, register
from omniselect.core.signals.authenticity_v2 import auth_full, auth_label


@register("auth_only", method_dir="AuthenticityOnly", family="builtin", fidelity="exact",
          requires=("auth",), display="Authenticity-only")
def auth_only(ctx: SelectionContext, k: int) -> list[int]:
    """Top k of the authenticity channel."""
    return top_k(ctx.auth, k)


@register("auth2_only", method_dir="AuthenticityOnly", family="builtin", fidelity="exact",
          requires=("labels",), display="Authenticity v2")
def auth2_only(ctx: SelectionContext, k: int) -> list[int]:
    """Top k of auth_label on ``extras['auth_features']`` (the track representation otherwise)."""
    feats = ctx.extras.get("auth_features", ctx.features)
    knn = int(ctx.extras.get("knn", 15))
    return top_k(auth_label(feats, ctx.labels, knn=knn, seed=ctx.seed), k)


@register("auth3_only", method_dir="AuthenticityOnly", family="builtin", fidelity="exact",
          requires=("labels",), display="Authenticity v3")
def auth3_only(ctx: SelectionContext, k: int) -> list[int]:
    """Top k of auth_full (label arm and inlier arm) on ``extras['auth_features']``."""
    feats = ctx.extras.get("auth_features", ctx.features)
    return top_k(auth_full(feats, ctx.labels, seed=ctx.seed), k)


@register("auth_bottom", method_dir="AuthenticityOnly", family="control", fidelity="exact",
          requires=("auth",), display="Authenticity bottom")
def auth_bottom(ctx: SelectionContext, k: int) -> list[int]:
    """The k records with the lowest authenticity (rank-inverting control)."""
    return [int(i) for i in np.argsort(ctx.auth, kind="stable")[:k]]
