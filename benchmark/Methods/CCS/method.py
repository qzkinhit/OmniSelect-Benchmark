"""CCS (Zheng et al., ICLR 2023), local EL2N-binned transfer.

Drop the hardest ``cutoff`` fraction by EL2N difficulty, split the remaining range into ``bins``
equal-width strata (half-open assignment), allocate the budget evenly starting from the sparsest
stratum, sample within each stratum with default_rng(seed), and fill any remainder in easy-to-hard
order. The official pipeline uses AUM scores. This row does not.
"""
from __future__ import annotations

import numpy as np

from benchmark.Methods._common import softmax
from omniselect.core.portfolio.registry import SelectionContext, register


def ccs_from_difficulty(diff: np.ndarray, k: int, *, bins: int = 50, cutoff: float = 0.1, seed: int = 0) -> list:
    """Exactly k unique records by coverage-centric stratified sampling over a difficulty vector."""
    diff = np.asarray(diff, float)
    n = len(diff)
    k = min(k, n)
    order = np.argsort(diff)
    keep = order[: int(n * (1.0 - cutoff))]
    d = diff[keep]
    if len(keep) < k:
        raise ValueError(f"CCS cutoff leaves {len(keep)} candidates for requested budget {k}")
    bins = max(1, min(int(bins), len(keep)))
    edges = np.linspace(float(d.min()), float(d.max()), bins + 1)
    assignment = np.searchsorted(edges[1:-1], d, side="right")
    strata = [keep[assignment == b] for b in range(bins)]
    rng = np.random.default_rng(seed)
    out: list[int] = []
    remaining_budget = k
    for position, (_, stratum) in enumerate(sorted(enumerate(strata), key=lambda item: (len(item[1]), item[0]))):
        remaining_strata = bins - position
        allocation = remaining_budget // remaining_strata
        take = min(len(stratum), allocation)
        if take:
            out.extend(int(i) for i in rng.permutation(stratum)[:take])
            remaining_budget -= take
    if remaining_budget:
        selected = set(out)
        rest = [int(i) for i in keep if int(i) not in selected]
        out.extend(rest[:remaining_budget])
    if len(out) != k or len(set(out)) != k:
        raise RuntimeError("CCS must return exactly k unique records")
    return out


def ccs(
    probs_or_logits: np.ndarray,
    labels: np.ndarray,
    k: int,
    *,
    is_logits: bool = False,
    bins: int = 50,
    cutoff: float = 0.1,
    seed: int = 0,
) -> list:
    """Exactly k unique records by coverage-centric stratified sampling."""
    P = softmax(np.asarray(probs_or_logits, float)) if is_logits else np.asarray(probs_or_logits, float)
    y = np.asarray(labels).astype(int)
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    diff = np.linalg.norm(P - onehot, axis=1)
    return ccs_from_difficulty(diff, k, bins=bins, cutoff=cutoff, seed=seed)


@register("ccs", method_dir="CCS", family="external", fidelity="local implementation",
          requires=("proba", "labels"), display="CCS")
def ccs_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """CCS from the probe probabilities with the run seed."""
    return ccs(ctx.proba, ctx.labels, k, is_logits=False, seed=ctx.seed)
