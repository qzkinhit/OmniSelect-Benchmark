"""Tab-AICL acquisition rules (Ma et al., 2026) for the TabPFN context set.

``tabpfn_coreset`` runs k-center greedy in feature space. ``tabpfn_margin`` keeps the k records
with the lowest TabPFN margin p(top1) - p(top2). ``tabpfn_hybrid`` takes half the budget by margin
and fills the rest by k-center greedy over the remaining pool. The caller supplies the TabPFN
probabilities, so the module imports no model code.
"""
from __future__ import annotations

import numpy as np

__all__ = ["tabpfn_coreset", "tabpfn_margin", "tabpfn_hybrid", "select"]


def _as2d(features: np.ndarray) -> np.ndarray:
    f = np.asarray(features, dtype=np.float64)
    return f.reshape(f.shape[0], -1)


def tabpfn_coreset(features: np.ndarray, k: int, seed: int = 0, *, init: list | None = None) -> list:
    """k-center-greedy coreset for the TabPFN context (representativeness)."""
    X = _as2d(features)
    n = X.shape[0]
    k = min(k, n)
    rng = np.random.default_rng(seed)
    if init:
        chosen = list(dict.fromkeys(int(i) for i in init))[:k]
    else:
        chosen = [int(rng.integers(n))]
    dist = np.min([np.linalg.norm(X - X[c][None, :], axis=1) for c in chosen], axis=0)
    while len(chosen) < k:
        i = int(np.argmax(dist))
        if i in chosen:
            dist[i] = -1.0
            continue
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(X - X[i][None, :], axis=1))
    return chosen[:k]


def tabpfn_margin(probs: np.ndarray, k: int) -> list:
    """Uncertainty acquisition: keep the k candidates with the smallest top1-top2 margin."""
    P = np.asarray(probs, float)
    s = np.sort(P, axis=1)
    margin = s[:, -1] - (s[:, -2] if P.shape[1] > 1 else 0.0)
    return [int(i) for i in np.argsort(margin, kind="stable")[:min(k, len(margin))]]


def tabpfn_hybrid(features: np.ndarray, probs: np.ndarray, k: int, seed: int = 0) -> list:
    """Half by margin (informative), fill by coreset diversity (representative), deduped."""
    k = min(k, _as2d(features).shape[0])
    k_m = k // 2
    m = tabpfn_margin(probs, k_m)
    out = list(dict.fromkeys(m))
    if len(out) < k:
        out = tabpfn_coreset(features, k, seed=seed, init=out)
    return out[:k]


def select(method: str, k: int, *, features=None, probs=None, seed: int = 0) -> list:
    """Dispatch one Tab-AICL acquisition rule by name."""
    if method in ("tabpfn_coreset", "coreset"):
        return tabpfn_coreset(features, k, seed=seed)
    if method in ("tabpfn_margin", "margin"):
        return tabpfn_margin(probs, k)
    if method in ("tabpfn_hybrid", "hybrid"):
        return tabpfn_hybrid(features, probs, k, seed=seed)
    raise ValueError(f"unknown Tab-AICL rule: {method}")


# ---- strategy adapters (tabular track only; they read TabPFN in-context probabilities) --------
from omniselect.core.portfolio.registry import SelectionContext, register  # noqa: E402


@register("tabpfn_coreset", method_dir="TabAICL", family="external", fidelity="one-shot transfer",
          display="TabPFN coreset")
def tabpfn_coreset_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """Representativeness rule on the track representation."""
    return [int(i) for i in tabpfn_coreset(ctx.features, k, seed=ctx.seed)]


@register("tabpfn_margin", method_dir="TabAICL", family="external", fidelity="one-shot transfer",
          requires=("aicl_probs",), display="TabPFN margin")
def tabpfn_margin_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """Uncertainty (prediction margin) rule on TabPFN probabilities from a seed context."""
    return [int(i) for i in tabpfn_margin(ctx.extras["aicl_probs"](), k)]


@register("tabpfn_hybrid", method_dir="TabAICL", family="external", fidelity="one-shot transfer",
          requires=("aicl_probs",), display="TabPFN hybrid")
def tabpfn_hybrid_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """Representativeness plus informativeness rule."""
    return [int(i) for i in tabpfn_hybrid(ctx.features, ctx.extras["aicl_probs"](), k, seed=ctx.seed)]
