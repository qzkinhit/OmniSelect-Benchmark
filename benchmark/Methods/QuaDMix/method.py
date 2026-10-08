"""QuaDMix (arXiv 2504.16511): quality and domain-aware sampling, published-core transfer.

``quadmix_expected_counts`` implements Eqs. 1 to 3 (merged normalized quality, token-weighted
within-domain percentile rank, sigmoid expected-repeat sampler). ``quadmix_published_core``
(quadmix_pub) takes k-means domains when native labels are missing and converts expected counts
to a fixed-size subset by Gumbel top-k with default_rng(seed). ``quadmix`` is the withdrawn
style proxy, kept only to replay historical portfolios.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from benchmark.Methods._common import as2d
from omniselect.core.portfolio.registry import SelectionContext, register


def quadmix_expected_counts(quality: np.ndarray, domains: np.ndarray, *,
                            alpha: Optional[np.ndarray] = None,
                            lambdas: float | np.ndarray = 100.0,
                            omegas: float | np.ndarray = 0.05,
                            etas: float | np.ndarray = 1.0,
                            epsilons: float | np.ndarray = 0.001,
                            token_weights: Optional[np.ndarray] = None,
                            quality_higher_is_better: bool = True) -> np.ndarray:
    """QuaDMix Eqs. 1 to 3 returning the expected sampling count of each record."""
    Q = np.asarray(quality, dtype=float)
    if Q.ndim == 1:
        Q = Q[:, None]
    d = np.asarray(domains)
    if len(Q) != len(d):
        raise ValueError("quality and domains length mismatch")
    n, n_criteria = Q.shape
    tw = np.ones(n, dtype=float) if token_weights is None else np.asarray(token_weights, dtype=float)
    if len(tw) != n or np.any(tw <= 0):
        raise ValueError("token_weights must be positive and match the pool")
    unique = np.unique(d)
    m = len(unique)

    def per_domain(value, name):
        a = np.asarray(value, dtype=float)
        if a.ndim == 0:
            return np.repeat(float(a), m)
        if a.shape != (m,):
            raise ValueError(f"{name} must be scalar or have one value per domain")
        return a

    lam = per_domain(lambdas, "lambdas")
    omg = per_domain(omegas, "omegas")
    eta = per_domain(etas, "etas")
    eps = per_domain(epsilons, "epsilons")
    if alpha is None:
        A = np.ones((m, n_criteria), dtype=float) / n_criteria
    else:
        A = np.asarray(alpha, dtype=float)
        if A.ndim == 1:
            A = np.repeat(A[None, :], m, axis=0)
        if A.shape != (m, n_criteria):
            raise ValueError("alpha must have shape (domains, criteria)")
        A = A / (A.sum(axis=1, keepdims=True) + 1e-12)

    qnorm = np.empty_like(Q)
    for j in range(n_criteria):
        lo, hi = float(Q[:, j].min()), float(Q[:, j].max())
        qnorm[:, j] = (Q[:, j] - lo) / (hi - lo + 1e-12)
    if quality_higher_is_better:
        qnorm = 1.0 - qnorm

    counts = np.zeros(n, dtype=float)
    for mi, label in enumerate(unique):
        idx = np.where(d == label)[0]
        merged = qnorm[idx] @ A[mi]
        order = np.argsort(merged, kind="mergesort")
        ordered_idx = idx[order]
        cumulative = np.cumsum(tw[ordered_idx])
        rank = cumulative / cumulative[-1]
        z = np.clip(-lam[mi] * (omg[mi] - rank), -60.0, 60.0)
        expected = np.where(rank <= omg[mi], eta[mi] / (1.0 + np.exp(z)) + eps[mi], eps[mi])
        counts[ordered_idx] = expected
    return counts


def quadmix_published_core(quality: np.ndarray, features: np.ndarray, k: int, *,
                           domains: Optional[np.ndarray] = None,
                           token_weights: Optional[np.ndarray] = None,
                           seed: int = 0, n_domains: int = 8,
                           lambdas: float | np.ndarray = 100.0,
                           omegas: float | np.ndarray = 0.05,
                           etas: float | np.ndarray = 1.0,
                           epsilons: float | np.ndarray = 0.001) -> list:
    """Fixed-budget QuaDMix: Eq. 3 expected counts, Gumbel top-k without replacement."""
    X = as2d(features)
    n = len(X)
    k = min(int(k), n)
    if domains is None:
        from sklearn.cluster import KMeans
        nc = min(max(1, int(n_domains)), n)
        domains = KMeans(n_clusters=nc, n_init=3, random_state=seed).fit_predict(X)
    counts = quadmix_expected_counts(
        quality, np.asarray(domains), lambdas=lambdas, omegas=omegas,
        etas=etas, epsilons=epsilons, token_weights=token_weights,
        quality_higher_is_better=True,
    )
    rng = np.random.default_rng(seed)
    keys = np.log(counts + 1e-30) + rng.gumbel(size=n)
    return [int(i) for i in np.argsort(-keys, kind="stable")[:k]]


def quadmix(quality: np.ndarray, features: np.ndarray, k: int, *, alpha: float = 0.5,
            bins: int = 20, seed: int = 0) -> list:
    """Withdrawn QuaDMix-style proxy: quality buckets with farthest-first picks inside each bucket."""
    q = np.asarray(quality, float)
    X = as2d(np.asarray(features, float))
    n = len(q)
    k = min(k, n)
    if k <= 0:
        return []
    order = np.argsort(q, kind="mergesort")
    edges = np.quantile(q, np.linspace(0, 1, bins + 1))
    bucket = np.searchsorted(edges[1:-1], q, side="right")
    weights = np.array([(0.5 * (edges[b] + edges[b + 1])) for b in range(bins)])
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
    weights = (1 - alpha) + alpha * weights
    weights /= weights.sum()
    out = []
    for b in range(bins):
        mem = order[bucket[order] == b]
        kb = int(round(k * weights[b]))
        if len(mem) and kb:
            sub = X[mem]
            rng = np.random.default_rng(seed + b)
            picked = [int(rng.integers(len(mem)))]
            available = np.ones(len(mem), dtype=bool)
            available[picked[0]] = False
            dist = np.linalg.norm(sub - sub[picked[0]][None], axis=1)
            while len(picked) < min(kb, len(mem)):
                i = int(np.argmax(np.where(available, dist, -np.inf)))
                picked.append(i)
                available[i] = False
                dist = np.minimum(dist, np.linalg.norm(sub - sub[i][None], axis=1))
            out.extend(int(mem[p]) for p in picked)
    if len(out) < k:
        selected = set(out)
        rest = [int(i) for i in order[::-1] if int(i) not in selected]
        out.extend(rest[: k - len(out)])
    out = out[:k]
    if len(out) != k or len(set(out)) != k:
        raise RuntimeError("quadmix must return exactly k unique records")
    return out


@register("quadmix_pub", method_dir="QuaDMix", family="external", fidelity="published-core transfer",
          requires=("auth",), display="QuaDMix-pub")
def quadmix_pub_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """quadmix_published_core with authenticity as quality and k-means domains on the representation."""
    return quadmix_published_core(ctx.auth, ctx.features, k, seed=ctx.seed)


@register("quadmix", method_dir="QuaDMix", family="external", fidelity="proxy (withdrawn)",
          requires=("auth",), display="QuaDMix-style proxy (withdrawn)")
def quadmix_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """Withdrawn style proxy. Only historical portfolio modes include it."""
    return quadmix(ctx.auth, ctx.features, k, seed=ctx.seed)
