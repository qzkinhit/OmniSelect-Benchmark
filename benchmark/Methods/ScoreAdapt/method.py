"""Disclosed EL2N/GraNd/CCS adaptations; never aliases of the original methods.

Forecasting uses a pool-warmed DLinear. Text uses a fixed pretrained LM and
RMS token-level scores. CCS retains its cutoff, equal-width strata and
sparse-first allocation; the text allocation unit is a token rather than a row.
"""
from __future__ import annotations

import numpy as np

from omniselect.core.portfolio.registry import SelectionContext, register

ADAPT_METHODS = frozenset(("el2n_adapt", "grand_adapt", "ccs_adapt"))


def _vector(values):
    out = np.asarray(values, dtype=float)
    if out.ndim != 1 or not len(out) or not np.isfinite(out).all():
        raise ValueError("adaptation scores must be a nonempty finite vector")
    return out


def squared_error_scores(phi, prediction, target):
    """Residual L2 and exact ||d(0.5||prediction-target||²)/dTheta||_F.

    phi includes every linear component's bias column, as DLinear's existing
    [trend, 1, seasonal, 1] representation does. This is one early checkpoint,
    not the original GraNd expectation over independently initialized networks.
    """
    phi, prediction, target = [np.asarray(x, dtype=float) for x in (phi, prediction, target)]
    if phi.ndim != 2 or prediction.ndim != 2 or prediction.shape != target.shape or len(phi) != len(target):
        raise ValueError("incompatible forecasting score arrays")
    error = _vector(np.linalg.norm(prediction - target, axis=1))
    gradient = _vector(error * np.linalg.norm(phi, axis=1))
    return {"el2n": error, "grand": gradient}


def _strata(difficulty, cutoff, bins):
    difficulty = _vector(difficulty)
    if not 0 <= cutoff < 1 or bins < 1:
        raise ValueError("invalid CCS cutoff/bins")
    order = np.argsort(difficulty)  # preserve the existing CCS tie behavior
    keep = order[:int(len(order) * (1.0 - cutoff))]
    if not len(keep):
        raise ValueError("CCS cutoff leaves no candidates")
    bins = max(1, min(int(bins), len(keep)))
    d = difficulty[keep]
    edges = np.linspace(float(d.min()), float(d.max()), bins + 1)
    assignment = np.searchsorted(edges[1:-1], d, side="right")
    strata = [keep[assignment == b] for b in range(bins)]
    strata = sorted(enumerate(strata), key=lambda item: (len(item[1]), item[0]))
    return keep, strata


def ccs_by_score(difficulty, k, *, cutoff=0.1, bins=50, seed=0):
    """Existing published-core CCS allocation, with an explicit difficulty input."""
    difficulty = _vector(difficulty)
    if not 0 <= k <= len(difficulty):
        raise ValueError("invalid CCS record budget")
    keep, strata = _strata(difficulty, cutoff, bins)
    if len(keep) < k:
        raise ValueError("CCS cutoff leaves fewer records than the requested budget")
    rng, out, remaining = np.random.default_rng(seed), [], int(k)
    for position, (_, stratum) in enumerate(strata):
        take = min(len(stratum), remaining // (len(strata) - position))
        if take:
            out.extend(int(i) for i in rng.permutation(stratum)[:take])
            remaining -= take
    if remaining:
        selected = set(out)
        out.extend([int(i) for i in keep if int(i) not in selected][:remaining])
    if len(out) != k or len(set(out)) != k:
        raise RuntimeError("CCS adaptation did not satisfy the record budget")
    return out


def _ccs_token_selection(difficulty, tokens, budget, *, cutoff, bins, seed):
    """Sparse-first token quotas, random prefixes within strata, then easy-first fill.

    Quotas never overshoot inside a stratum. The final fill stops at the first
    record reaching the domain budget, matching the text track's budget cut.
    If the easiest 90% cannot supply that budget, fail rather than re-admit the
    hardest records or silently shrink the training budget.
    """
    keep, strata = _strata(difficulty, cutoff, bins)
    if budget <= 0 or tokens[keep].sum() < budget:
        raise ValueError("CCS retained records cannot supply the domain token budget")
    rng, out, remaining = np.random.default_rng(seed), [], int(budget)
    for position, (_, stratum) in enumerate(strata):
        quota = remaining // (len(strata) - position)
        shuffled = rng.permutation(stratum)
        cumulative = np.cumsum(tokens[shuffled])
        take = int(np.searchsorted(cumulative, quota, side="right"))
        out.extend(int(i) for i in shuffled[:take])
        remaining -= int(tokens[shuffled[:take]].sum())
    if remaining:
        selected = set(out)
        for index in keep:
            if int(index) not in selected:
                out.append(int(index))
                remaining -= int(tokens[index])
                if remaining <= 0:
                    break
    return out


def ccs_token_order(difficulty, domains, tokens, budgets, *, cutoff=0.1, bins=50, seed=0):
    """Full permutation whose per-domain budget prefixes are the CCS selections."""
    difficulty = _vector(difficulty)
    tokens = np.asarray(tokens)
    domains = np.asarray(domains)
    if tokens.shape != difficulty.shape or domains.shape != difficulty.shape:
        raise ValueError("incompatible token-budget inputs")
    if not np.issubdtype(tokens.dtype, np.integer) or np.any(tokens <= 0):
        raise ValueError("token counts must be positive integers")
    if set(domains) != set(budgets):
        raise ValueError("domain budgets do not cover the pool")
    out = []
    for domain in sorted(budgets):
        indices = np.flatnonzero(domains == domain)
        selected = _ccs_token_selection(difficulty[indices], tokens[indices], int(budgets[domain]),
                                       cutoff=cutoff, bins=bins, seed=seed)
        out.extend(int(indices[i]) for i in selected)
    chosen = set(out)
    out.extend(i for i in range(len(difficulty)) if i not in chosen)
    return out


def _scores(ctx, method):
    payload = ctx.extras["adapt_scores"]
    ctx.extras.setdefault("strategy_info", {})[method] = {
        **payload["info"], "method": method, "adaptation": True,
        "fidelity": "disclosed score/gradient adaptation; not an original-task reproduction",
    }
    return payload


def _rank(ctx, k, method, channel):
    score = _vector(_scores(ctx, method)[channel])
    if len(score) != ctx.n:
        raise ValueError("adaptation score count differs from pool size")
    order = [int(i) for i in np.argsort(-score, kind="stable")]
    return order if ctx.extras.get("budget_cut") is not None else order[:k]


@register("el2n_adapt", method_dir="ScoreAdapt", family="external", fidelity="qualitative protocol transfer",
          requires=("adapt_scores",), display="EL2N-adapt")
def el2n_adapt(ctx: SelectionContext, k: int) -> list[int]:
    return _rank(ctx, k, "el2n_adapt", "el2n")


@register("grand_adapt", method_dir="ScoreAdapt", family="external", fidelity="proxy",
          requires=("adapt_scores",), display="GraNd-adapt")
def grand_adapt(ctx: SelectionContext, k: int) -> list[int]:
    return _rank(ctx, k, "grand_adapt", "grand")


@register("ccs_adapt", method_dir="ScoreAdapt", family="external", fidelity="published-core transfer",
          requires=("adapt_scores",), display="CCS-adapt")
def ccs_adapt(ctx: SelectionContext, k: int) -> list[int]:
    payload = _scores(ctx, "ccs_adapt")
    info = ctx.extras["strategy_info"]["ccs_adapt"]
    info.update({"cutoff": 0.1, "bins": 50, "difficulty": "el2n_adapt", "sampling_seed": ctx.seed})
    if "tokens" in payload:
        info["allocation_unit"] = "tokens_per_domain"
        return ccs_token_order(payload["el2n"], payload["domains"], payload["tokens"],
                               payload["budgets"], seed=ctx.seed)
    info["allocation_unit"] = "records"
    return ccs_by_score(payload["el2n"], k, seed=ctx.seed)
