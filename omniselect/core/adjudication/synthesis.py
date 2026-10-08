"""Challenger-only candidates synthesized on the construction split.

``consensus_vote`` weights each of the top-3 candidates by its construction gain over the
random reference and keeps the k records with the most votes. ``coordinate_ascent`` starts from
the best grid cell and moves one channel weight by +-0.15 while the construction utility
improves, for at most two rounds. With a held-half ``check`` the steps are fitted on one half of
the construction split and a step is accepted only if the utility on the other half does not
decrease. ``policy_search`` is the historical group-relative search.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from omniselect.core.selection.fusion_grid import fusion_select, parse_cell_name

Scored = tuple[str, list[int], float, bool]


def consensus_votes(
    scored_con: Sequence[Scored], n: int, top: int = 3, random_name: str = "random"
) -> tuple[np.ndarray, list[Scored]]:
    """Vote vector over records and the top-``top`` members that cast the votes.

    votes[i] += max(g - g_random, 0) + 1e-9 for each member selecting i. Without a random
    candidate the minimum construction gain is the base.
    """
    ranked = sorted(scored_con, key=lambda t: -t[2])[:top]
    randg = next((t[2] for t in scored_con if t[0] == random_name), None)
    base = randg if randg is not None else min(t[2] for t in scored_con)
    votes = np.zeros(n)
    for _, sel, g, _ref in ranked:
        votes[np.asarray(sel, dtype=int)] += max(g - base, 0.0) + 1e-9
    return votes, ranked


def consensus_vote(
    scored_con: Sequence[Scored], S: np.ndarray, k: int, n: int, top: int = 3, random_name: str = "random"
) -> list[int]:
    """The k records with the most votes. Ties broken by 1e-9 times the first channel."""
    votes, _ = consensus_votes(scored_con, n, top, random_name)
    return [int(i) for i in np.argsort(-(votes + 1e-9 * S[0]), kind="stable")[:k]]


def consensus_diverse(
    votes: np.ndarray, records: Sequence, features: np.ndarray, k: int, lam: float
) -> list[int]:
    """Votes used as importance in BudgetSelector(lam). Historical METHOD_V2 candidate."""
    from omniselect.core.selection.budget_select import BudgetSelector

    return [int(i) for i in BudgetSelector(lam=lam).select(records, votes.astype(float), k, features=features)]


def consensus_complementary(
    ranked: Sequence[Scored], consensus: list[int], threshold: float
) -> list[int]:
    """The vote selection when the top-3 overlap |A & B & C| / |A | B | C| is below ``threshold``,
    otherwise the selection of the best member. Historical METHOD_V2C candidate."""
    sets = [set(int(i) for i in sel) for _, sel, _g, _r in ranked]
    inter = sets[0] & sets[1] & sets[2] if len(sets) >= 3 else set()
    union = set().union(*sets) if sets else set()
    overlap = len(inter) / max(1, len(union))
    if overlap < float(threshold):
        return list(consensus)
    return [int(i) for i in max(ranked, key=lambda t: t[2])[1]]


def coordinate_ascent(
    scored_con: Sequence[Scored],
    S: np.ndarray,
    records: Sequence,
    features: np.ndarray,
    k: int,
    prefilter: np.ndarray,
    probe: Callable[[list[int]], float],
    rounds: int = 2,
    step: float = 0.15,
    post: Callable[[list[int]], list[int]] | None = None,
    check: Callable[[list[int]], float] | None = None,
    stats: dict | None = None,
    similarity=None,
) -> tuple[str, list[int]] | None:
    """Coordinate ascent over channel weights from the construction-best grid cell.

    Returns (name, selection) of the final weights, or None when no grid cell is present.
    ``post`` truncates each full order to a token budget on text. With ``check`` (held half) the
    start cell is the best grid cell under ``probe`` (the fitting half), a step that raises ``probe``
    is proposed, and it is accepted only if ``check`` (the other half) does not decrease. ``stats``
    receives the trials, proposed and accepted steps and both halves' utilities of the final weights.
    """
    cells = [t for t in scored_con if t[0].startswith("fuse ") and "solver=" not in t[0]]
    if not cells:
        return None
    if check is not None:
        start = max(cells, key=lambda t: float(probe(list(t[1]))))
    else:
        start = max(cells, key=lambda t: t[2])
    parsed = parse_cell_name(start[0])
    if parsed is None:
        return None
    w0, q0, lam0 = parsed

    def sel_of(wv: np.ndarray) -> list[int]:
        wv = np.clip(wv, 0, None)
        wv = wv / (wv.sum() + 1e-12)
        sel = fusion_select(S, prefilter, records, features, wv, q0, lam0, k, similarity=similarity)
        return post(sel) if post is not None else sel

    cur_w, cur_g = w0.copy(), float(probe(sel_of(w0)))
    cur_h = float(check(sel_of(w0))) if check is not None else None
    trials = proposed = accepted = 0
    for _ in range(int(rounds)):
        improved = False
        for c in range(len(cur_w)):
            for delta in (step, -step):
                trial = cur_w.copy()
                trial[c] = max(0.0, trial[c] + delta)
                sel = sel_of(trial)
                g = float(probe(sel))
                trials += 1
                if g > cur_g:
                    proposed += 1
                    if check is not None:
                        h = float(check(sel))
                        if h < cur_h:
                            continue
                        cur_h = h
                    accepted += 1
                    cur_w, cur_g, improved = trial, g, True
        if not improved:
            break
    if stats is not None:
        stats.update({"start_cell": start[0], "trials": trials, "proposed_steps": proposed,
                      "accepted_steps": accepted, "u_fit_half": cur_g, "u_check_half": cur_h,
                      "held_half": check is not None})
    wn = cur_w / (cur_w.sum() + 1e-12)
    name = f"fuse learned w={tuple(round(float(x), 2) for x in wn)} q={q0} lam={lam0}"
    return name, sel_of(cur_w)


def policy_search(
    S: np.ndarray,
    records: Sequence,
    features: np.ndarray,
    k: int,
    prefilter: np.ndarray,
    probe: Callable[[list[int]], float],
    seed: int,
) -> tuple[str, list[int]] | None:
    """Group-relative evolutionary search over (weights, q, lam). Historical, off by default."""
    rng = np.random.default_rng(seed + 31)
    group, steps = 6, 4
    mu = np.zeros(5)
    sd = np.array([1.0, 1.0, 1.0, 1.5, 1.5])

    def cfg(th: np.ndarray) -> tuple[np.ndarray, float, float]:
        e = np.exp(th[:3] - th[:3].max())
        return e / e.sum(), float(0.5 / (1.0 + np.exp(-th[3]))), float(0.8 / (1.0 + np.exp(-th[4])))

    best_r, best_cfg = -np.inf, None
    for _ in range(steps):
        thetas = mu + sd * rng.standard_normal((group, 5))
        rs = np.zeros(group)
        for gi in range(group):
            w, q, lam = cfg(thetas[gi])
            rs[gi] = float(probe(fusion_select(S, prefilter, records, features, w, q, lam, k)))
            if rs[gi] > best_r:
                best_r, best_cfg = rs[gi], (w, q, lam)
        loo = (rs.sum() - rs) / max(group - 1, 1)
        adv = rs - loo
        grad = (adv[:, None] * (thetas - mu) / (sd ** 2)).mean(0)
        move = 0.8 * (sd ** 2) * grad
        mnorm = float(np.linalg.norm(move / sd))
        if mnorm > 0.75:
            move *= 0.75 / mnorm
        mu = mu + move
    if best_cfg is None:
        return None
    w, q, lam = best_cfg
    name = f"grpo_policy w={tuple(round(float(x), 2) for x in w)} q={q:.2f} lam={lam:.2f}"
    return name, fusion_select(S, prefilter, records, features, w, q, lam, k)
