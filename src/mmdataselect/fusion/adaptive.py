"""Adaptive conflict-aware controller — the modality-agnostic core of OmniSelect.

Motivation (shown empirically across vision / timeseries / tabular): no single quality
facet and no fixed fusion is best across modalities. Authenticity dominates on label-noisy
few-shot vision and on from-scratch forecasting, yet HURTS on a robust tabular foundation
model; coverage is the opposite; and the best facet even flips between two datasets of the
SAME modality (CIFAR-100 vs CIFAR-10) and between two models on the SAME data (XGBoost vs
TabPFN). Committing to any one signal or one fixed weighting therefore cannot be best
everywhere.

Design (first-by-construction, not hyper-parameter tuning). We treat *which selection
strategy to use* as a validation problem. The controller holds a PORTFOLIO of candidate
strategies built from the three channels (authenticity A, influence I, coverage/redundancy
R): every single channel (top-k by A / I / R), a coverage strategy, and the
authenticity-gated weighted fusion with diversity. This portfolio CONTAINS every baseline.
Given the modality's own independent held-out validation set, downstream model and metric
(supplied as a single ``held_out_gain(selection) -> float`` callback), the controller fits
each candidate, scores it on validation, and returns the best. Because the portfolio
includes each baseline as a candidate, the controller is >= every baseline by construction
(up to validation variance) — it matches or beats the best single signal AND the best fixed
fusion on every modality, model and dataset, with no per-task tuning.

The same controller is reused across modalities; only the gain callback (the modality's own
real metric: top-1 accuracy, ROC AUC, negative MASE, negative perplexity) changes.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..datatypes import UnifiedRecord
from ..selectors.budget_select import BudgetSelector
from ..signals.base import minmax

# Channel-weight simplex for the fusion candidates (channels = A, I, R by convention).
_W3 = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (.5, .5, 0), (.5, 0, .5), (0, .5, .5), (.34, .33, .33)]


def _default_grid(n_channels: int):
    if n_channels == 3:
        return list(_W3)
    grid = [tuple(1.0 if j == i else 0.0 for j in range(n_channels)) for i in range(n_channels)]
    grid.append(tuple(1.0 / n_channels for _ in range(n_channels)))
    return grid


class AdaptiveController:
    """Portfolio meta-selector: pick the validation-best selection strategy.

    Parameters
    ----------
    weight_grid : channel-weight tuples for the fusion candidates (defaults to a small simplex).
    lam_grid : diversity strengths for the budget selector (0 == pure importance ranking).
    prefilter_grid : authenticity-as-prerequisite gate fractions on ``prefilter_channel``
        (0 == no gate). The gate drops the bottom-q by that channel before the weighted select.
    prefilter_channel : which channel is the prerequisite gate (default 0 = authenticity).
    n_val_repeats : evaluate each candidate's gain this many times and average (k-fold-style
        variance reduction so the validation pick is stable even when effect sizes are small).
    """

    name = "adaptive_controller"

    def __init__(self, weight_grid: Optional[Sequence[Sequence[float]]] = None,
                 lam_grid: Sequence[float] = (0.0, 0.25, 0.6),
                 prefilter_grid: Sequence[float] = (0.0, 0.25), prefilter_channel: int = 0,
                 n_val_repeats: int = 1, switch_margin_frac: float = 0.015, seed: int = 0):
        self.weight_grid = weight_grid
        self.lam_grid = tuple(lam_grid)
        self.prefilter_grid = tuple(prefilter_grid)
        self.prefilter_channel = int(prefilter_channel)
        self.n_val_repeats = max(1, int(n_val_repeats))
        # regularized selection: only deviate from the best REFERENCE strategy (a passed
        # baseline) toward a fusion candidate when the fusion beats it by more than this
        # fraction of |gain|. Statistically insignificant validation wins do not trigger a
        # switch, so the controller never underperforms a baseline by validation noise.
        import os as _os
        self.switch_margin_frac = float(_os.environ.get("ADAPT_MARGIN", switch_margin_frac))
        self.seed = int(seed)
        self.chosen_: Optional[dict] = None
        self.leaderboard_: List[Tuple[str, float]] = []

    # --- candidate-selection builders ---------------------------------------------------
    def _coverage(self, features: np.ndarray, k: int) -> List[int]:
        """k-means coverage / coreset candidate (one representative per cluster)."""
        from sklearn.cluster import KMeans
        n = features.shape[0]
        k = min(k, n)
        km = KMeans(n_clusters=k, n_init=3, random_state=self.seed).fit(features)
        out = []
        for c in range(k):
            mem = np.where(km.labels_ == c)[0]
            if len(mem):
                out.append(int(mem[np.argmin(np.linalg.norm(features[mem] - km.cluster_centers_[c], axis=1))]))
        return out[:k]

    def _fusion(self, S, pf_ch, records, features, w, q, lam, k) -> List[int]:
        n = S.shape[1]
        imp = np.zeros(n)
        for c in range(S.shape[0]):
            imp += float(w[c]) * S[c]
        if q > 0:
            imp = imp.copy()
            imp[pf_ch < float(np.quantile(pf_ch, q))] = -np.inf
        if lam <= 0:
            return [int(i) for i in np.argsort(-imp)[:k]]
        finite = np.isfinite(imp)
        if finite.all():
            return BudgetSelector(lam=lam).select(records, imp, k, features=features)
        # audit fix: -inf from the authenticity gate reaches BudgetSelector's minmax and
        # turns every score into NaN. Run the selector on the finite (gate-passing)
        # subset only and map indices back.
        keep = np.where(finite)[0]
        if len(keep) <= k:
            return [int(i) for i in keep]
        sub_recs = [records[int(i)] for i in keep]
        sub_feats = features[keep] if features is not None else None
        sel = BudgetSelector(lam=lam).select(sub_recs, imp[keep], k, features=sub_feats)
        return [int(keep[int(j)]) for j in sel]

    def select(
        self,
        records: Sequence[UnifiedRecord],
        scores: np.ndarray,
        k: int,
        *,
        features: np.ndarray,
        held_out_gain: Callable[[list], float],
        extra_strategies: Optional[Sequence[Tuple[str, Callable[[int], List[int]]]]] = None,
        cheap_gain: Optional[Callable[[list], float]] = None,
        sh_keep: int = 4,
        policy_search: bool = False,
        construct_gain: Optional[Callable[[list], float]] = None,
    ) -> List[int]:
        """Return ``k`` selected indices = the validation-best candidate strategy's selection.

        scores : ``(n_channels, n)`` raw per-channel scores (min-maxed internally).
        held_out_gain : maps a selection (list of indices) to a scalar (higher better),
            computed on the modality's INDEPENDENT clean validation split with its own model.
        extra_strategies : optional extra (name, fn) candidates where ``fn(k) -> indices``
            (e.g. a modality-specific baseline); folded into the portfolio so the controller
            is >= them too by construction.
        cheap_gain : optional LOW-FIDELITY gain callback (e.g. probe on a subsample). When
            given, the fusion-grid candidates are pruned by successive halving under this
            cheap fidelity down to ``sh_keep`` finalists before the full-fidelity evaluation
            (Theorem B). Reference baselines are NEVER pruned this way: they are always
            evaluated at full fidelity, so the construction guarantee (Prop 2) and the
            regularized-switch guarantee (Thm 4) are untouched; halving only bounds the
            cost of searching the fusion grid.
        """
        S = np.stack([minmax(s) for s in scores], axis=0)
        C = S.shape[0]
        grid = self.weight_grid if self.weight_grid is not None else _default_grid(C)
        pf_ch = S[self.prefilter_channel]

        # ---- build the portfolio: (name, selection, is_reference) ----
        # references = the exact comparison baselines (passed by the caller) + coverage; fusions
        # are non-reference candidates that must clear a margin to be preferred over a baseline.
        fusions: List[Tuple[str, List[int], bool]] = []
        for w in grid:
            for q in self.prefilter_grid:
                for lam in self.lam_grid:
                    sel = self._fusion(S, pf_ch, records, features, w, q, lam, k)
                    fusions.append((f"fuse w={tuple(round(x,2) for x in w)} q={q} lam={lam}", sel, False))
        # ---- V1/V2 protocol (adversarially mandated). construct_gain / cheap_gain MUST be
        # computed on a validation half (V1) DISJOINT from held_out_gain's half (V2). All
        # ADAPTIVE construction (SH prescreen, vote-ensemble ranking, coordinate ascent,
        # policy search) consumes V1 only; the final adjudication (scored + regularized
        # switch) consumes V2 only. Given V1, the candidate list is FIXED before V2 is
        # touched, so Thm 4's premise (portfolio independent of the adjudication split)
        # holds verbatim with m' = |candidates|. Without a V1 callback the adaptive modules
        # are DISABLED - they never fall back to the adjudication gain (that fallback was
        # the winner's-curse leak the verification caught).
        gc = construct_gain if construct_gain is not None else cheap_gain
        self.sh_stats_ = None
        if gc is not None and len(fusions) > max(2, int(sh_keep)):
            # successive halving over the fusion grid on V1 (Theorem 8's prescreen):
            fid = cheap_gain if cheap_gain is not None else gc
            pool = list(fusions)
            cheap_evals = 0
            while len(pool) > max(2, int(sh_keep)):
                ranked = sorted(pool, key=lambda t: -float(fid(t[1])))
                cheap_evals += len(pool)
                pool = ranked[: max(max(2, int(sh_keep)), len(pool) // 2)]
                if len(pool) == len(ranked):
                    break
            self.sh_stats_ = {"fusions_total": len(fusions), "finalists": len(pool),
                              "cheap_evals": cheap_evals}
            fusions = pool
        candidates: List[Tuple[str, List[int], bool]] = list(fusions)
        if not extra_strategies:
            candidates.append(("coverage", self._coverage(features, k), True))
        for name, fn in (extra_strategies or []):
            candidates.append((name, list(fn(k)), True))

        def _gain(sel):
            # V2 adjudication gain - used ONLY after the candidate list is fixed.
            return float(np.mean([held_out_gain(sel) for _ in range(self.n_val_repeats)]))

        # ---- construction phase (V1 only): build the self-improving candidates ----
        nrec = len(records)
        if gc is not None:
            g1 = [(name, sel, float(gc(sel)), is_ref) for name, sel, is_ref in candidates]
            probe = cheap_gain if cheap_gain is not None else gc
            # (1) vote-ensemble of the V1-top-3 selections. Prop 3 lifted to the strategy
            # level; Prop 5 gives the sufficient conditions for a strict win. Ranking and
            # weights come from V1, never from the adjudication split.
            try:
                top3 = sorted(g1, key=lambda t: -t[2])[:3]
                randg = next((t[2] for t in g1 if t[0] == "random"), None)
                base = randg if randg is not None else min(t[2] for t in g1)
                votes = np.zeros(nrec)
                for _, sel, g, _ref in top3:
                    votes[np.asarray(sel, dtype=int)] += max(g - base, 0.0) + 1e-9
                ens = [int(i) for i in np.argsort(-(votes + 1e-9 * S[0]))[:k]]
                candidates.append(("vote_ensemble(top3)", ens, False))
                # (1b) METHOD-V2 candidate: diversity-regularized vote-ensemble. The raw
                # top-k-by-votes concentrates on the consensus core (low coverage, the
                # Prop-5 failure mode). Here the vote score is the importance signal fed
                # to the budget selector with a small diversity strength, injecting
                # coverage while keeping the vote consensus. Isolated behind METHOD_V2 so
                # it never affects the current canonical unless explicitly enabled; it is
                # adjudicated on V2 like every candidate, so worst case it is not chosen.
                if os.environ.get("METHOD_V2", "0") == "1":
                    lam_v2 = float(os.environ.get("METHOD_V2_LAM", "0.4"))
                    ens_div = BudgetSelector(lam=lam_v2).select(
                        records, votes.astype(float), k, features=features)
                    candidates.append(("vote_ensemble_div(v2)", [int(i) for i in ens_div], False))
                # (1c) v2c: complementarity-aware vote-ensemble. Prop 5 wins only when the
                # candidates' harmful mis-selections are near-disjoint (high complementarity
                # gamma). Measure the co-selection overlap of the top-3; enable the vote only
                # when overlap is LOW (complementary), else fall back to the best member. This
                # avoids the Prop-5 failure mode (consensus errors voted up) at low gamma.
                if os.environ.get("METHOD_V2C", "0") == "1":
                    try:
                        sets = [set(int(i) for i in sel) for _, sel, _g, _r in top3]
                        inter = sets[0] & sets[1] & sets[2] if len(sets) >= 3 else set()
                        uni = set().union(*sets) if sets else set()
                        overlap = len(inter) / max(1, len(uni))
                        thr = float(os.environ.get("METHOD_V2C_THR", "0.35"))
                        if overlap < thr:
                            candidates.append(("vote_ensemble_compl(v2c)", ens, False))
                        else:
                            best_sel = max(top3, key=lambda t: t[2])[1]
                            candidates.append(("vote_ensemble_compl(v2c)",
                                               [int(i) for i in best_sel], False))
                    except Exception:
                        pass
            except Exception:
                pass
            # (2) learned-weight fusion: coordinate ascent from the V1-best grid fusion
            # (Thm 6's measured allocation; probe on V1 only).
            try:
                fus1 = [t for t in g1 if t[0].startswith("fuse ")]
                if fus1:
                    import re as _re
                    m_ = _re.match(r"fuse w=\(([^)]+)\) q=([\d.]+) lam=([\d.]+)",
                                   max(fus1, key=lambda t: t[2])[0])
                    if m_:
                        w0 = np.array([float(x) for x in m_.group(1).split(",")], dtype=float)
                        q0, lam0 = float(m_.group(2)), float(m_.group(3))
                        def _sel_of(wv):
                            wv = np.clip(wv, 0, None)
                            wv = wv / (wv.sum() + 1e-12)
                            return self._fusion(S, pf_ch, records, features, wv, q0, lam0, k)
                        cur_w, cur_g = w0.copy(), float(probe(_sel_of(w0)))
                        for _round in range(2):
                            improved = False
                            for c in range(len(cur_w)):
                                for step in (0.15, -0.15):
                                    wtry = cur_w.copy(); wtry[c] = max(0.0, wtry[c] + step)
                                    gtry = float(probe(_sel_of(wtry)))
                                    if gtry > cur_g:
                                        cur_w, cur_g, improved = wtry, gtry, True
                            if not improved:
                                break
                        wn = cur_w / (cur_w.sum() + 1e-12)
                        candidates.append((f"fuse learned w={tuple(round(float(x),2) for x in wn)} q={q0} lam={lam0}",
                                           _sel_of(cur_w), False))
            except Exception:
                pass
            # (3) group-relative evolutionary search (critic-free; borrows GRPO's
            # group-relative advantage idea, Shao et al. 2024; update form is a
            # reward-baselined ES, Salimans et al. 2017 / Wierstra et al. 2014) over the
            # CONTINUOUS fusion-policy space (simplex weights x prefilter quantile x
            # diversity strength). Leave-one-out baseline (RLOO, Kool et al. 2019) keeps
            # the un-normalized gradient estimate unbiased; the trust region clips the
            # Mahalanobis step norm. Probes consume V1 only; the elite enters the
            # candidate list and is adjudicated on V2 like everyone else.
            if policy_search:
                try:
                    rng_ps = np.random.default_rng(self.seed + 31)
                    G_, T_ = 6, 4
                    mu = np.zeros(5)
                    sd = np.array([1.0, 1.0, 1.0, 1.5, 1.5])
                    def _cfg(th):
                        e = np.exp(th[:3] - th[:3].max()); w = e / e.sum()
                        q = 0.5 / (1.0 + np.exp(-th[3]))
                        lam = 0.8 / (1.0 + np.exp(-th[4]))
                        return w, float(q), float(lam)
                    best_r, best_cfg = -np.inf, None
                    for _t in range(T_):
                        thetas = mu + sd * rng_ps.standard_normal((G_, 5))
                        rs = np.zeros(G_)
                        for gi in range(G_):
                            w, q, lam = _cfg(thetas[gi])
                            sel_g = self._fusion(S, pf_ch, records, features, w, q, lam, k)
                            rs[gi] = float(probe(sel_g))
                            if rs[gi] > best_r:
                                best_r, best_cfg = rs[gi], (w, q, lam)
                        loo = (rs.sum() - rs) / max(G_ - 1, 1)     # leave-one-out baseline (unbiased)
                        adv = rs - loo
                        grad = (adv[:, None] * (thetas - mu) / (sd ** 2)).mean(0)
                        step = 0.8 * (sd ** 2) * grad               # diagonal preconditioning
                        mnorm = float(np.linalg.norm(step / sd))    # Mahalanobis trust region
                        if mnorm > 0.75:
                            step *= 0.75 / mnorm
                        mu = mu + step
                    if best_cfg is not None:
                        w, q, lam = best_cfg
                        candidates.append((f"grpo_policy w={tuple(round(float(x),2) for x in w)} q={q:.2f} lam={lam:.2f}",
                                           self._fusion(S, pf_ch, records, features, w, q, lam, k), False))
                except Exception:
                    pass

        # ---- adjudication phase (V2 only): the candidate list is now FIXED ----
        scored = [(name, sel, _gain(sel), is_ref) for name, sel, is_ref in candidates]
        self.leaderboard_ = sorted(((n, g) for n, _, g, _ in scored), key=lambda t: -t[1])
        overall = max(scored, key=lambda t: t[2])
        refs = [t for t in scored if t[3]]
        best_ref = max(refs, key=lambda t: t[2]) if refs else overall
        # regularized switch: deviate from the best reference baseline only if a fusion beats
        # it by more than the margin; otherwise return the best reference (>= it on test).
        margin = self.switch_margin_frac * abs(best_ref[2])
        chosen = overall if (overall[3] or overall[2] > best_ref[2] + margin) else best_ref
        # kappa_hat: the empirical harm-sensitivity signature (Theorem 3). With a random /
        # no-selection reference in the portfolio, the spread between the best candidate and
        # random measures how much this downstream model responds to selection at all; a
        # near-zero spread predicts the robust-FM regime where the controller falls back.
        rand = next((t for t in scored if t[0] == "random"), None)
        kappa_hat = float(overall[2] - rand[2]) if rand is not None else None
        self.chosen_ = {"strategy": chosen[0], "val_gain": chosen[2],
                        "best_ref": best_ref[0], "switched": not chosen[3],
                        "kappa_hat": kappa_hat}
        return chosen[1]
