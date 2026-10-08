"""Algorithm 1: build candidates, screen and synthesize on con, rank on rank, decide with a gate.

``adjudicate`` takes channel scores, executed reference strategies and utility callbacks per
split, and returns an ``Election`` with every candidate, its construction and ranking utilities,
the precheck result and the gate decision. Under the canonical configuration it reproduces the
2026-07 ``AdaptiveController`` (kept below as a compatibility wrapper with the same interface).
The callbacks hide the learner, so the same code serves every track.
"""
from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from omniselect.config.config import OmniSelectConfig
from omniselect.core.adjudication.precheck import PrecheckResult, headroom_precheck
from omniselect.core.adjudication.screening import kendall_tau, low_fidelity_halving, successive_halving
from omniselect.core.adjudication.synthesis import (
    consensus_complementary,
    consensus_diverse,
    consensus_votes,
    coordinate_ascent,
    policy_search,
)
from omniselect.core.gates import FamilyDecision, argmax_gate, margin_gate, run_family
from omniselect.core.gates.audit import audit_family, test_paired
from omniselect.core.gates.paired import paired_sample
from omniselect.core.gates.surrogate import UnitScores
from omniselect.core.selection.fusion_grid import (
    FusionGrid,
    build_cells,
    default_grid,
    fusion_select,
    parse_cell_name,
)
from omniselect.core.signals.base import minmax
from omniselect.tools.pairing import exact_kmeans_representatives, validate_selection
from omniselect.utils.hashing import sel_sha12

Gain = Callable[[list], float]
Units = Callable[[str, list], UnitScores]


@dataclass
class CandidateRecord:
    """One portfolio member as seen by the controller."""

    name: str
    selection: list[int]
    is_reference: bool
    stage: str                       # grid | finalist | reference | synthesized | screened_out
    u_con: Optional[float] = None
    u_rank: Optional[float] = None
    aliases: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)   # construction measurements (scores.json selection_info)

    @property
    def role(self) -> str:
        """reference, challenger or screened_out."""
        if self.stage == "screened_out":
            return "screened_out"
        return "reference" if self.is_reference else "challenger"


@dataclass
class Election:
    """Controller output: elected candidate plus the complete decision trace."""

    elected: str
    selection: list[int]
    reference: str
    decision: dict[str, Any]
    candidates: list[CandidateRecord]
    screening: dict[str, Any]
    precheck: Optional[PrecheckResult]
    construction_errors: list[dict[str, str]]
    timings: dict[str, float]

    def leaderboard(self) -> list[tuple[str, float]]:
        """(name, u_rank) sorted by u_rank descending, stable in candidate order."""
        rows = [(c.name, c.u_rank) for c in self.candidates if c.u_rank is not None]
        return sorted(rows, key=lambda t: -t[1])


def adjudicate(
    *,
    records: Sequence,
    scores: np.ndarray,
    features: np.ndarray,
    k: int,
    cfg: OmniSelectConfig,
    grid: Optional[FusionGrid],
    references: Sequence[Tuple[str, list]],
    gain_rank: Gain,
    gain_con: Optional[Gain] = None,
    cheap_gain: Optional[Gain] = None,
    screen_gain: Optional[Gain] = None,
    con_halves: Optional[Tuple[Gain, Gain]] = None,
    challengers: Sequence[Tuple[str, list]] = (),
    cooperative: Sequence[Tuple[str, list]] = (),
    units: Optional[Units] = None,
    full_selection: Optional[list] = None,
    random_name: str = "random",
    n_val_repeats: int = 1,
    seed: int = 0,
    order_cut: Optional[Callable[[list], list]] = None,
    similarity: Optional[Any] = None,
) -> Election:
    """Run Algorithm 1 and return the Election.

    scores: (channels, n) raw channel scores, min-max normalized here. Channel 0 is the
    prefilter unless ``grid.prefilter_channel`` says otherwise.
    references: executed reference-eligible strategies (name, k indices), in portfolio order.
    challengers: extra challenger-only strategies (name, k indices), added after the grid.
    cooperative: cooperative grid cells (name, selection) of protocol v2.2, challenger-only. They are
    deduplicated and screened on the construction split like the grid, keeping
    ``cooperative.screen_keep`` finalists, and enter after the grid finalists.
    gain_con, gain_rank: utility of a selection on the construction and ranking splits.
    screen_gain: low-fidelity construction utility used by ``screening.kind=low_fidelity``.
    con_halves: construction utilities on the fitting half and the checking half of V_con, used by
    ``synthesis.held_half``.
    units: per-unit bounded surrogate of a selection on a named split (statistical gates, precheck).
    full_selection: all pool indices, needed by the headroom precheck.
    order_cut: for budgets that are not a record count (text tokens), grid cells, the consensus
    vote and coordinate ascent build full orders (k = n) and ``order_cut`` truncates each order to
    the budget. Selections may then differ in size.
    similarity: a Similarity replacing the feature dot product of the diversity term (coverage.space).
    With ``synthesis.infomax_solver`` each screening finalist adds a challenger "<cell> solver=infomax"
    that solves the relaxed objective of the same cell (core/selection/relaxation.py).
    """
    timings: dict[str, float] = {}
    nrec = len(records)
    try:
        k = operator.index(k)
    except TypeError as exc:
        raise TypeError(f"k must be an integer, got {k!r}") from exc
    if not 0 <= k <= nrec:
        raise ValueError(f"k must be in [0, {nrec}], got {k}")
    raw = np.asarray(scores)
    if raw.ndim != 2 or raw.shape[1:] != (nrec,):
        raise ValueError(f"scores must have shape (channels, records), got {raw.shape} for {nrec} records")
    if raw.shape[0] == 0 or not np.isfinite(raw).all():
        raise ValueError("scores must contain at least one finite-valued channel")
    feats = np.asarray(features)
    if feats.ndim < 2 or feats.shape[0] != nrec or not np.isfinite(feats).all():
        raise ValueError(f"features must be a finite matrix with one row per record, got {feats.shape}")
    grid = grid if grid is not None else FusionGrid(weights=[], q_grid=(), lam_grid=())
    if not 0 <= grid.prefilter_channel < raw.shape[0]:
        raise ValueError(
            "prefilter_channel is outside the supplied score channels: "
            f"{grid.prefilter_channel} for {raw.shape[0]} channels"
        )
    S = np.stack([minmax(s) for s in raw], axis=0)
    prefilter = S[grid.prefilter_channel]
    errors: list[dict[str, str]] = []
    k_build = nrec if order_cut is not None else k

    def post(sel: list) -> list:
        return [int(i) for i in order_cut(list(sel))] if order_cut is not None else sel

    def check(sel: list, method: str) -> list:
        size = len(sel) if order_cut is not None else k
        return validate_selection(sel, pool_size=nrec, expected_size=size, method=method)

    def record_error(stage: str, error: Exception) -> None:
        errors.append({"stage": stage, "error_type": type(error).__name__, "message": str(error)})

    # ---- grid cells (challenger-only) ----
    t0 = time.perf_counter()
    dedupe = bool(cfg.cache.by_subset_hash)
    cell_list = build_cells(S, records, features, k_build, grid, dedupe=False,
                            similarity=similarity) if grid.weights != [] else []
    cell_list = [(name, post(sel)) for name, sel in cell_list]
    if dedupe:
        cell_list = _dedupe_cells(cell_list, grid)
    fusions = [CandidateRecord(name, sel, False, "grid", aliases=list(grid.aliases.get(name, [])))
               for name, sel in cell_list]
    timings["grid"] = time.perf_counter() - t0

    # ---- screening on con ----
    t0 = time.perf_counter()
    gc = gain_con if gain_con is not None else cheap_gain
    fusions, screened_out, screening = _screen(fusions, cfg.screening.sh_keep, cfg, gc, cheap_gain, screen_gain)
    timings["screening"] = time.perf_counter() - t0

    # ---- cooperative cells (challenger-only), screened on con like the grid ----
    coop_finalists: list[CandidateRecord] = []
    if cooperative:
        t0 = time.perf_counter()
        coop_cells = [(name, list(sel)) for name, sel in cooperative]
        coop_aliases: dict[str, list[str]] = {}
        if dedupe:
            coop_cells, coop_aliases = _dedupe(coop_cells)
        coop = [CandidateRecord(name, sel, False, "grid", aliases=list(coop_aliases.get(name, [])))
                for name, sel in coop_cells]
        coop_finalists, coop_out, screening["cooperative"] = _screen(coop, cfg.cooperative.screen_keep, cfg, gc,
                                                                     cheap_gain, screen_gain)
        screened_out = screened_out + coop_out
        timings["cooperative_screening"] = time.perf_counter() - t0

    # ---- candidate list: finalists (and their solver challengers), extra challengers, references ----
    candidates: list[CandidateRecord] = list(fusions)
    if cfg.synthesis.infomax_solver:
        t_solver = time.perf_counter()
        for c in list(fusions):
            try:
                sel = _solver_selection(c, S, prefilter, records, features, k_build, similarity, post, order_cut)
            except Exception as exc:  # noqa: BLE001 - recorded in the run record
                record_error(f"infomax solver {c.name}", exc)
                continue
            if sel is None:
                continue
            name = f"{c.name} solver=infomax"
            if sel_sha12(sel) == sel_sha12(c.selection):
                c.aliases.append(name)         # the relaxed solution equals the greedy cell (lam = 0)
                continue
            candidates.append(CandidateRecord(name, sel, False, "solver"))
        timings["solver"] = time.perf_counter() - t_solver
    candidates.extend(coop_finalists)
    for name, sel in challengers:
        candidates.append(CandidateRecord(name, list(sel), False, "challenger_member"))
    if not references and order_cut is None:
        cov = _coverage(features, k, seed)
        candidates.append(CandidateRecord("coverage", cov, True, "reference"))
    for name, sel in references:
        candidates.append(CandidateRecord(name, check(sel, f"controller reference {name}"), True, "reference"))

    # ---- construction on con: synthesized challengers ----
    t0 = time.perf_counter()
    if gc is not None:
        for c in candidates:
            c.u_con = float(gc(c.selection))
        scored_con = [(c.name, c.selection, c.u_con, c.is_reference) for c in candidates]
        probe = cheap_gain if cheap_gain is not None else gc
        synthesized: list[CandidateRecord] = []
        if cfg.synthesis.consensus:
            try:
                votes, ranked = consensus_votes(scored_con, nrec, top=cfg.synthesis.consensus_top,
                                                random_name=random_name)
                ens = post([int(i) for i in np.argsort(-(votes + 1e-9 * S[0]), kind="stable")[:k_build]])
                synthesized.append(CandidateRecord("vote_ensemble(top3)", ens, False, "synthesized"))
                if cfg.synthesis.consensus_diversity_lam > 0:
                    div = post(consensus_diverse(votes, records, features, k_build,
                                                 cfg.synthesis.consensus_diversity_lam))
                    synthesized.append(CandidateRecord("vote_ensemble_div(v2)", div, False, "synthesized"))
                if cfg.synthesis.consensus_complementarity_threshold > 0:
                    try:
                        comp = consensus_complementary(ranked, ens, cfg.synthesis.consensus_complementarity_threshold)
                        synthesized.append(CandidateRecord("vote_ensemble_compl(v2c)", comp, False, "synthesized"))
                    except Exception as exc:  # noqa: BLE001
                        record_error("vote_ensemble_complementarity", exc)
            except Exception as exc:  # noqa: BLE001 - recorded in the run record
                record_error("vote_ensemble", exc)
        if cfg.synthesis.coordinate_ascent:
            try:
                ca_probe, ca_check, ca_stats = probe, None, {}
                if cfg.synthesis.held_half:
                    if con_halves is None:
                        raise ValueError("synthesis.held_half needs the utilities of the two V_con halves")
                    ca_probe, ca_check = con_halves
                learned = coordinate_ascent(scored_con, S, records, features, k_build, prefilter, ca_probe,
                                            rounds=cfg.synthesis.ca_rounds, step=cfg.synthesis.ca_step,
                                            post=post if order_cut is not None else None, check=ca_check,
                                            stats=ca_stats, similarity=similarity)
                if learned is not None:
                    synthesized.append(CandidateRecord(learned[0], learned[1], False, "synthesized",
                                                       info=ca_stats if cfg.synthesis.held_half else {}))
            except Exception as exc:  # noqa: BLE001
                record_error("learned_weight_fusion", exc)
        if cfg.synthesis.policy_search:
            try:
                found = policy_search(S, records, features, k_build, prefilter, probe, seed)
                if found is not None:
                    synthesized.append(CandidateRecord(found[0], post(found[1]), False, "synthesized"))
            except Exception as exc:  # noqa: BLE001
                record_error("policy_search", exc)
        for c in synthesized:
            c.u_con = float(gc(c.selection)) if cfg.gate.pair_split == "con" else None
        candidates.extend(synthesized)
    timings["construction"] = time.perf_counter() - t0

    # ---- ranking on rank: the candidate list is fixed from here ----
    t0 = time.perf_counter()
    for c in candidates:
        c.selection = check(c.selection, f"controller candidate {c.name}")
    for c in candidates:
        values = [gain_rank(c.selection) for _ in range(max(1, int(n_val_repeats)))]
        value = float(np.mean(values))
        if not np.isfinite(value):
            raise ValueError("gain_rank (held_out_gain) returned a non-finite value")
        c.u_rank = value
    timings["ranking"] = time.perf_counter() - t0

    overall = max(candidates, key=lambda c: c.u_rank)
    refs = [c for c in candidates if c.is_reference]
    best_ref = max(refs, key=lambda c: c.u_rank) if refs else overall
    by_name = {c.name: c for c in candidates}

    # ---- headroom precheck ----
    precheck: Optional[PrecheckResult] = None
    if cfg.precheck.enabled:
        if units is None or full_selection is None or random_name not in by_name:
            record_error("precheck", ValueError("precheck needs units, the full pool and a random candidate"))
        else:
            precheck = headroom_precheck(
                units("rank", list(full_selection)),
                units("rank", by_name[random_name].selection),
                delta=cfg.gate.delta,
                multiplier=cfg.precheck.threshold_multiplier,
                decide=cfg.precheck.decide,
                threshold_kind=cfg.precheck.threshold_kind,
                relative_threshold=cfg.precheck.threshold,
                best_reference=units("rank", best_ref.selection),
            )

    # ---- decision ----
    t0 = time.perf_counter()
    if precheck is not None and precheck.skipped:
        family = FamilyDecision("precheck", random_name, random_name, False, 0, [])
    else:
        family = _decide(cfg, candidates, best_ref, overall, units, seed)
    audit: Optional[dict[str, Any]] = None
    if cfg.gate.audit:
        if units is None:
            record_error("audit", ValueError("gate.audit needs per-unit scores"))
        else:
            try:
                audit = _audit(cfg, candidates, best_ref, units, seed)
            except Exception as exc:  # noqa: BLE001 - recorded in the run record, the election stands
                record_error("audit", exc)
    timings["confirmation"] = time.perf_counter() - t0
    elected = by_name[family.elected]
    rand = by_name.get(random_name)
    decision = {
        "gate": family.to_dict(),
        "elected": elected.name,
        "elected_sel_sha12": sel_sha12(elected.selection),
        "reference": family.reference,
        "rank_best_reference": best_ref.name,
        "overall_rank_best": overall.name,
        "adopted": bool(family.adopted),
        "switched": not elected.is_reference,
        "kappa_hat": float(overall.u_rank - rand.u_rank) if rand is not None else None,
        "u_rank_elected": elected.u_rank,
    }
    if audit is not None:
        decision["audit"] = audit
    all_candidates = list(screened_out) + candidates
    return Election(
        elected=elected.name,
        selection=elected.selection,
        reference=family.reference,
        decision=decision,
        candidates=all_candidates,
        screening=screening,
        precheck=precheck,
        construction_errors=errors,
        timings=timings,
    )


def _solver_selection(c: CandidateRecord, S: np.ndarray, prefilter: np.ndarray, records: Sequence,
                      features: np.ndarray, k_build: int, similarity, post, order_cut) -> Optional[list[int]]:
    """Selection of the relaxed objective of grid cell ``c`` (same weights, q, lam and similarity).

    With a token budget (``order_cut``) the relaxation selects as many records as the greedy cell kept,
    the remaining records follow in the order of the fused importance, and ``post`` cuts the order.
    """
    parsed = parse_cell_name(c.name)
    if parsed is None:
        return None
    w, q, lam = parsed
    if order_cut is None:
        return fusion_select(S, prefilter, records, features, w, q, lam, k_build, similarity=similarity,
                             method="infomax")
    top = fusion_select(S, prefilter, records, features, w, q, lam, len(c.selection), similarity=similarity,
                        method="infomax")
    imp = np.asarray(w, dtype=float) @ S
    chosen = set(int(i) for i in top)
    rest = [int(i) for i in np.argsort(-imp, kind="stable") if int(i) not in chosen]
    return post([int(i) for i in top] + rest)


def _dedupe_cells(cells: list[tuple[str, list]], grid: FusionGrid) -> list[tuple[str, list]]:
    """Keep the first cell of each distinct sorted subset. Record the others as aliases in ``grid``."""
    kept, grid.aliases = _dedupe(cells)
    return kept


def _dedupe(cells: list[tuple[str, list]]) -> tuple[list[tuple[str, list]], dict[str, list[str]]]:
    """The first cell of each distinct sorted subset, and the names merged into each kept cell."""
    first: dict[str, str] = {}
    kept: list[tuple[str, list]] = []
    aliases: dict[str, list[str]] = {}
    for name, sel in cells:
        sha = sel_sha12(sel)
        if sha in first:
            aliases.setdefault(first[sha], []).append(name)
            continue
        first[sha] = name
        kept.append((name, sel))
    return kept, aliases


def _screen(cands: list[CandidateRecord], keep: int, cfg: OmniSelectConfig, gc: Optional[Gain],
            cheap_gain: Optional[Gain], screen_gain: Optional[Gain]
            ) -> tuple[list[CandidateRecord], list[CandidateRecord], dict[str, Any]]:
    """Successive halving of ``cands`` on the construction split down to max(2, keep) finalists.

    Returns (finalists, screened-out candidates, trace). Nothing is screened when
    ``screening.kind=off``, without a construction utility, or with at most ``keep`` candidates.
    ``low_fidelity`` screens on ``screen_gain`` and records the full-fidelity utility of the finalists.
    """
    screening: dict[str, Any] = {"kind": cfg.screening.kind, "applied": False, "total": len(cands)}
    keep = max(2, int(keep))
    if cfg.screening.kind == "off" or gc is None or len(cands) <= keep:
        return cands, [], screening
    cells = [(c.name, c.selection, c) for c in cands]
    if cfg.screening.kind == "low_fidelity":
        if screen_gain is None:
            raise ValueError("screening.kind=low_fidelity needs a screen_gain callback")
        pool, trace, full = low_fidelity_halving(cells, screen_gain, gc, keep)
        names = [cell[0] for cell in pool]
        last_low = {name: value for round_ in trace.rounds for name, value in round_}
        screening.update({
            "fidelity": "low",
            "subsample": float(cfg.screening.low_fidelity_fraction),
            "finalists_full_u_con": full,
            "kendall_tau_finalists": kendall_tau([last_low[n] for n in names], [full[n] for n in names]),
        })
    else:
        fid = cheap_gain if cheap_gain is not None else gc
        pool, trace = successive_halving(cells, fid, keep)
    finalists = [cell[2] for cell in pool]
    kept_names = {c.name for c in finalists}
    screened_out = [c for c in cands if c.name not in kept_names]
    last_value = {name: value for round_ in trace.rounds for name, value in round_}
    for c in screened_out:
        c.stage = "screened_out"
        c.u_con = last_value.get(c.name)
    for c in finalists:
        c.stage = "finalist"
    screening.update({"applied": True, "finalists": trace.finalists, "evaluations": trace.evaluations,
                      "rounds": trace.rounds, "screened_out": trace.screened_out})
    return finalists, screened_out, screening


def _decide(
    cfg: OmniSelectConfig,
    candidates: list[CandidateRecord],
    best_ref: CandidateRecord,
    overall: CandidateRecord,
    units: Optional[Units],
    seed: int,
) -> FamilyDecision:
    """Apply cfg.gate to the reference and the ordered challenger family."""
    gate = cfg.gate
    if gate.kind in ("margin", "argmax"):
        # The top-K ranking challengers are all tested and recorded. Their order is by u_rank and the
        # threshold depends on the reference only, so the first that passes is the best challenger
        # whenever any passes: the election equals the comparison of the best reference with the
        # overall ranking argmax.
        ranked = sorted((c for c in candidates if not c.is_reference and c.u_rank is not None),
                        key=lambda c: -c.u_rank)[: gate.k_challengers]
        by = {c.name: c for c in ranked}

        def rank_test(name: str, k_family: int):
            chal = by[name]
            if gate.kind == "margin":
                return margin_gate(best_ref.name, best_ref.u_rank, name, chal.u_rank, gate.margin_frac)
            return argmax_gate(best_ref.name, best_ref.u_rank, name, chal.u_rank)

        return run_family(gate.kind, best_ref.name, [c.name for c in ranked], rank_test, test_all=True)
    if units is None:
        raise ValueError(f"gate {gate.kind} needs per-unit scores")
    key = (lambda c: c.u_con) if gate.pair_split == "con" else (lambda c: c.u_rank)
    refs = [c for c in candidates if c.is_reference]
    reference = _first_max(refs, key) if refs else best_ref
    pool = [c for c in candidates if not c.is_reference and key(c) is not None]
    ordered = sorted(pool, key=lambda c: -key(c))[: gate.k_challengers]
    ref_units = units(gate.split, reference.selection)
    clip = gate.text_clip if ref_units.kind == "text_neg_nll" else None

    def test_one(name: str, k_family: int):
        chal = next(c for c in ordered if c.name == name)
        sample = paired_sample(ref_units, units(gate.split, chal.selection), clip=clip)
        return test_paired(gate.kind, reference.name, name, sample, k=k_family, seed=seed, delta=gate.delta,
                           eps=gate.eps, n_boot=gate.n_boot, p_beat_min=gate.p_beat_min, order=gate.reading_order)

    return run_family(gate.kind, reference.name, [c.name for c in ordered], test_one,
                      test_all=gate.kind in ("bootstrap", "eprocess"))


def _audit(
    cfg: OmniSelectConfig,
    candidates: list[CandidateRecord],
    best_ref: CandidateRecord,
    units: Units,
    seed: int,
) -> dict[str, Any]:
    """Audit of the best ranking reference against the top-K ranking challengers on gate.audit_split."""
    gate = cfg.gate
    ordered = sorted((c for c in candidates if not c.is_reference and c.u_rank is not None),
                     key=lambda c: -c.u_rank)[: gate.k_challengers]
    ref_units = units(gate.audit_split, best_ref.selection)
    if len(ref_units.values) == 0:
        return {"split": gate.audit_split, "skipped": "the audit split has no units"}
    clip = gate.text_clip if ref_units.kind == "text_neg_nll" else None
    samples = [(c.name, paired_sample(ref_units, units(gate.audit_split, c.selection), clip=clip)) for c in ordered]
    return audit_family(best_ref.name, samples, k=gate.k_challengers, seed=seed, delta=gate.delta, eps=gate.eps,
                        n_boot=gate.n_boot, order=gate.reading_order, split=gate.audit_split)


def _first_max(items: Sequence[CandidateRecord], key: Callable[[CandidateRecord], Any]) -> CandidateRecord:
    """First item with the largest key (an exact tie keeps the earlier item)."""
    best = items[0]
    for item in items[1:]:
        if key(item) is not None and (key(best) is None or key(item) > key(best)):
            best = item
    return best


def _coverage(features: np.ndarray, k: int, seed: int) -> list[int]:
    """k-means coverage candidate: the record nearest each of k centroids."""
    from sklearn.cluster import KMeans

    n = features.shape[0]
    k = min(k, n)
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(features)
    return exact_kmeans_representatives(features, km.labels_, km.cluster_centers_, k)


class AdaptiveController:
    """Canonical controller interface of the 2026-07 runners, implemented by ``adjudicate``.

    ``select`` returns the elected selection and sets ``chosen_``, ``leaderboard_``,
    ``sh_stats_`` and ``construction_errors_`` as before. The switch margin is
    ``switch_margin_frac`` times |u_rank(best reference)|.
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
        self.switch_margin_frac = float(switch_margin_frac)
        self.seed = int(seed)
        self.chosen_: Optional[dict] = None
        self.leaderboard_: List[Tuple[str, float]] = []
        self.election_: Optional[Election] = None

    def select(
        self,
        records: Sequence,
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
        """Return the elected k indices. ``held_out_gain`` scores the ranking split."""
        raw = np.asarray(scores)
        if raw.ndim == 2 and not 0 <= self.prefilter_channel < raw.shape[0]:
            raise ValueError(
                "prefilter_channel is outside the supplied score channels: "
                f"{self.prefilter_channel} for {raw.shape[0]} channels"
            )
        cfg = OmniSelectConfig.preset("canonical")
        cfg.gate.margin_frac = self.switch_margin_frac
        cfg.screening.sh_keep = int(sh_keep)
        cfg.synthesis.policy_search = bool(policy_search)
        n_channels = raw.shape[0] if raw.ndim == 2 else 3
        grid = FusionGrid(
            weights=[tuple(w) for w in self.weight_grid] if self.weight_grid is not None else default_grid(n_channels),
            q_grid=self.prefilter_grid,
            lam_grid=self.lam_grid,
            prefilter_channel=self.prefilter_channel,
        )
        references = [(name, fn(k)) for name, fn in (extra_strategies or [])]
        election = adjudicate(
            records=records, scores=scores, features=features, k=k, cfg=cfg, grid=grid,
            references=references, gain_rank=held_out_gain, gain_con=construct_gain,
            cheap_gain=cheap_gain, n_val_repeats=self.n_val_repeats, seed=self.seed,
        )
        self.election_ = election
        self.construction_errors_ = list(election.construction_errors)
        self.leaderboard_ = election.leaderboard()
        self.sh_stats_ = None
        if election.screening.get("applied"):
            self.sh_stats_ = {"fusions_total": election.screening["total"],
                              "finalists": election.screening["finalists"],
                              "cheap_evals": election.screening["evaluations"]}
        elected = next(c for c in election.candidates if c.name == election.elected)
        self.chosen_ = {
            "strategy": elected.name,
            "val_gain": elected.u_rank,
            "best_ref": election.reference,
            "switched": not elected.is_reference,
            "kappa_hat": election.decision["kappa_hat"],
            "construction_error_count": len(election.construction_errors),
        }
        return list(election.selection)


__all__ = ["AdaptiveController", "CandidateRecord", "Election", "adjudicate", "fusion_select"]
