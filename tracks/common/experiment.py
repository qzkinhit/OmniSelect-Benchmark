"""The one driver every track calls: load, signals, strategies, adjudication, run record.

``run_cell`` executes one (batch, track, dataset, learner, seed, protocol) cell. It computes
every strategy of the portfolio and the method rows under the paired RNG, scores candidates
through the train-once cache, runs omniselect's controller, fits every candidate at the
reported schedule, and writes the run record of docs/RUN_RECORD.md. A cell whose decision.json
exists is skipped. ``main`` is the command-line entry point.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from omniselect.config.config import PROTOCOLS, OmniSelectConfig, parse_overrides
from omniselect.core.adjudication.cache import FitCache
from omniselect.core.adjudication.controller import adjudicate
from omniselect.core.gates.surrogate import unit_scores
from omniselect.core.portfolio import registry
from omniselect.core.portfolio.membership import (
    DIAGNOSTIC_ONLY,
    challenger_members,
    membership,
    reference_members,
)
from omniselect.core.portfolio.registry import SelectionContext
from omniselect.core.selection.cooperative import CooperativeFamily, neighbour_graph
from omniselect.core.selection.cooperative import cell_name as coop_cell_name
from omniselect.store.metrics import standard_metrics, utility
from omniselect.store.run_record import RunRecord
from omniselect.tools.pairing import validate_selection
from omniselect.utils.hashing import sel_sha12
from tracks.common.pairing import reset_for_strategy
from tracks.common.task import Track, TrackConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKS: dict[str, str] = {
    "vision": "tracks.vision.clip_probe:VisionTrack",
    "native": "tracks.vision.native_resnet:NativeTrack",
    "timeseries": "tracks.timeseries.dlinear:ForecastTrack",
    "process": "tracks.process.tep_mlp:ProcessTrack",
    "tabular": "tracks.tabular.tabpfn:TabularTrack",
    "text": "tracks.text.smollm_finetune:TextTrack",
}
REPORT_SPLITS = ("con", "rank", "conf", "test")
# Tracks whose load() reads each robustness setting. Other combinations raise before the cell starts.
ROBUSTNESS_SUPPORT: dict[str, tuple[str, ...]] = {
    "mechanism_set=unseen": ("vision", "timeseries", "process", "tabular"),
    "val_noise_kind=symmetric": ("vision", "process", "tabular"),
    "val_noise_kind=natural": ("vision",),
    "val_noise_kind=prior_shift": ("vision", "process", "tabular"),
    "injection_ratio": ("vision", "timeseries", "process", "tabular"),
}
ROBUSTNESS_REASONS: dict[str, str] = {
    "timeseries": "forecasting validation targets are real-valued windows, label noise is undefined",
    "text": "the text pool carries its corruption in the registered files and held-out records have no label",
    "native": "the native ResNet-18 protocol injects no corruption and keeps the dataset labels",
}


def check_robustness(track_name: str, ocfg: OmniSelectConfig) -> None:
    """Raise NotImplementedError for a robustness setting that ``track_name`` does not read."""
    rb = ocfg.robustness
    wanted = []
    if rb.mechanism_set != "canonical":
        wanted.append(f"mechanism_set={rb.mechanism_set}")
    if rb.val_noise_kind != "none" and (rb.val_noise_rate > 0 or rb.val_noise_kind == "natural"):
        wanted.append(f"val_noise_kind={rb.val_noise_kind}")
    if rb.injection_ratio >= 0:
        wanted.append("injection_ratio")
    for key in wanted:
        if track_name not in ROBUSTNESS_SUPPORT.get(key, ()):
            reason = ROBUSTNESS_REASONS.get(track_name, "not read by this track")
            raise NotImplementedError(f"robustness {key} is not available on track {track_name}: {reason}")


class SkipCell(RuntimeError):
    """Raised by a track when a required model or dataset is unavailable."""


def load_track(name: str) -> Track:
    """Instantiate the Track class registered under ``name``."""
    if name not in TRACKS:
        raise KeyError(f"unknown track {name!r}; known: {sorted(TRACKS)}")
    module_name, cls_name = TRACKS[name].split(":")
    return getattr(importlib.import_module(module_name), cls_name)()


def resolve_configs(track: Track, dataset: str, *, learner: Optional[str], seed: int, protocol: str,
                    smoke: bool, track_overrides: dict[str, Any], omni_overrides: dict[str, Any]
                    ) -> tuple[TrackConfig, OmniSelectConfig]:
    """TrackConfig and OmniSelectConfig of one cell (preset, then track overrides, then CLI)."""
    tcfg = track.config(dataset, learner=learner, seed=seed, smoke=smoke, **track_overrides)
    merged = {**track.omni_overrides(protocol, tcfg), **omni_overrides, "seed": seed}
    ocfg = OmniSelectConfig.preset(protocol, **merged)
    return tcfg, ocfg


def purity(tags: np.ndarray, selection: list[int]) -> float:
    """Fraction of selected records tagged 'high'."""
    if len(selection) == 0:
        return float("nan")
    return float(np.mean(np.asarray(tags)[np.asarray(selection, dtype=int)] == "high"))


def run_cell(track: Track, tcfg: TrackConfig, ocfg: OmniSelectConfig, *, out_root: Path, batch: str,
             cli: Optional[list[str]] = None, overwrite: bool = False, standalone: bool = False) -> dict[str, Any]:
    """Run one cell and return a summary dict (status, elected, test utility, cell path).

    ``standalone`` runs only the method rows (no portfolio, no controller), as the per-method
    runners under benchmark/MethodsRunScript do.
    """
    check_robustness(track.name, ocfg)
    cell = RunRecord.cell_path(out_root, batch, track.name, tcfg.dataset, tcfg.learner, tcfg.seed)
    record = RunRecord(cell, repo_root=REPO_ROOT)
    if record.is_complete() and not overwrite:
        print(f"[skip] {cell} has decision.json")
        return {"status": "skipped_complete", "cell": str(cell)}
    record.prepare(overwrite=overwrite)
    t_start = time.perf_counter()
    record.write_config(omniselect=ocfg.to_dict(), track=tcfg.to_dict(), cli=cli or sys.argv,
                        extra={"protocol": ocfg.protocol, "batch": batch})
    record.log(f"cell {track.name}/{tcfg.dataset}/{tcfg.learner}/seed_{tcfg.seed} protocol={ocfg.protocol}")
    timings: dict[str, Any] = {}

    t0 = time.perf_counter()
    data = track.load(tcfg, ocfg)
    timings["load"] = time.perf_counter() - t0
    split_ids = {
        "pool": data.pool_ids,
        "con": [data.val_ids[i] for i in data.splits.con],
        "rank": [data.val_ids[i] for i in data.splits.rank],
        "conf": [data.val_ids[i] for i in data.splits.conf],
        "test": data.test_ids,
    }
    record.write_splits(ids=split_ids, pool_tags=[str(t) for t in data.tags], blocks=data.blocks,
                        pool_mechanisms=data.mechanisms,
                        extra={"budget": data.budget, "utility": data.utility, "s0": data.s0,
                               "provenance": data.provenance})
    record.log(f"pool {data.n} budget {data.budget} splits {data.splits.sizes()} test {len(data.test_ids)}")
    if data.provenance.get("encoding") is not None:
        record.add_config_fields({"encoding": data.provenance["encoding"]})

    member_rows = membership(track.name, ocfg.portfolio.membership)
    ref_names = reference_members(track.name, ocfg.portfolio.membership)
    chal_names = challenger_members(track.name, ocfg.portfolio.membership)
    method_rows = [m for m in tcfg.methods if m not in ("full", "mmds_adapt")]
    if standalone:
        ref_names, chal_names = [], []
    wanted = list(dict.fromkeys(ref_names + chal_names + method_rows))

    t0 = time.perf_counter()
    sig = track.signals(data, tcfg, ocfg)
    timings["signals"] = time.perf_counter() - t0
    channel_info: dict[str, Any] = {}
    if ocfg.signals.alignment:
        t0 = time.perf_counter()
        aligned = track.alignment(data, sig, tcfg, ocfg)
        timings["alignment"] = time.perf_counter() - t0
        channel_info = {"seconds": {"authenticity_influence_redundancy": timings["signals"],
                                    "alignment": timings["alignment"]},
                        "alignment_source": None if aligned is None else aligned["source"]}
        if aligned is not None:
            sig.alignment = np.asarray(aligned["scores"], dtype=float)
            sig.saved["alignment"] = sig.alignment
            sig.extras["gradient_factors"] = (aligned.get("phi"), aligned.get("err"))
            record.log(f"alignment channel from {aligned['source']} ({timings['alignment']:.1f}s)")
        else:
            record.log("alignment channel undefined on this track")
    family = None
    coop_grid = tcfg.cooperative_grid() if (ocfg.cooperative.enabled and not standalone) else []
    if coop_grid or {"coop_herding", "clean_top"} & set(wanted):
        t0 = time.perf_counter()
        family, clean_info = cooperative_family(track, data, sig, tcfg, ocfg, coop_grid, wanted)
        timings["cooperative_signals"] = time.perf_counter() - t0
        if clean_info is not None:
            channel_info["cleanliness"] = clean_info
            record.log(f"cleanliness score: cross-validated AUC of V_con against the pool {clean_info['cv_auc']} "
                       f"({clean_info['seconds']:.1f}s)")
    record.write_signals({"authenticity": sig.auth, "influence": sig.influence, "redundancy": sig.redundancy,
                          **sig.saved}, reference_ids=sig.reference_ids, reference_source=sig.reference_source)

    if channel_info:
        record.write_json("signals/timings.json", channel_info)
    learner = track.learner(data, tcfg, ocfg)
    available = [s for s in REPORT_SPLITS if s == "test" or len(data.splits.get(s)) > 0]
    eager = tuple(available) if not learner.keep_models else ()
    cache = FitCache(learner, enabled=ocfg.cache.by_subset_hash, eager_splits=eager,
                     sort_subsets=track.sorted_training)
    sort_scoring = track.sorted_training and (ocfg.cache.by_subset_hash or
                                              (track.canonical_sorted_scoring and ocfg.paired_rng))
    sort_report = track.sorted_training and ocfg.paired_rng
    scoring_units: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    def evaluate(sel: list[int], stage: str, split: str):
        """Per-unit outputs of the ``stage`` fit of ``sel`` on ``split``, through the fit cache."""
        order = sorted(int(i) for i in sel) if sort_scoring else [int(i) for i in sel]
        return cache.evaluate(order, stage, [split]).per_unit[split]

    def gain(split: str, stage: str) -> Callable[[list[int]], float]:
        """Utility callback of ``split`` for the controller. It keeps the per-unit outputs it scored."""
        def g(sel: list[int]) -> float:
            pu = evaluate(sel, stage, split)
            scoring_units[(sel_sha12(sel), stage)] = pu
            return utility(data.utility, pu)
        return g

    def units(split: str, sel: list[int]):
        """Bounded per-unit surrogate of ``sel`` on ``split``, read by the statistical gates and the audit."""
        pu = evaluate(sel, split, split)
        return unit_scores(data.utility, pu, s0=data.s0, block_steps=data.block_steps)

    halves_cache: dict[str, Any] = {}

    def con_half(which: int) -> Callable[[list[int]], float]:
        """Utility of a selection on one half of V_con (0 fits, 1 checks), from the con-stage fit."""
        def g(sel: list[int]) -> float:
            pu = evaluate(sel, "con", "con")
            if "idx" not in halves_cache:
                halves_cache["idx"] = held_halves(pu, tcfg.seed)
                halves_cache["sizes"] = [int(len(h)) for h in halves_cache["idx"]]
            return utility(data.utility, slice_units(pu, halves_cache["idx"][which]))
        return g

    def screen_gain(sel: list[int]) -> float:
        """Construction utility of a fit at the screen stage on a seeded subsample of ``sel``."""
        sub = screening_subsample(sel, ocfg.screening.low_fidelity_fraction, tcfg.seed)
        pu = cache.evaluate(sorted(sub) if sort_scoring else sub, "screen", ["con"], eager=False).per_unit["con"]
        return utility(data.utility, pu)

    has_con = len(data.splits.con) > 0
    ctx = SelectionContext(
        n=data.n, seed=tcfg.seed, features=sig.features, auth=sig.auth, influence=sig.influence,
        redundancy=sig.redundancy, labels=data.labels, proba=sig.proba, records=data.records,
        construction_gain=gain("con", "member") if has_con else None,
        extras={**sig.extras, "auth_q": tcfg.auth_q, "lam": tcfg.lam, "dsdm_runs": tcfg.dsdm_runs,
                "dmf_rounds": tcfg.dmf_rounds, "fixed_fusion_gate": ocfg.fixes.fixed_fusion_gate,
                "density_seed": ocfg.fixes.density_seed, "budget_cut": data.budget_cut, "tags": data.tags,
                "strategy_info": {}, "selection_device": tcfg.selection_device, "alignment": sig.alignment,
                "cooperative": family, "cleanliness": sig.saved.get("cleanliness")},
    )
    overrides = track.strategy_overrides(data, sig, tcfg, ocfg)

    selections: dict[str, list[int]] = {}
    selection_secs: dict[str, float] = {}
    skipped: dict[str, str] = {}
    registry.ensure_loaded()
    k = data.budget
    t0 = time.perf_counter()
    for name in wanted:
        fn = overrides.get(name)
        if fn is None:
            if name not in registry.REGISTRY:
                skipped[name] = "no implementation registered"
                continue
            strategy = registry.REGISTRY[name]
            missing = registry.missing_inputs(strategy, ctx)
            if missing:
                skipped[name] = f"missing inputs {missing}"
                continue
            fn = strategy.fn
        reset_for_strategy(tcfg.seed, name, ocfg.paired_rng)
        s0 = time.perf_counter()
        try:
            sel = fn(ctx, k)
            if data.budget_cut is not None and len(sel) == data.n:
                sel = data.budget_cut(list(sel))
                selections[name] = [int(i) for i in sel]
            else:
                selections[name] = validate_selection(sel, pool_size=data.n, expected_size=k, method=name)
        except NotImplementedError as exc:
            skipped[name] = f"not implemented: {exc}"
            continue
        selection_secs[name] = time.perf_counter() - s0
        if name in ctx.extras["strategy_info"]:
            ctx.extras["strategy_info"][name]["selection_secs"] = selection_secs[name]
        record.log(f"strategy {name:22s} n={len(selections[name])} sel={sel_sha12(selections[name])} "
                   f"({selection_secs[name]:.2f}s)")
    timings["member_selection"] = time.perf_counter() - t0
    for name, reason in skipped.items():
        record.log(f"strategy {name} skipped: {reason}")

    coop_cells: list[tuple[str, list[int]]] = []
    t0 = time.perf_counter()
    for score_name, rho, selector in coop_grid:
        name = coop_cell_name(score_name, rho, selector)
        sel, info = family.select(score_name, rho, selector, k, name=name)
        sel = data.budget_cut(sel) if data.budget_cut is not None else validate_selection(
            sel, pool_size=data.n, expected_size=k, method=name)
        coop_cells.append((name, [int(i) for i in sel]))
        ctx.extras["strategy_info"][name] = info
        selection_secs[name] = info["seconds"]
        record.log(f"cooperative {name:44s} n={len(sel)} admissible={info['admissible_size']} "
                   f"outliers={info['outlier_drops']} duplicates={info['duplicate_drops']} refill={info['refill']}")
    if coop_grid:
        timings["cooperative_grid"] = time.perf_counter() - t0
    if family is not None:
        for name, kept in family.admissible_sets.items():   # analysis only: tags are read after every selection
            if name in ctx.extras["strategy_info"]:
                ctx.extras["strategy_info"][name]["admissible_purity"] = purity(data.tags, list(kept))

    references = [(n, selections[n]) for n in ref_names if n in selections]
    challengers = [(n, selections[n]) for n in chal_names if n in selections]
    full = list(range(data.n))
    base_protocol = "v2" if ocfg.protocol in ("v2_1", "v2_2") else ocfg.protocol
    grid = tcfg.fusion_grid() if (tcfg.grid_weights != () and base_protocol in tcfg.grid_protocols) else None
    if grid is not None and sig.alignment is not None:
        from omniselect.core.selection.fusion_grid import default_grid, extend_grid

        grid.weights = extend_grid(grid.weights if grid.weights is not None else default_grid(3))
    similarity = None
    if ocfg.coverage.space != "feature":
        from omniselect.core.selection.similarity import Similarity

        phi, err = sig.extras.get("gradient_factors", (None, None))
        if phi is None:
            raise ValueError(f"coverage.space={ocfg.coverage.space} needs the gradient factors of the alignment "
                             "channel (signals.alignment=true on a track that defines it)")
        similarity = Similarity.build(ocfg.coverage.space, sig.features, phi, err)
    election = None
    t0 = time.perf_counter()
    if not standalone:
        needs_units = ocfg.gate.kind in ("lcb", "bootstrap", "eprocess") or ocfg.precheck.enabled or ocfg.gate.audit
        election = adjudicate(
            records=data.records, scores=sig.channels(), features=sig.features,
            k=k if data.budget_cut is None else data.n, cfg=ocfg, grid=grid,
            references=references, challengers=challengers, cooperative=coop_cells, gain_rank=gain("rank", "rank"),
            gain_con=gain("con", "con") if has_con else None, units=units if needs_units else None,
            screen_gain=screen_gain if ocfg.screening.kind == "low_fidelity" else None,
            con_halves=(con_half(0), con_half(1)) if (ocfg.synthesis.held_half and has_con) else None,
            full_selection=full, random_name="random", seed=tcfg.seed, order_cut=data.budget_cut,
            similarity=similarity,
        )
        timings["controller"] = time.perf_counter() - t0
        timings.update({f"controller_{k2}": v for k2, v in election.timings.items()})
        record.log(f"elected {election.elected} (reference {election.reference}, "
                   f"gate {election.decision['gate']['kind']}, adopted {election.decision['adopted']})")

    # ---- reported fit of every candidate, method row, full and the elected subset ----
    t0 = time.perf_counter()
    candidates = {c.name: c for c in election.candidates} if election is not None else {}
    rows: dict[str, dict[str, Any]] = {}
    to_report: list[tuple[str, list[int], str, str]] = []
    for c in (election.candidates if election is not None else []):
        if c.stage == "screened_out" and not ocfg.report_all_candidates:
            continue
        to_report.append((c.name, c.selection, c.role, c.stage))
    for name in method_rows:
        if name in selections and name not in candidates:
            role = "diagnostic" if name in DIAGNOSTIC_ONLY else "baseline_row"
            to_report.append((name, selections[name], role, "method_row"))
    if "full" in tcfg.methods or (ocfg.precheck.enabled and not standalone):
        to_report.append(("full", full, "baseline_row", "full"))
    if not ocfg.report_all_candidates:
        keep = set(method_rows) | {"full"} | ({election.elected} if election is not None else set())
        to_report = [row for row in to_report if row[0] in keep]
    candidate_timing: dict[str, Any] = {}
    for name, sel, role, stage in to_report:
        order = sorted(int(i) for i in sel) if sort_report else [int(i) for i in sel]
        entry = cache.evaluate(order, "report", available)
        per_unit = {s: entry.per_unit[s] for s in available}
        c = candidates.get(name)
        scoring: dict[str, Any] = {}
        scoring_pu = {}
        for stage_name in ("con", "rank"):
            pu = scoring_units.get((sel_sha12(sel), stage_name))
            if pu is not None and learner.fidelity(stage_name) != learner.fidelity("report"):
                scoring_pu[stage_name] = pu
        if c is not None:
            scoring = {"u_con": c.u_con, "u_rank": c.u_rank, "aliases": c.aliases}
            if c.info:
                ctx.extras["strategy_info"][name] = {**c.info, "halves": halves_cache.get("sizes")}
        utilities = {s: utility(data.utility, per_unit[s]) for s in available}
        scores = {
            "utility_name": data.utility,
            "utility": utilities,
            "test_utility": utilities.get("test"),
            "purity": purity(data.tags, sel),
            "fit_secs": entry.fit_secs,
            "score_secs": entry.score_secs,
            "fidelity": entry.fidelity,
            "scoring": scoring,
            "selection_secs": selection_secs.get(name),
        }
        if name in ctx.extras["strategy_info"]:
            scores["selection_info"] = ctx.extras["strategy_info"][name]
        elected_name = election.elected if election is not None else None
        if ocfg.save_models == "all" or (ocfg.save_models == "elected" and name == elected_name):
            scores["model_path"] = _save_model(learner, entry, record.candidate_dir(name) / "model", name)
        keep_units = ocfg.store.per_unit_scope == "all" or stage != "screened_out"
        record.write_candidate(name, selection=sel, n_pool=data.n, budget=len(sel), role=role, stage=stage,
                               selection_secs=selection_secs.get(name, 0.0), per_unit=per_unit, scores=scores,
                               scoring_per_unit=scoring_pu or None, write_per_unit=keep_units)
        rows[name] = {"name": name, "role": role, "stage": stage, "sel_sha12": sel_sha12(sel), "n": len(sel),
                      "purity": scores["purity"], **{f"u_{s}": utilities.get(s) for s in available},
                      "scoring_u_con": scoring.get("u_con"), "scoring_u_rank": scoring.get("u_rank"),
                      "test_metrics": standard_metrics(per_unit["test"])}
        candidate_timing[name] = {"selection_secs": selection_secs.get(name), "fit_secs": entry.fit_secs,
                                  "score_secs": sum(entry.score_secs.values())}
        if "selection_info" in scores:
            candidate_timing[name]["selection_info"] = scores["selection_info"]
        record.log(f"report {name:28s} role={role:12s} test_{data.utility}={utilities.get('test'):.4f}")
    timings["final_fit"] = time.perf_counter() - t0

    # ---- leaderboard, metrics, timings, decision ----
    leaderboard = sorted(
        (row for row in rows.values() if row["name"] in candidates),
        key=lambda r: -(r["scoring_u_rank"] if r["scoring_u_rank"] is not None else -np.inf),
    )
    record.write_json("leaderboard.json", {
        "utility": data.utility,
        "candidate_dirs": record.candidate_index(),
        "rows": leaderboard,
        "screened_out": [c.name for c in election.candidates if c.stage == "screened_out"] if election else [],
        "membership": [r.to_dict() for r in member_rows],
        "skipped_strategies": skipped,
    })
    elected = election.elected if election is not None else None
    elected_row = rows.get(elected, {}) if elected else {}
    # metrics.json: the main-table rows of the track. The row id "mmds_adapt" holds the elected candidate.
    metric_rows = {}
    for name in tcfg.methods:
        if name == "mmds_adapt":
            if election is not None:
                metric_rows[name] = {**elected_row, "picked": elected}
        elif name in rows:
            metric_rows[name] = rows[name]
    if "full" in rows:
        metric_rows.setdefault("full", rows["full"])
    if election is not None:
        metric_rows.setdefault("mmds_adapt", {**elected_row, "picked": elected})
    record.write_json("metrics.json", {"utility": data.utility, "rows": metric_rows})

    stats = cache.stats()
    fde = _fde(cache, learner, data.n)
    timings["total"] = time.perf_counter() - t_start
    timing_payload = {"stages": timings, "cache": stats, "fde": fde, "candidates": candidate_timing}
    if election is not None and election.screening.get("fidelity") == "low":
        timing_payload["screening"] = _screening_savings(election, cache, learner, data.n)
        election.screening["kendall_tau_all_cells"] = _screening_tau(election, rows)
    record.write_json("timings.json", timing_payload)
    if election is not None:
        test_value = elected_row.get("u_test")
        decision = {
            **election.decision,
            "precheck": election.precheck.to_dict() if election.precheck is not None else {"enabled": False},
            "screening": election.screening,
            "construction_errors": election.construction_errors,
        }
    else:
        only = method_rows[0] if method_rows else None
        test_value = rows.get(only, {}).get("u_test") if only else None
        decision = {"controller": "off", "elected": None, "method_rows": method_rows}
    decision.update({"protocol": ocfg.protocol, "elected_test_utility": test_value, "utility": data.utility})
    record.log(f"done elected={elected} test_{data.utility}={test_value} total {timings['total']:.1f}s")
    record.finish(decision=decision, index_root=Path(out_root) / batch, status="ok", elected=elected,
                  test=test_value, extra={"track": track.name, "dataset": tcfg.dataset, "learner": tcfg.learner,
                                          "seed": tcfg.seed, "protocol": ocfg.protocol, "standalone": standalone})
    return {"status": "ok", "cell": str(cell), "elected": elected, "test": test_value,
            "seconds": timings["total"]}


def cooperative_family(track: Track, data, sig, tcfg: TrackConfig, ocfg: OmniSelectConfig,
                       grid: list, wanted: list[str]) -> tuple[CooperativeFamily, Optional[dict[str, Any]]]:
    """Neighbour graph, gate scores and the CooperativeFamily of a cell, plus the cleanliness record.

    The learned cleanliness score is computed when a grid cell or clean_top reads it. It is stored in
    ``sig.saved["cleanliness"]`` (signals/cleanliness.npy) and its features, cross-validated AUC and
    seconds are returned for signals/timings.json.
    """
    co = ocfg.cooperative
    inputs = track.cooperative_inputs(data, sig, tcfg, ocfg)
    graph = neighbour_graph(inputs["veto_features"], inputs["metric"], co.knn)
    scores = {"authenticity": np.asarray(sig.auth, dtype=float)}
    clean_info = None
    if any(cell[0] == "cleanliness" for cell in grid) or "clean_top" in wanted:
        if inputs.get("cleanliness") is None or len(data.splits.con) == 0:
            raise ValueError(f"the cleanliness score needs V_con records and track inputs ({track.name})")
        from omniselect.core.signals.cleanliness import cleanliness_features, learned_cleanliness

        t0 = time.perf_counter()
        con_graph = neighbour_graph(inputs["veto_features"], inputs["metric"], co.knn,
                                    query=inputs["con_veto_features"])
        Fp, Fc, names = cleanliness_features(inputs["cleanliness"], graph, con_graph, tcfg.seed, ocfg.cleanliness)
        score, auc = learned_cleanliness(Fp, Fc, tcfg.seed, ocfg.cleanliness)
        scores["cleanliness"] = score
        sig.saved["cleanliness"] = score
        clean_info = {"features": names, "cv_auc": auc, "n_pool": int(len(Fp)), "n_con": int(len(Fc)),
                      "seconds": time.perf_counter() - t0, "neighbour_secs": graph.seconds}
    family = CooperativeFamily(
        scores=scores, graph=graph, selector_features=inputs["selector_features"], seed=tcfg.seed,
        reference_rho=co.reference_rho, outlier_z=co.outlier_z if co.outlier_veto else None,
        duplicate_ratio=co.duplicate_ratio if co.duplicate_veto else None, budget_gate=inputs.get("budget_gate"),
        budget_cut=data.budget_cut, token_orders=inputs.get("token_orders", {}))
    return family, clean_info


def _save_model(learner, entry, directory: Path, name: str) -> Optional[str]:
    """Save the fitted model of ``entry`` when the learner supports it. Returns the path."""
    saver = getattr(learner, "save", None)
    if saver is None or entry.model is None:
        return None
    return str(saver(entry.model, directory, name))


def held_halves(pu: dict[str, np.ndarray], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two halves of the V_con units: stratified by domain (text) or class, contiguous in time for
    forecasting windows (the earlier half fits), otherwise a seeded split. Seed stable_seed(seed, 'held_half')."""
    from omniselect.tools.pairing import stable_seed

    rng = np.random.default_rng(stable_seed(seed, "held_half"))
    if "start" in pu and "domain" not in pu:
        order = np.argsort(np.asarray(pu["start"]), kind="stable")
        half = len(order) // 2
        return np.sort(order[:half]), np.sort(order[half:])
    strata = pu.get("domain", pu.get("target"))
    if strata is None or np.asarray(strata).ndim != 1:
        n = len(next(iter(pu.values())))
        perm = rng.permutation(n)
        return np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    strata = np.asarray(strata)
    first, second = [], []
    for value in sorted(set(strata.tolist()), key=str):
        members = rng.permutation(np.flatnonzero(strata == value))
        cut = (len(members) + int(rng.integers(0, 2))) // 2
        first.extend(members[:cut].tolist())
        second.extend(members[cut:].tolist())
    return np.sort(np.asarray(first, dtype=int)), np.sort(np.asarray(second, dtype=int))


def slice_units(pu: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    """Per-unit dict restricted to the units ``idx`` (arrays with one row per unit; ``classes`` kept)."""
    n = next(len(np.asarray(v)) for k, v in pu.items() if k != "classes" and np.ndim(v) >= 1)
    return {k: (np.asarray(v)[idx] if k != "classes" and np.ndim(v) >= 1 and len(np.asarray(v)) == n else v)
            for k, v in pu.items()}


def screening_subsample(selection: list[int], fraction: float, seed: int) -> list[int]:
    """Seeded subsample of round(fraction * |selection|) records, kept in the selection's order.

    The generator is default_rng(stable_seed(seed, 'screen', sel_sha12(selection))), so a
    selection always maps to the same subsample.
    """
    from omniselect.tools.pairing import stable_seed

    sel = [int(i) for i in selection]
    size = max(1, int(round(float(fraction) * len(sel))))
    rng = np.random.default_rng(stable_seed(seed, "screen", sel_sha12(sel)))
    keep = np.sort(rng.choice(len(sel), size=min(size, len(sel)), replace=False))
    return [sel[int(i)] for i in keep]


def _screening_savings(election, cache: FitCache, learner, n_pool: int) -> dict[str, Any]:
    """Low-fidelity fits and FDE against the full-fidelity fits of screened-out cells the controller avoided."""
    low_events = [e for e in cache.events if e.stage == "screen" and e.fitted]
    factor = learner.cost_factor if hasattr(learner, "cost_factor") else (lambda stage: 1.0)
    low_fde = sum(e.n_records / max(n_pool, 1) * factor("screen") for e in low_events)
    out = [c for c in election.candidates if c.stage == "screened_out"]
    avoided = sum(len(c.selection) / max(n_pool, 1) * factor("con") for c in out)
    return {"kind": "low_fidelity", "low_fidelity_fits": len(low_events), "low_fidelity_fde": float(low_fde),
            "low_fidelity_evaluations": int(election.screening.get("evaluations", 0)),
            "full_fidelity_fits_avoided": len(out), "full_fidelity_fde_avoided": float(avoided),
            "net_fde_saved": float(avoided - low_fde),
            "definition": "fits the controller would make at the construction schedule for screened-out cells "
                          "under screening.kind=full, minus the low-fidelity fits"}


def _screening_tau(election, rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Kendall tau between first-round low-fidelity values and reported-fit u_con over all grid cells."""
    from omniselect.core.adjudication.screening import kendall_tau

    rounds = election.screening.get("rounds") or []
    first = dict(rounds[0]) if rounds else {}
    names = [n for n in first if rows.get(n, {}).get("u_con") is not None]
    tau = kendall_tau([first[n] for n in names], [rows[n]["u_con"] for n in names]) if names else None
    return {"tau": tau, "n": len(names), "low": "first successive-halving round (screen stage)",
            "full": "u_con of the reported fit"}


def _fde(cache: FitCache, learner, n_pool: int) -> dict[str, Any]:
    """Full-data-training equivalents: sum over fits of (records / N) times the stage cost factor."""
    by_stage: dict[str, float] = {}
    for event in cache.events:
        if not event.fitted:
            continue
        factor = learner.cost_factor(event.stage) if hasattr(learner, "cost_factor") else 1.0
        by_stage[event.stage] = by_stage.get(event.stage, 0.0) + event.n_records / max(n_pool, 1) * factor
    layers = {
        "construction": by_stage.get("con", 0.0) + by_stage.get("member", 0.0) + by_stage.get("screen", 0.0),
        "ranking": by_stage.get("rank", 0.0),
        "confirmation": by_stage.get("conf", 0.0),
        "report": by_stage.get("report", 0.0),
    }
    return {"by_stage": by_stage, "layers": layers, "total": float(sum(by_stage.values())),
            "definition": "sum over fits of records / pool size times schedule over reported schedule"}


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the driver."""
    p = argparse.ArgumentParser(description="Run one OmniSelect benchmark cell and write its run record.")
    p.add_argument("--track", required=True, choices=sorted(TRACKS))
    p.add_argument("--dataset", required=True)
    p.add_argument("--learner", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--protocol", default="v2", choices=list(PROTOCOLS),
                   help="protocol preset (v2_2 is the protocol of the paper, v2 and v2_1 are its ablations)")
    p.add_argument("--batch", default="local")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results_and_logs")
    p.add_argument("--methods", default=None, help="comma-separated method rows (overrides the track default)")
    p.add_argument("--set", dest="omni_set", action="append", default=[], help="OmniSelectConfig key=value")
    p.add_argument("--track-set", dest="track_set", action="append", default=[], help="TrackConfig key=value")
    p.add_argument("--smoke", action="store_true", help="tiny pool and schedules for a laptop check")
    p.add_argument("--overwrite", action="store_true", help="rewrite a complete cell")
    p.add_argument("--standalone", action="store_true", help="run the method rows only, without the controller")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point: exit 0 on success or skip, 3 when a model or dataset is unavailable, 1 on error."""
    args = build_parser().parse_args(argv)
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    track = load_track(args.track)
    track_overrides = parse_overrides(args.track_set)
    if args.methods:
        track_overrides["methods"] = args.methods
    try:
        tcfg, ocfg = resolve_configs(track, args.dataset, learner=args.learner, seed=args.seed,
                                     protocol=args.protocol, smoke=args.smoke, track_overrides=track_overrides,
                                     omni_overrides=parse_overrides(args.omni_set))
        summary = run_cell(track, tcfg, ocfg, out_root=args.out, batch=args.batch,
                           cli=[sys.executable, "-m", "tracks.common.experiment", *(argv or sys.argv[1:])],
                           overwrite=args.overwrite, standalone=args.standalone)
    except SkipCell as exc:
        print(f"[skipped] {args.track}/{args.dataset}: {exc}")
        return 3
    except Exception:  # noqa: BLE001 - reported to the queue log
        traceback.print_exc()
        return 1
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
