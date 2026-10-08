"""Replay every gate from stored per-unit files, without training.

For each complete cell margin, argmax and retain use the best reference and the best challenger on
the ranking split. Lcb, bootstrap and eprocess order the reference and the top-K challengers on
gate.pair_split and read per-unit paired differences of the reported fits on the confirmation split
(the ranking split when the cell has none) in the configured reading order. Every adoption gets the
e-value of its pair with K = k (certified when it reaches K / delta, with the units read). The
margin rule is also replayed at every margin_frac of MARGIN_SWEEP (gates "margin@<value>"). Rows
join the elected test utility and the gain over random, one CSV per batch plus summaries.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from omniselect.core.gates import GateResult, argmax_gate, margin_gate, run_family
from omniselect.core.gates.audit import audit_family, test_paired
from omniselect.core.gates.base import FamilyDecision
from omniselect.core.gates.paired import paired_sample
from omniselect.core.gates.surrogate import unit_scores
from omniselect.store.replay import find_cells

KINDS = ("margin", "argmax", "retain", "lcb", "bootstrap", "eprocess")
MARGIN_SWEEP = (0.0, 0.005, 0.015, 0.03, 0.05)   # margin_frac values replayed as gates "margin@<value>"


def sweep_name(frac: float) -> str:
    """Gate name of the margin rule at ``frac`` in the sweep."""
    return f"margin@{float(frac):g}"
STATISTICAL = ("lcb", "bootstrap", "eprocess")


def _per_unit(cell: Path, candidate_dir: str, split: str) -> dict[str, np.ndarray]:
    with np.load(cell / "candidates" / candidate_dir / "per_unit" / f"{split}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _gate_settings(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Gate constants of the cell's configuration with CLI overrides (records of 96af9ab lack some keys)."""
    gate = (config.get("omniselect") or {}).get("gate") or {}
    settings = {
        "k": int(gate.get("k_challengers", 1)),
        "delta": float(gate.get("delta", 0.05)),
        "eps": float(gate.get("eps", 0.0)),
        "margin_frac": float(gate.get("margin_frac", 0.015)),
        "n_boot": int(gate.get("n_boot", 1000)),
        "p_beat_min": float(gate.get("p_beat_min", 0.9)),
        "clip": float(gate.get("text_clip", 1.0)),
        "ts_block_steps": int(gate.get("ts_block_steps", 0)),
        "reading_order": str(gate.get("reading_order", "stratified")),
        "pair_split": str(gate.get("pair_split", "rank")),
        "margin_sweep": tuple(MARGIN_SWEEP),
    }
    settings.update({key: value for key, value in overrides.items() if value is not None})
    return settings


def _test_utility(cell: Path, board: dict[str, Any], name: str) -> Optional[float]:
    for row in board.get("rows", []):
        if row.get("name") == name and row.get("u_test") is not None:
            return float(row["u_test"])
    directory = board.get("candidate_dirs", {}).get(name)
    if directory and (cell / "candidates" / directory / "scores.json").is_file():
        value = json.loads((cell / "candidates" / directory / "scores.json").read_text()).get("test_utility")
        return None if value is None else float(value)
    metrics_path = cell / "metrics.json"
    if metrics_path.is_file():
        row = json.loads(metrics_path.read_text()).get("rows", {}).get(name) or {}
        if row.get("u_test") is not None:
            return float(row["u_test"])
    return None


def replay_cell(cell: Path, **overrides: Any) -> dict[str, Any]:
    """Decisions of every gate kind for one cell, the audit, and the elected test utility of each.

    ``overrides`` replaces gate constants of the cell configuration (k, delta, eps, margin_frac,
    n_boot, p_beat_min, clip, reading_order). Returns the references, the challengers, the unit
    split, the recorded decision, the audit of the ranking reference against the top-K ranking
    challengers, and one FamilyDecision dict per gate kind with ``certified`` (the adopted pair's
    e-value with K = k reaches K / delta) and ``units_read`` (units read when it did).
    """
    board = json.loads((cell / "leaderboard.json").read_text())
    splits = json.loads((cell / "splits.json").read_text())
    config = json.loads((cell / "config.json").read_text())
    decision = json.loads((cell / "decision.json").read_text())
    settings = _gate_settings(config, overrides)
    seed = int(config["track"]["seed"])
    utility_name = board["utility"]
    rows = [r for r in board["rows"] if r.get("scoring_u_rank") is not None]
    pair_key = "scoring_u_con" if settings["pair_split"] == "con" else "scoring_u_rank"

    def ordered(key: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        usable = [r for r in rows if r.get(key) is not None]
        refs = [r for r in usable if r["role"] == "reference"]
        best = refs[0]
        for row in refs[1:]:                  # first maximum, as the controller breaks ties
            if row[key] > best[key]:
                best = row
        return best, sorted((r for r in usable if r["role"] == "challenger"), key=lambda r: -r[key])

    rank_reference, rank_challengers = ordered("scoring_u_rank")
    reference, chal = ordered(pair_key)
    chal = chal[: settings["k"]]
    split = "conf" if splits["counts"].get("conf", 0) > 0 else "rank"
    dirs = board["candidate_dirs"]
    track = config["track"]
    block = settings["ts_block_steps"] or int(track.get("L", 0)) + int(track.get("H", 0))
    unit_cache: dict[str, Any] = {}
    certificates: dict[tuple[str, str], dict[str, Any]] = {}

    def units(name: str):
        if name not in unit_cache:
            unit_cache[name] = unit_scores(utility_name, _per_unit(cell, dirs[name], split),
                                           s0=float(splits.get("s0") or 1.0),
                                           block_steps=block if utility_name == "neg_mase" else 0)
        return unit_cache[name]

    def sample(ref: str, name: str):
        ref_units = units(ref)
        clip = settings["clip"] if ref_units.kind == "text_neg_nll" else None
        return paired_sample(ref_units, units(name), clip=clip)

    def certificate(ref: str, name: str) -> dict[str, Any]:
        if (ref, name) not in certificates:
            certificates[(ref, name)] = audit_family(
                ref, [(name, sample(ref, name))], k=settings["k"], seed=seed, delta=settings["delta"],
                eps=settings["eps"], n_boot=settings["n_boot"], order=settings["reading_order"],
                split=split)["entries"][0]
        return certificates[(ref, name)]

    ref_name = reference["name"]
    rank_ref = rank_reference["name"]
    audit_names = [r["name"] for r in rank_challengers[: settings["k"]]]
    for name in audit_names:
        certificate(rank_ref, name)
    out: dict[str, Any] = {"reference": ref_name, "rank_reference": rank_ref, "pair_split": settings["pair_split"],
                           "challengers": [r["name"] for r in chal], "unit_split": split, "settings": settings,
                           "recorded": {"kind": (decision.get("gate") or {}).get("kind"),
                                        "reference": decision.get("reference"), "elected": decision.get("elected")},
                           "audit": {"reference": rank_ref, "K": settings["k"], "split": split,
                                     "reading_order": settings["reading_order"],
                                     "entries": [certificate(rank_ref, n) for n in audit_names]},
                           "gates": {}}
    for kind in KINDS:
        if kind == "retain":
            out["gates"][kind] = FamilyDecision("retain", rank_ref, rank_ref, False, 0, []).to_dict()
            continue

        def test_one(name: str, k_family: int, kind: str = kind) -> GateResult:
            if kind in ("margin", "argmax"):
                row = next(r for r in rank_challengers if r["name"] == name)
                gate_fn = margin_gate if kind == "margin" else argmax_gate
                extra = (settings["margin_frac"],) if kind == "margin" else ()
                return gate_fn(rank_ref, rank_reference["scoring_u_rank"], name, row["scoring_u_rank"], *extra)
            return test_paired(kind, ref_name, name, sample(ref_name, name), k=k_family, seed=seed,
                               delta=settings["delta"], eps=settings["eps"], n_boot=settings["n_boot"],
                               p_beat_min=settings["p_beat_min"], order=settings["reading_order"])

        if kind in ("margin", "argmax"):
            family, family_ref = [r["name"] for r in rank_challengers[: settings["k"]]], rank_ref
        else:
            family, family_ref = [r["name"] for r in chal], ref_name
        out["gates"][kind] = run_family(kind, family_ref, family, test_one,
                                        test_all=kind != "lcb").to_dict()
    sweep_family = [r["name"] for r in rank_challengers[: settings["k"]]]
    for frac in settings["margin_sweep"]:
        rank_u = {r["name"]: r["scoring_u_rank"] for r in rank_challengers}

        def margin_at(name: str, k_family: int, frac: float = frac) -> GateResult:
            return margin_gate(rank_ref, rank_reference["scoring_u_rank"], name, rank_u[name], frac)

        out["gates"][sweep_name(frac)] = run_family("margin", rank_ref, sweep_family, margin_at,
                                                     test_all=True).to_dict()
    random_test = _test_utility(cell, board, "random")
    recorded = out["recorded"]
    for gate in list(out["gates"].values()) + [recorded]:
        gate["u_test"] = _test_utility(cell, board, gate["elected"]) if gate.get("elected") else None
        gate["u_test_random"] = random_test
        gate["gain_over_random"] = (None if gate["u_test"] is None or random_test is None
                                    else gate["u_test"] - random_test)
        adopted = gate.get("adopted", gate.get("elected") not in (None, gate.get("reference")))
        ref = gate.get("reference")
        if adopted and ref and gate["elected"] in dirs and ref in dirs:
            cert = certificate(ref, gate["elected"])
            gate["certified"], gate["units_read"] = cert["certified"], cert["stop_index"]
        else:
            gate["certified"], gate["units_read"] = (None, None)
        gate.setdefault("adopted", bool(adopted))
    return out


def cell_rows(cell: Path, result: dict[str, Any], batch: str) -> list[dict[str, Any]]:
    """CSV rows of one replayed cell: one per gate kind plus the recorded decision."""
    config = json.loads((cell / "config.json").read_text())
    track = config["track"]
    key = {"batch": batch, "cell": str(cell), "track": track["track"], "dataset": track["dataset"],
           "learner": track["learner"], "seed": track["seed"], "protocol": config.get("protocol")}
    rows = []
    recorded = result["recorded"]
    for kind, gate in list(result["gates"].items()) + [("recorded", recorded)]:
        stats = [{"challenger": t["challenger"], "adopted": t["adopted"], **t["statistics"]}
                 for t in gate.get("tests", [])]
        rows.append({**key, "gate": kind, "reference": gate.get("reference"), "elected": gate["elected"],
                     "adopted": gate.get("adopted"), "certified": gate.get("certified"),
                     "units_read": gate.get("units_read"), "k_tested": gate.get("k_tested"),
                     "unit_split": result["unit_split"], "u_test": gate["u_test"],
                     "u_test_random": gate["u_test_random"], "gain_over_random": gate["gain_over_random"],
                     "matches_recorded": (gate["elected"] == recorded["elected"]
                                          if kind == recorded["kind"] else None),
                     "statistics": json.dumps(stats if kind != "recorded" else {"kind": recorded["kind"]},
                                              default=float, allow_nan=False)})
    return rows


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Per-gate and per-task summaries.

    Per gate: cells, adoption rate, mean gain over random, mean relative gain, cells below random,
    adoptions and certified adoptions. Per task and gate: mean elected test utility, adoptions,
    certified adoptions and the units read when the threshold was reached (median, or none). The
    margin sweep appears as gates "margin@<margin_frac>".
    """
    present = list(dict.fromkeys(r["gate"] for r in rows))
    sweep = sorted((g for g in present if g.startswith("margin@")), key=lambda g: float(g.split("@")[1]))
    gates = list(KINDS) + sweep + ["recorded"]
    by_gate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_gate[row["gate"]].append(row)
    table = []
    for gate in gates:
        group = by_gate.get(gate, [])
        gains = [r["gain_over_random"] for r in group if r["gain_over_random"] is not None]
        rel = [r["gain_over_random"] / abs(r["u_test_random"]) for r in group
               if r["gain_over_random"] is not None and r["u_test_random"] not in (None, 0)]
        adopted = [bool(r["adopted"]) for r in group if r["adopted"] is not None]
        table.append({"task": "all", "gate": gate, "cells": len(group),
                      "adoption_rate": float(np.mean(adopted)) if adopted else None,
                      "adoptions": int(sum(adopted)), "certified_adoptions": int(sum(bool(r["certified"])
                                                                                      for r in group)),
                      "mean_gain_over_random": float(np.mean(gains)) if gains else None,
                      "mean_relative_gain": float(np.mean(rel)) if rel else None,
                      "cells_worse_than_random": int(sum(g < 0 for g in gains))})
    lines = ["| Gate | Cells | Adoptions | Certified adoptions | Mean gain over random | Mean relative gain | "
             "Cells below random |", "|---|---:|---:|---:|---:|---:|---:|"]
    for t in table:
        lines.append(f"| {t['gate']} | {t['cells']} | {t['adoptions']} | {t['certified_adoptions']} | "
                     f"{_fmt(t['mean_gain_over_random'])} | {_fmt(t['mean_relative_gain'])} | "
                     f"{t['cells_worse_than_random']} |")
    lines += ["", "Per task and gate: mean elected test utility (task units, higher is better), adoptions, "
              "certified adoptions (e-value of the adopted pair reaches K / delta), and the median number of "
              "units read when the threshold was reached.", "",
              "| Task | Gate | Cells | Mean elected test utility | Adoptions | Certified | Units read |",
              "|---|---|---:|---:|---:|---:|---|"]
    for task in sorted({f"{r['track']}/{r['dataset']}" for r in rows}):
        for gate in gates:
            group = [r for r in by_gate.get(gate, []) if f"{r['track']}/{r['dataset']}" == task]
            if not group:
                continue
            tests = [r["u_test"] for r in group if r["u_test"] is not None]
            reads = [r["units_read"] for r in group if r["certified"] and r["units_read"] is not None]
            entry = {"task": task, "gate": gate, "cells": len(group),
                     "mean_elected_test_utility": float(np.mean(tests)) if tests else None,
                     "adoptions": int(sum(bool(r["adopted"]) for r in group)),
                     "certified_adoptions": int(sum(bool(r["certified"]) for r in group)),
                     "units_read_median": float(np.median(reads)) if reads else None}
            table.append(entry)
            read_text = _fmt(entry["units_read_median"], 0) if reads else "none"
            lines.append(f"| {task} | {gate} | {entry['cells']} | {_fmt(entry['mean_elected_test_utility'])} | "
                         f"{entry['adoptions']} | {entry['certified_adoptions']} | {read_text} |")
    return table, "\n".join(lines) + "\n"


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Rows with the union of their keys as header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def replay_batch(batch: Path, out_dir: Path, *, write_cell_json: bool = False,
                 **overrides: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay every complete cell of ``batch``. Writes gates_<batch>.csv, gates_summary_<batch>.{csv,md}."""
    name = batch.name
    rows: list[dict[str, Any]] = []
    for cell in find_cells(batch):
        if not (cell / "decision.json").is_file() or not (cell / "leaderboard.json").is_file():
            continue
        if json.loads((cell / "decision.json").read_text()).get("controller") == "off":
            continue                       # standalone method runs have no election
        result = replay_cell(cell, **overrides)
        if write_cell_json:
            (cell / "gate_replay.json").write_text(json.dumps(result, indent=2, default=float))
        rows.extend(cell_rows(cell, result, name))
        print(json.dumps({"cell": str(cell), **{k: v["elected"] for k, v in result["gates"].items()}}))
    table, markdown = summarize(rows)
    if rows:
        write_csv(out_dir / f"gates_{name}.csv", rows)
        write_csv(out_dir / f"gates_summary_{name}.csv", table)
        (out_dir / f"gates_summary_{name}.md").write_text(f"# Gate replay of `{batch}`\n\n{markdown}")
    return rows, table


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Replay gates from stored per-unit files.")
    parser.add_argument("paths", type=Path, nargs="+", help="batch directories (one CSV each)")
    parser.add_argument("--out", type=Path, help="output directory (default <batch>/gate_replay)")
    parser.add_argument("--csv", type=Path, help="also write all rows of all batches to this file")
    parser.add_argument("--k", type=int, help="family size K (default: gate.k_challengers of each cell)")
    parser.add_argument("--delta", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--n-boot", type=int, dest="n_boot")
    parser.add_argument("--p-beat-min", type=float, dest="p_beat_min")
    parser.add_argument("--reading-order", choices=("stratified", "uniform"), dest="reading_order")
    parser.add_argument("--margin-sweep", type=float, nargs="*", dest="margin_sweep",
                        help=f"margin_frac values replayed as extra gates (default {' '.join(map(str, MARGIN_SWEEP))})")
    parser.add_argument("--write-cell-json", action="store_true", help="also write <cell>/gate_replay.json")
    args = parser.parse_args(argv)
    overrides = {"k": args.k, "delta": args.delta, "eps": args.eps, "n_boot": args.n_boot,
                 "p_beat_min": args.p_beat_min, "reading_order": args.reading_order,
                 "margin_sweep": tuple(args.margin_sweep) if args.margin_sweep is not None else None}
    everything: list[dict[str, Any]] = []
    for path in args.paths:
        out_dir = args.out or (path / "gate_replay")
        rows, table = replay_batch(path, out_dir, write_cell_json=args.write_cell_json, **overrides)
        everything.extend(rows)
        for t in table:
            print(json.dumps({"batch": path.name, **t}))
    if args.csv and everything:
        write_csv(args.csv, everything)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
