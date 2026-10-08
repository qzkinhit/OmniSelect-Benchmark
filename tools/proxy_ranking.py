"""Proxy ranking: does a cheap learner rank the candidates of a cell as an expensive learner does?

The tool pairs cells of a cheap batch (for example main with the CLIP probe, DLinear and
SmolLM2-135M) with cells of an expensive batch (for example the robustness/learner swaps with
ResNet-18, chronos-bolt and SmolLM2-360M) on (track, dataset, seed). Over the candidates present
in both with the same sel_sha12 it computes Kendall tau of the ranking utilities and of the test
utilities, whether both elect the same candidate, and the expensive learner's test utility of the
cheap learner's election minus the expensive learner's best test utility (regret). One CSV and one
markdown table.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from omniselect.core.adjudication.screening import kendall_tau
from omniselect.store.replay import find_cells


def _cells(batch: Path) -> dict[tuple[str, str, int], list[Path]]:
    out: dict[tuple[str, str, int], list[Path]] = {}
    for cell in find_cells(batch):
        if not (cell / "decision.json").is_file() or not (cell / "leaderboard.json").is_file():
            continue
        track = json.loads((cell / "config.json").read_text())["track"]
        out.setdefault((track["track"], track["dataset"], int(track["seed"])), []).append(cell)
    return out


def _rows(cell: Path) -> dict[str, dict[str, Any]]:
    board = json.loads((cell / "leaderboard.json").read_text())
    return {r["name"]: r for r in board.get("rows", [])}


def compare(cheap: Path, expensive: Path) -> dict[str, Any]:
    """Agreement statistics of one cheap and one expensive cell of the same task and seed."""
    a, b = _rows(cheap), _rows(expensive)
    common = [n for n in a if n in b and a[n].get("sel_sha12") == b[n].get("sel_sha12")]

    def vec(rows, key):
        return [rows[n].get(key) for n in common]

    rank_a, rank_b = vec(a, "u_rank"), vec(b, "u_rank")
    test_a, test_b = vec(a, "u_test"), vec(b, "u_test")
    ok_rank = [i for i, (x, y) in enumerate(zip(rank_a, rank_b)) if x is not None and y is not None]
    ok_test = [i for i, (x, y) in enumerate(zip(test_a, test_b)) if x is not None and y is not None]
    tau_rank = kendall_tau([rank_a[i] for i in ok_rank], [rank_b[i] for i in ok_rank]) if len(ok_rank) > 1 else None
    tau_test = kendall_tau([test_a[i] for i in ok_test], [test_b[i] for i in ok_test]) if len(ok_test) > 1 else None
    elected_a = json.loads((cheap / "decision.json").read_text()).get("elected")
    elected_b = json.loads((expensive / "decision.json").read_text()).get("elected")
    regret: Optional[float] = None
    if elected_a in b and b[elected_a].get("u_test") is not None and ok_test:
        best = max(test_b[i] for i in ok_test)
        regret = float(b[elected_a]["u_test"]) - float(best)
    return {"cheap_cell": str(cheap), "expensive_cell": str(expensive), "common_candidates": len(common),
            "kendall_tau_rank": tau_rank, "kendall_tau_test": tau_test, "cheap_elected": elected_a,
            "expensive_elected": elected_b, "same_election": elected_a == elected_b,
            "regret_of_cheap_election": regret}


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point: --cheap <batch> --expensive <batch> [...] --out <dir>."""
    parser = argparse.ArgumentParser(description="Rank agreement of cheap and expensive learners on the same cells.")
    parser.add_argument("--cheap", type=Path, required=True)
    parser.add_argument("--expensive", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    cheap = _cells(args.cheap)
    rows = []
    for batch in args.expensive:
        for key, cells in sorted(_cells(batch).items()):
            for cell in cells:
                for base in cheap.get(key, []):
                    learner_a = json.loads((base / "config.json").read_text())["track"]["learner"]
                    learner_b = json.loads((cell / "config.json").read_text())["track"]["learner"]
                    if learner_a == learner_b:
                        continue
                    rows.append({"track": key[0], "dataset": key[1], "seed": key[2], "cheap_learner": learner_a,
                                 "expensive_learner": learner_b, **compare(base, cell)})
    args.out.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(args.out / "proxy_ranking.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["| Task | Cheap | Expensive | Seeds | Kendall tau (rank) | Kendall tau (test) | Same election | "
             "Mean regret |", "|---|---|---|---:|---:|---:|---:|---:|"]
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((f"{r['track']}/{r['dataset']}", r["cheap_learner"], r["expensive_learner"]), []).append(r)

    def mean(values):
        values = [v for v in values if v is not None]
        return "" if not values else f"{float(np.mean(values)):.3f}"

    for (task, a, b), group in sorted(groups.items()):
        lines.append(f"| {task} | {a} | {b} | {len(group)} | {mean(r['kendall_tau_rank'] for r in group)} | "
                     f"{mean(r['kendall_tau_test'] for r in group)} | "
                     f"{sum(bool(r['same_election']) for r in group)} of {len(group)} | "
                     f"{mean(r['regret_of_cheap_election'] for r in group)} |")
    (args.out / "proxy_ranking.md").write_text("# Proxy ranking\n\n" + "\n".join(lines) + "\n")
    print(f"proxy_ranking: {len(rows)} pairs, {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
