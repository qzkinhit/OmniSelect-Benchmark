"""Aggregate the run records of a batch into CSV files.

``cells.csv`` has one row per cell (track, dataset, learner, seed, protocol, elected strategy,
its test utility, gate, adoption, precheck, total seconds, FDE). ``rows.csv`` has one row per
method row of metrics.json with the test utility and standard test metrics. ``summary.csv``
averages rows.csv over seeds per (track, dataset, protocol, method). No value is computed that is
not in the run records.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from omniselect.store.replay import find_cells


def cell_rows(cell: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(one cell summary, the method rows of metrics.json)."""
    config = json.loads((cell / "config.json").read_text())
    decision = json.loads((cell / "decision.json").read_text())
    timings = json.loads((cell / "timings.json").read_text())
    metrics = json.loads((cell / "metrics.json").read_text())
    track = config["track"]
    key = {"track": track["track"], "dataset": track["dataset"], "learner": track["learner"],
           "seed": track["seed"], "protocol": config.get("protocol"), "batch": config.get("batch")}
    gate = decision.get("gate", {})
    summary = {**key, "elected": decision.get("elected"), "elected_test": decision.get("elected_test_utility"),
               "utility": decision.get("utility"), "gate": gate.get("kind"), "adopted": decision.get("adopted"),
               "precheck_skipped": decision.get("precheck", {}).get("skipped"),
               "seconds": timings["stages"].get("total"), "fde": timings.get("fde", {}).get("total"),
               "cache_hits": timings.get("cache", {}).get("cache_hits"), "cell": str(cell)}
    rows = []
    for name, row in metrics["rows"].items():
        rows.append({**key, "method": name, "picked": row.get("picked"), "test_utility": row.get("u_test"),
                     "purity": row.get("purity"), "sel_sha12": row.get("sel_sha12"),
                     **{f"test_{k}": v for k, v in (row.get("test_metrics") or {}).items()}})
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows with the union of their keys as the header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Aggregate run records into CSV files.")
    parser.add_argument("--batch", type=Path, required=True, help="batch directory (or any directory above cells)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    cells, rows = [], []
    for cell in find_cells(args.batch):
        if not (cell / "decision.json").is_file():
            continue
        summary, method_rows = cell_rows(cell)
        cells.append(summary)
        rows.extend(method_rows)
    if not cells:
        print(f"no complete cells under {args.batch}")
        return 1
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        if row["test_utility"] is not None:
            groups[(row["track"], row["dataset"], row["protocol"], row["method"])].append(row["test_utility"])
    summary_rows = [{"track": k[0], "dataset": k[1], "protocol": k[2], "method": k[3], "n_seeds": len(v),
                     "mean_test_utility": float(np.mean(v)), "std_test_utility": float(np.std(v))}
                    for k, v in sorted(groups.items())]
    write_csv(args.out / "cells.csv", cells)
    write_csv(args.out / "rows.csv", rows)
    write_csv(args.out / "summary.csv", summary_rows)
    print(f"{len(cells)} cells, {len(rows)} method rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
