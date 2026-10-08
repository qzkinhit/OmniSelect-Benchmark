"""Recompute a metric for every candidate of every cell from its per-unit files.

``python -m omniselect.store.recompute <path> --metric macro_f1 --split test`` reads
candidates/*/per_unit/<split>.npz, computes the metric with omniselect.store.metrics, and
prints one JSON line per candidate. ``--check`` compares the value with the one stored in
scores.json at write time and exits 1 on any difference larger than ``--tol``. A candidate without
per-unit files (``store.per_unit_scope=finalists``) is reported as scalar only and skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from omniselect.store.metrics import METRICS, compute
from omniselect.store.replay import find_cells


def load_per_unit(path: Path) -> dict[str, np.ndarray]:
    """Arrays of one per-unit npz file."""
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def recompute_cell(cell: Path, metric: str, split: str) -> list[dict[str, Any]]:
    """One row per candidate: name, recomputed value, stored value (or None)."""
    rows = []
    for cdir in sorted((cell / "candidates").iterdir()):
        path = cdir / "per_unit" / f"{split}.npz"
        if not path.is_file():
            if not (cdir / "per_unit").is_dir():
                rows.append({"cell": str(cell), "candidate": cdir.name, "metric": metric, "split": split,
                             "value": None, "stored": None, "scalar_only": True})
            continue
        try:
            value = compute(metric, load_per_unit(path))
        except (KeyError, ValueError, IndexError) as exc:
            rows.append({"cell": str(cell), "candidate": cdir.name, "metric": metric, "split": split,
                         "value": None, "stored": None, "error": f"{type(exc).__name__}: {exc}"})
            continue
        stored = None
        scores_path = cdir / "scores.json"
        if scores_path.is_file():
            stored = json.loads(scores_path.read_text()).get("metrics", {}).get(split, {}).get(metric)
        rows.append({"cell": str(cell), "candidate": cdir.name, "metric": metric, "split": split,
                     "value": value, "stored": stored})
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Recompute metrics from per-unit run-record files.")
    parser.add_argument("path", type=Path, help="a cell directory or any directory above cells")
    parser.add_argument("--metric", required=True, choices=sorted(METRICS))
    parser.add_argument("--split", default="test", choices=["con", "rank", "conf", "test"])
    parser.add_argument("--check", action="store_true", help="compare with the values stored in scores.json")
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--csv", type=Path, help="also write the rows to this CSV file")
    args = parser.parse_args(argv)
    cells = find_cells(args.path)
    if not cells:
        print(f"no run records under {args.path}", file=sys.stderr)
        return 1
    rows = [row for cell in cells for row in recompute_cell(cell, args.metric, args.split)]
    mismatches = 0
    checked = 0
    for row in rows:
        if args.check and isinstance(row["value"], float) and row["stored"] is not None:
            checked += 1
            if abs(row["value"] - float(row["stored"])) > args.tol:
                mismatches += 1
                row["mismatch"] = True
        print(json.dumps(row, default=float))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)
    if args.check:
        scalar = sum(1 for r in rows if r.get("scalar_only"))
        print(f"recompute: {len(rows)} rows, {checked} checked against stored values, {mismatches} mismatches, "
              f"{scalar} scalar-only candidates skipped")
        if checked == 0:
            print("recompute: no stored values for this metric and split", file=sys.stderr)
            return 1
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
