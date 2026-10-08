"""Replay: rebuild each stored selection from selection.npz and verify its sel_sha12.

For every candidates/<name>/selection.npz under the given cells the script checks that ``idx``
is sorted, unique and inside [0, n), recomputes sel_sha12 and compares it with the stored value
and with scores.json, and checks that decision.json's elected sha matches the elected candidate.
With ``--ids`` it prints the pool ids of one candidate's subset read from splits.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from omniselect.utils.hashing import sel_sha12


def find_cells(path: Path) -> list[Path]:
    """Cell directories (those holding candidates/) at or below ``path``."""
    path = Path(path)
    if (path / "candidates").is_dir():
        return [path]
    return sorted(p.parent for p in path.rglob("candidates") if p.is_dir())


def load_selection(npz_path: Path) -> dict[str, Any]:
    """Arrays of one selection.npz as Python values."""
    with np.load(npz_path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def verify_cell(cell: Path) -> list[str]:
    """Problems found in one cell (empty list when every selection verifies)."""
    problems: list[str] = []
    sha_by_name: dict[str, str] = {}
    for npz in sorted((cell / "candidates").glob("*/selection.npz")):
        data = load_selection(npz)
        idx = data["idx"].astype(np.int64)
        n = int(data["n"])
        name = str(data["name"])
        if not np.all(np.diff(idx) > 0):
            problems.append(f"{npz}: idx is not sorted and unique")
        if len(idx) and (idx[0] < 0 or idx[-1] >= n):
            problems.append(f"{npz}: idx outside [0, {n})")
        sha = sel_sha12(idx)
        if sha != str(data["sel_sha12"]):
            problems.append(f"{npz}: sel_sha12 {sha} != stored {data['sel_sha12']}")
        scores_path = npz.parent / "scores.json"
        if scores_path.is_file() and json.loads(scores_path.read_text()).get("sel_sha12") != sha:
            problems.append(f"{scores_path}: sel_sha12 differs from selection.npz")
        sha_by_name[name] = sha
    decision_path = cell / "decision.json"
    if decision_path.is_file():
        decision = json.loads(decision_path.read_text())
        elected = decision.get("elected")
        stored = decision.get("elected_sel_sha12")
        if elected in sha_by_name and stored and sha_by_name[elected] != stored:
            problems.append(f"{decision_path}: elected sha {stored} != candidate sha {sha_by_name[elected]}")
    return problems


def subset_ids(cell: Path, candidate_dir: str) -> list[Any]:
    """Pool ids (from splits.json) of the records selected by one candidate."""
    data = load_selection(cell / "candidates" / candidate_dir / "selection.npz")
    pool = json.loads((cell / "splits.json").read_text())["ids"]["pool"]
    return [pool[int(i)] for i in data["idx"]]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Exit status 1 when any selection fails verification."""
    parser = argparse.ArgumentParser(description="Verify stored selections of run records.")
    parser.add_argument("path", type=Path, help="a cell directory or any directory above cells")
    parser.add_argument("--ids", metavar="CANDIDATE_DIR", help="print the pool ids of one candidate")
    args = parser.parse_args(argv)
    cells = find_cells(args.path)
    if not cells:
        print(f"no run records under {args.path}", file=sys.stderr)
        return 1
    if args.ids:
        for cell in cells:
            print(json.dumps({"cell": str(cell), "ids": subset_ids(cell, args.ids)}))
        return 0
    failures = 0
    total = 0
    for cell in cells:
        problems = verify_cell(cell)
        total += len(list((cell / "candidates").glob("*/selection.npz")))
        for problem in problems:
            print(problem)
        failures += len(problems)
    print(f"replay: {len(cells)} cells, {total} selections, {failures} problems")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
