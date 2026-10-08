"""Audit finished run records for missing values, out-of-range utilities, broken selections and outliers.

Usage: python -m tools.audit_records --root results_and_logs [--batches main main_nc ...] [--out DIR]

For every cell directory (a directory holding config.json) under the given batches it checks:
  cell level   decision.json present and parseable, elected strategy on the leaderboard, elected and
               random test utilities finite and inside the range of the task's utility, not every
               candidate with the same test utility, cells without decision.json that are not running
               (crashed or killed), signal arrays finite;
  candidate    selection.npz present, indices unique, inside the pool, count equal to the stored n,
               stored pool size equal to splits.json, sel_sha12 equal to sha256 of the sorted indices,
               test utility finite and in range,
               scores.json test utility equal to the leaderboard value, per-unit files present when
               scores.json says they were written, their arrays finite and of the split sizes;
  batch level  elected test utility of a seed more than 4 robust z-scores away from the other seeds of
               the same task (median and MAD), elected below random by more than 2% of |random|.
Severity: ERROR means the cell cannot be used as it is, WARN means it needs a look, INFO is recorded.
Writes AUDIT_<timestamp>.md, AUDIT_latest.md and audit_latest.json under --out.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict

import numpy as np

RANGES = {  # mathematical bounds of the reported utility
    "accuracy": (0.0, 1.0),
    "macro_f1": (0.0, 1.0),
    "balanced_accuracy": (0.0, 1.0),
    "auc": (0.0, 1.0),
    "neg_mase": (-math.inf, 0.0),       # MASE is nonnegative, with no finite upper bound
    "neg_gmean_ppl": (-math.inf, -1.0), # perplexity is at least one, with no finite upper bound
}
HIGH_ERROR_THRESHOLDS = {"neg_mase": -5.0, "neg_gmean_ppl": -200.0}


def utility_severity(utility: str, value) -> str | None:
    """Impossible or non-finite values are errors. Finite high-error tails remain warnings."""
    if not finite(value):
        return "ERROR"
    lo, hi = RANGES.get(utility, (-math.inf, math.inf))
    if not lo <= value <= hi:
        return "ERROR"
    if utility in HIGH_ERROR_THRESHOLDS and value < HIGH_ERROR_THRESHOLDS[utility]:
        return "WARN"
    return None


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def sel_sha12(idx) -> str:
    return hashlib.sha256(str(sorted(int(i) for i in idx)).encode()).hexdigest()[:12]


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def rows_of(lb):
    return lb if isinstance(lb, list) else lb.get("rows", [])


def running_cells(root: str) -> set[str]:
    """Cell paths (as written in the queue files) of the jobs that have a running_<name> marker."""
    qs = os.path.join(root, "queue_state")
    names = {f[len("running_"):] for f in os.listdir(qs) if f.startswith("running_")} if os.path.isdir(qs) else set()
    cells = set()
    qdir = os.path.join(os.path.dirname(os.path.abspath(root)), "run_omniselect", "queues")
    for q in glob.glob(os.path.join(qdir, "*.txt")):
        with open(q) as fh:
            for line in fh:
                if "# cell=" in line and " name=" in line:
                    cell = line.split("# cell=", 1)[1].split()[0]
                    name = line.split(" name=", 1)[1].split()[0]
                    if name in names:
                        cells.add(os.path.normpath(cell))
    return cells


def audit_cell(cell: str, issues: list, running: set) -> dict | None:
    rel = cell
    add = lambda sev, what: issues.append((sev, rel, what))  # noqa: E731
    dec_p, lb_p = os.path.join(cell, "decision.json"), os.path.join(cell, "leaderboard.json")
    if not os.path.exists(dec_p):
        log = os.path.join(cell, "log.txt")
        age = time.time() - os.path.getmtime(log) if os.path.exists(log) else None
        busy = os.path.normpath(cell) in running or (age is not None and age < 3600)
        if not busy:
            add("ERROR", "no decision.json and no recent log activity: crashed or killed, rerun this cell")
        return None
    try:
        dec, lb = load_json(dec_p), load_json(lb_p)
        splits = load_json(os.path.join(cell, "splits.json"))
    except Exception as e:  # noqa: BLE001
        add("ERROR", f"unreadable record: {e}")
        return None
    utility = splits.get("utility") or (load_json(os.path.join(cell, "metrics.json")).get("utility"))
    lo, hi = RANGES.get(str(utility), (-math.inf, math.inf))
    rows = {r["name"]: r for r in rows_of(lb)}
    if dec.get("controller") == "off":   # standalone method rows (R5): no election to check
        rows = {}
        for sp_ in glob.glob(os.path.join(cell, "candidates", "*", "scores.json")):
            sc_ = load_json(sp_)
            rows[sc_.get("name", os.path.basename(os.path.dirname(sp_)))] = {
                "u_test": sc_.get("test_utility"), "role": sc_.get("role")}
        if not rows:
            add("ERROR", "standalone cell without candidate scores")
        for n_, r_ in rows.items():
            if not finite(r_["u_test"]):
                add("ERROR", f"standalone row {n_} test utility is {r_['u_test']!r}")
            elif not (lo <= r_["u_test"] <= hi):
                add("ERROR", f"standalone row {n_} test utility {r_['u_test']} outside [{lo}, {hi}]")
        dec = dict(dec, elected=None)
    counts = {k: len(v) for k, v in (splits.get("ids") or {}).items()}
    n_pool = counts.get("pool")
    elected = dec.get("elected")
    if dec.get("controller") == "off":
        elected = None
    elif elected not in rows:
        add("ERROR", f"elected {elected!r} missing from leaderboard")
        return None
    u_el = rows[elected].get("u_test") if elected else None
    if elected is None:
        pass
    elif not finite(u_el):
        add("ERROR", f"elected {elected} test utility is {u_el!r}")
    elif not (lo <= u_el <= hi):
        add("ERROR", f"elected {elected} test utility {u_el} outside [{lo}, {hi}] for {utility}")
    u_rand = rows.get("random", {}).get("u_test")
    if "random" in rows and not finite(u_rand):
        add("ERROR", f"random test utility is {u_rand!r}")
    tests = [r.get("u_test") for r in rows.values() if r.get("role") != "diagnostic"]
    bad = [n for n, r in rows.items() if not finite(r.get("u_test")) and r.get("role") != "diagnostic"]
    if bad:
        add("ERROR", f"{len(bad)} candidates without a finite test utility: {', '.join(bad[:6])}")
    good = [t for t in tests if finite(t)]
    if len(good) >= 3 and max(good) - min(good) < 1e-12:
        add("WARN", f"all {len(good)} candidates have the same test utility {good[0]}: check learner and data")
    out_of_range = [n for n, r in rows.items() if finite(r.get("u_test")) and not (lo <= r["u_test"] <= hi)]
    if out_of_range:
        add("ERROR", f"{len(out_of_range)} candidates outside [{lo}, {hi}]: {', '.join(out_of_range[:6])}")
    high_error = [n for n, r in rows.items() if utility_severity(str(utility), r.get("u_test")) == "WARN"]
    if high_error:
        threshold = HIGH_ERROR_THRESHOLDS[str(utility)]
        add("WARN", f"{len(high_error)} candidates have high error ({utility} below heuristic {threshold}, "
                    f"not a mathematical bound): {', '.join(high_error[:6])}")
    if utility == "auc" and finite(u_el) and u_el < 0.5:
        add("WARN", f"elected AUC {u_el} below 0.5")
    # signals
    for f in glob.glob(os.path.join(cell, "signals", "*.npy")):
        try:
            a = np.load(f, allow_pickle=False)
            if a.dtype.kind in "fc" and not np.all(np.isfinite(a)):
                add("ERROR", f"signal {os.path.basename(f)} has {int((~np.isfinite(a)).sum())} non-finite entries")
        except Exception as e:  # noqa: BLE001
            add("ERROR", f"signal {os.path.basename(f)} unreadable: {e}")
    # candidates
    text = str(utility) == "neg_gmean_ppl"
    for cdir in sorted(glob.glob(os.path.join(cell, "candidates", "*"))):
        name = os.path.basename(cdir)
        sp = os.path.join(cdir, "selection.npz")
        if not os.path.exists(sp):
            add("ERROR", f"{name}: selection.npz missing")
            continue
        try:
            z = np.load(sp, allow_pickle=False)
            idx = z["idx"]
            if len(np.unique(idx)) != len(idx):
                add("ERROR", f"{name}: {len(idx) - len(np.unique(idx))} duplicate indices")
            if len(idx) and (idx.min() < 0 or (n_pool is not None and idx.max() >= n_pool)):
                add("ERROR", f"{name}: indices outside the pool [0, {n_pool})")
            if "n" in z.files and n_pool is not None and int(z["n"]) != n_pool:
                add("ERROR", f"{name}: stored pool size {int(z['n'])} but splits.json pool has {n_pool}")
            if "budget" in z.files and not text and len(idx) > int(z["budget"]):
                add("ERROR", f"{name}: {len(idx)} indices exceed the budget {int(z['budget'])}")
            if "sel_sha12" in z.files and str(z["sel_sha12"]) != sel_sha12(idx):
                add("ERROR", f"{name}: sel_sha12 does not match the stored indices")
            if len(idx) == 0:
                add("ERROR", f"{name}: empty selection")
        except Exception as e:  # noqa: BLE001
            add("ERROR", f"{name}: selection.npz unreadable: {e}")
            continue
        scp = os.path.join(cdir, "scores.json")
        if not os.path.exists(scp):
            add("ERROR", f"{name}: scores.json missing")
            continue
        sc = load_json(scp)
        tu = sc.get("test_utility")
        if name in rows and finite(tu) and finite(rows[name].get("u_test")) and abs(tu - rows[name]["u_test"]) > 1e-6:
            add("ERROR", f"{name}: scores.json test utility {tu} differs from leaderboard {rows[name]['u_test']}")
        if sc.get("per_unit_written"):
            for split in ("con", "rank", "conf", "test"):
                pf = os.path.join(cdir, "per_unit", f"{split}.npz")
                if not os.path.exists(pf):
                    if split == "test" or (split in counts and counts[split] > 0):
                        add("ERROR", f"{name}: per_unit/{split}.npz missing although per_unit_written")
                    continue
                try:
                    pu = np.load(pf, allow_pickle=False)
                    for k in pu.files:
                        a = pu[k]
                        if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
                            add("ERROR", f"{name}: per_unit/{split}.npz[{k}] has non-finite values")
                except Exception as e:  # noqa: BLE001
                    add("ERROR", f"{name}: per_unit/{split}.npz unreadable: {e}")
    parts = cell.split(os.sep)
    return {"cell": cell, "batch": parts[1] if len(parts) > 1 else "", "task": "/".join(parts[-4:-1]),
            "elected": elected, "u_el": u_el, "u_rand": u_rand, "utility": utility}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_and_logs")
    ap.add_argument("--batches", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = args.root.rstrip("/")
    out = args.out or os.path.join(root, "summary", "audit")
    os.makedirs(out, exist_ok=True)
    batches = args.batches or sorted(
        d for d in os.listdir(root) if d.startswith("R") and os.path.isdir(os.path.join(root, d))
        and not d.startswith("R0") and "partial" not in d and "lesscpu" not in d)
    running = running_cells(root)
    issues: list = []
    cells: list = []
    for b in batches:
        for cfg in glob.glob(os.path.join(root, b, "**", "config.json"), recursive=True):
            if f"{os.sep}candidates{os.sep}" in cfg:
                continue
            r = audit_cell(os.path.dirname(cfg), issues, running)
            if r:
                cells.append(r)
    # batch-level outliers and below-random
    groups = defaultdict(list)
    for r in cells:
        # group by batch variant directory + task, i.e. everything but the seed directory
        groups[os.path.dirname(r["cell"])].append(r)
    for g, rs in groups.items():
        vals = [r["u_el"] for r in rs if finite(r["u_el"])]
        if len(vals) >= 4:
            med = statistics.median(vals)
            mad = statistics.median([abs(v - med) for v in vals]) or 1e-9
            for r in rs:
                if (finite(r["u_el"]) and abs(r["u_el"] - med) / (1.4826 * mad) > 4
                        and abs(r["u_el"] - med) > 0.01 * max(abs(med), 1e-9)):
                    issues.append(("WARN", r["cell"], f"elected test {r['u_el']:.4f} is an outlier against "
                                   f"the other seeds (median {med:.4f})"))
    for r in cells:
        if finite(r["u_el"]) and finite(r["u_rand"]) and r["u_el"] < r["u_rand"] - 0.02 * abs(r["u_rand"]):
            issues.append(("WARN", r["cell"], f"elected {r['elected']} {r['u_el']:.4f} "
                           f"below random {r['u_rand']:.4f} by more than 2%"))
        elif finite(r["u_el"]) and finite(r["u_rand"]) and r["u_el"] < r["u_rand"]:
            issues.append(("INFO", r["cell"], f"elected {r['elected']} {r['u_el']:.4f} below random {r['u_rand']:.4f}"))
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    per_batch = defaultdict(int)
    for r in cells:
        per_batch[r["batch"]] += 1
    sev_count = defaultdict(int)
    for s, _, _ in issues:
        sev_count[s] += 1
    lines = [f"# Run-record audit {ts}", "",
             f"Cells audited: {len(cells)} ({', '.join(f'{b} {n}' for b, n in sorted(per_batch.items()))}).",
             f"Issues: ERROR {sev_count['ERROR']}, WARN {sev_count['WARN']}, INFO {sev_count['INFO']}.", "",
             "Utility ranges use mathematical bounds. MASE > 5 and perplexity > 200 are high-error WARNs.", ""]
    for sev in ("ERROR", "WARN", "INFO"):
        items = [(c, w) for s, c, w in issues if s == sev]
        if items:
            lines += [f"## {sev} ({len(items)})", ""]
            lines += [f"- `{c}`: {w}" for c, w in sorted(items)] + [""]
    text = "\n".join(lines) + "\n"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for name in (f"AUDIT_{stamp}.md", "AUDIT_latest.md"):
        with open(os.path.join(out, name), "w") as fh:
            fh.write(text)
    with open(os.path.join(out, "audit_latest.json"), "w") as fh:
        json.dump({"time": ts, "utility_bounds_policy": "mathematical_ranges_with_high_error_warnings_v2",
                   "cells": len(cells), "per_batch": per_batch, "counts": sev_count,
                   "issues": [{"severity": s, "cell": c, "what": w} for s, c, w in issues]}, fh, indent=1)
    print(f"audit {ts}: {len(cells)} cells, ERROR {sev_count['ERROR']}, "
          f"WARN {sev_count['WARN']}, INFO {sev_count['INFO']}")


if __name__ == "__main__":
    main()
