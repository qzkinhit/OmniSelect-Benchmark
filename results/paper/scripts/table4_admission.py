"""Table 4 of the paper: share (%) of each kind of dirty record that a strategy selects.

For every reported run of a task, the share of a kind is (selected pool records of the kind) / (pool records of the
kind), and a cell is the mean over the runs of the task. Where a run record carries candidates/<dir>/selection.npz, the
shares are counted per kind. ETTh1, ETTh2, CSTR, and Steam carry no selection files, so their single column (all)
pools the dirty kinds through the fields n and purity of the run record: n * (1 - purity) / (pool records not tagged
clean). The Text kinds come from pool_mechanisms (truncation TR, template TP, crossdomain DM, lowtier LQ).

Rows:
  Random         random selection of the run (random)
  Auth-only      auth_only
  Coverage-only  coreset (coverage_text on Text)
  Infl-only      influence_only
  Best baseline  on each task, the external baseline with the best mean in the main table (paper_numbers.json, means),
                 chosen among Herding, Density, QuaDMix, DMF, EL2N, GraNd, and CCS; on the forecasting tasks and Text,
                 EL2N, GraNd, and CCS are the /adapt versions, whose records are the separate runs in main_adapt/
  Coop (ours)    coop gate=authenticity rho=1.3 within=kcenter, the cooperative strategy of Definition 4.1 with
                 authenticity admission and k-center selection, built by OmniSelect on every task
Column minima are taken over the rows other than Random after rounding to one decimal, as printed. The column Mean
averages the 21 columns of a row.

Usage: python3 table4_admission.py <directory holding main/ and main_adapt/> <paper_numbers.json> <output json>
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

REC = Path(sys.argv[1])
NUMBERS = Path(sys.argv[2])
OUT = Path(sys.argv[3])

TASKS = {"C-100": "vision/cifar100/clip_vitb32", "C-100N": "vision/cifar100n/clip_vitb32",
         "ETTh1": "timeseries/ETTh1/dlinear", "ETTm1": "timeseries/ETTm1/dlinear", "ETTh2": "timeseries/ETTh2/dlinear",
         "CSTR": "timeseries/daisy_cstr/dlinear", "Steam": "timeseries/daisy_steamgen/dlinear",
         "TEP21": "process/tep21/mlp", "Elec": "tabular/electricity/tabpfn", "Text": "text/five_domain/smollm2_135m"}
COLUMNS = [("C-100", "LE"), ("C-100", "ND"), ("C-100N", "LE"), ("ETTh1", "all"),
           ("ETTm1", "IN"), ("ETTm1", "FS"), ("ETTm1", "SW"), ("ETTm1", "ND"), ("ETTh2", "all"), ("CSTR", "all"),
           ("Steam", "all"), ("TEP21", "LE"), ("TEP21", "FN"), ("TEP21", "ND"), ("Elec", "LE"), ("Elec", "FN"),
           ("Elec", "ND"), ("Text", "TR"), ("Text", "TP"), ("Text", "DM"), ("Text", "LQ")]
ROWS = ["Random", "Auth-only", "Coverage-only", "Infl-only", "Best baseline", "Coop (ours)"]
TS = {"ETTh1", "ETTm1", "ETTh2", "CSTR", "Steam"}
LOWER_IS_BETTER = TS | {"Text"}
EXTERNAL = ["herding", "density", "quadmix_pub", "dmf_pub", "el2n", "grand", "ccs"]
ADAPTED = {"el2n", "grand", "ccs"}   # adapted to forecasting and text, evaluated in the main_adapt runs
TAG = {"flip": "LE", "dup": "ND", "flat": "FS", "shuffle": "SW",
       "truncation": "TR", "template": "TP", "crossdomain": "DM", "lowtier": "LQ"}


def best_external(means, task):
    m = {k: means[task][k] for k in EXTERNAL if means[task].get(k) is not None}
    pick = min if task in LOWER_IS_BETTER else max
    return pick(m, key=m.get)


def record_name(task, key):
    """(record batch, strategy name in the run record) of a main-table key on a task."""
    text = task == "Text"
    if key in ADAPTED and (task in TS or text):
        return "main_adapt", f"{key}_adapt"
    if text and key in ("herding", "density", "coreset"):
        return "main", {"herding": "herding_text", "density": "density_text", "coreset": "coverage_text"}[key]
    return "main", key


def kinds_of(task, sp):
    if task == "Text":
        return np.array([TAG.get(m, "clean") if m else "clean" for m in sp["pool_mechanisms"]])
    out = []
    for t in sp["pool_tags"]:
        if t == "high":
            out.append("clean")
        elif t == "corrupt":
            out.append("IN" if task in TS else "FN")
        else:
            out.append(TAG.get(t, t))
    return np.array(out)


def run_rows(run):
    lb = json.loads(Path(run, "leaderboard.json").read_text())
    rows = {r["name"]: r for r in lb["rows"]}
    if not rows:   # the main_adapt runs keep their method rows in metrics.json
        rows = json.loads(Path(run, "metrics.json").read_text())["rows"]
    return lb, rows


def selection(run, lb, name):
    cdir = Path(run, "candidates")
    for d in (lb.get("candidate_dirs", {}).get(name), name, name.replace(" ", "_")):
        if d and Path(cdir, d, "selection.npz").is_file():
            return np.load(Path(cdir, d, "selection.npz"))["idx"]
    return None


def task_shares(task, batch, name):
    """Mean share (%) per kind over the runs of a task, plus the pooled share (all) of the dirty kinds."""
    per = []
    runs = sorted(glob.glob(str(REC / batch / TASKS[task] / "seed_*")))
    for run in runs:
        sp = json.loads(Path(run, "splits.json").read_text())
        lb, rows = run_rows(run)
        kinds = kinds_of(task, sp)
        idx = selection(run, lb, name)
        if idx is not None:
            sel = np.zeros(len(kinds), bool)
            sel[np.asarray(idx, int)] = True
            d = {k: 100 * float(sel[kinds == k].mean()) for k in set(kinds.tolist()) if k != "clean"}
            d["all"] = 100 * float(sel[kinds != "clean"].mean())
        else:
            r = rows[name]
            d = {"all": 100 * r["n"] * (1 - r["purity"]) / int((kinds != "clean").sum())}
        per.append(d)
    keys = set.intersection(*(set(d) for d in per))
    return {k: sum(d[k] for d in per) / len(per) for k in keys}


def main():
    means = json.loads(NUMBERS.read_text())["means"]
    best = {t: best_external(means, t) for t in TASKS}
    fixed = {"Random": "random", "Auth-only": "auth_only", "Coverage-only": "coreset", "Infl-only": "influence_only",
             "Coop (ours)": "coop gate=authenticity rho=1.3 within=kcenter"}
    vals, cache = {}, {}
    for row in ROWS:
        vals[row] = []
        for task, kind in COLUMNS:
            batch, name = record_name(task, best[task] if row == "Best baseline" else fixed[row])
            if (task, batch, name) not in cache:
                cache[(task, batch, name)] = task_shares(task, batch, name)
            d = cache[(task, batch, name)]
            if kind not in d:
                raise SystemExit(f"{row} on {task}: no per-kind selection for {kind} in {batch}/{name}")
            vals[row].append(d[kind])
    printed = {r: [float(f"{v:.1f}") for v in vals[r]] for r in ROWS}
    rivals = ROWS[1:]
    minima = [min(printed[r][j] for r in rivals) for j in range(len(COLUMNS))]
    mean = {r: sum(vals[r]) / len(vals[r]) for r in ROWS}
    result = {
        "definition": __doc__.split("Usage")[0].strip(),
        "columns": [f"{t}/{k}" for t, k in COLUMNS],
        "best_baseline": {t: {"main_table_key": k, "record": "/".join(record_name(t, k))} for t, k in best.items()},
        "share_pct": vals,
        "printed": printed,
        "mean_pct": mean,
        "mean_printed": {r: float(f"{mean[r]:.1f}") for r in ROWS},
        "column_minima": {r: [f"{t}/{k}" for j, (t, k) in enumerate(COLUMNS) if printed[r][j] == minima[j]] for r in rivals},
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print("best baseline:", {t: k for t, k in best.items()})
    for r in ROWS:
        print(f"{r:13s}", " ".join(f"{v:5.1f}" for v in printed[r]), f"| {result['mean_printed'][r]:.1f}")


if __name__ == "__main__":
    main()
