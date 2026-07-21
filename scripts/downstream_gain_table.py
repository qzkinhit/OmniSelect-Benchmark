#!/usr/bin/env python
"""Unified downstream gain table (audit item 4).

Reads the pubcore-paired per-seed result dumps under outputs/ and, for every
method, computes per-seed PAIRED deltas within the same seed:
  (a) vs same-budget random  -> fairness comparison
  (b) vs full (no selection) -> reference only, full uses more data

No experiments are run; this script only aggregates already-produced JSONs.

Usage (on the server):
  /root/miniconda3/bin/python scripts/downstream_gain_table.py \
      --outputs /root/autodl-tmp/OmniSelect/outputs --outdir /tmp/gain_table
"""
import argparse
import glob
import hashlib
import json
import math
import os

# per-view primary metric and direction
VIEW_METRIC = {
    "vision": ("acc", "higher"),
    "timeseries": ("mase", "lower"),
    "tep": ("f1", "higher"),
    "tabular": ("auc", "higher"),
}
VIEW_LABEL = {
    "vision": "vision (cifar100, acc)",
    "timeseries": "time_etth1 (mase)",
    "tep": "tep (f1)",
    "tabular": "tabular (electricity, auc)",
}
# two-sided t critical value for df=2, 95%
T_CRIT_DF2 = 4.302652729911275

FORMULAS = {
    "delta_vs_random_pp (acc/auc/f1)": "100 * (metric_method[seed] - metric_random[seed]), paired within seed; positive = method better",
    "delta_vs_random_abs (mase)": "mase_random[seed] - mase_method[seed], paired within seed; positive = method better (MASE reduction)",
    "delta_vs_random_rel_pct (mase)": "100 * (mase_random[seed] - mase_method[seed]) / mase_random[seed]",
    "gap_vs_full_pp (acc/auc/f1)": "100 * (metric_method[seed] - metric_full[seed]); full uses more data, reference only",
    "gap_vs_full_abs (mase)": "mase_method[seed] - mase_full[seed]; positive = method worse than full; full uses more data, reference only",
    "mean/std": "sample mean and sample std (ddof=1) over the n=3 paired per-seed deltas",
    "ci95": "mean +- t_{0.975,df=2} * std / sqrt(3), t = 4.30265 (n=3 seeds)",
}

DISCLOSURE = ("full/noselect uses more data - reference only; fairness judgments "
              "vs same-budget random and same-budget baselines")

TEXT_EXCLUSION = ("text view excluded: QZ3 lane and text-controls lane still running; "
                  "will be added once QZ3+controls close")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mean_std_ci(vals):
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, None, [None, None]
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    half = T_CRIT_DF2 * sd / math.sqrt(n) if n == 3 else None
    ci = [m - half, m + half] if half is not None else [None, None]
    return m, sd, ci


def load_runs(outputs_root):
    pattern = os.path.join(outputs_root, "*", "*", "run_id=pubcore-paired-*", "seed_*", "results.json")
    files = sorted(glob.glob(pattern))
    runs = []
    for f in files:
        rel = os.path.relpath(f, outputs_root)
        view = rel.split(os.sep)[0]
        if view not in VIEW_METRIC:
            continue
        with open(f) as fh:
            dump = json.load(fh)  # _trial_dump dict
        runs.append({
            "path": f,
            "rel": rel,
            "sha256": sha256_file(f),
            "view": view,
            "seed": dump.get("seed"),
            "results": dump["results"],
        })
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--outdir", default="experiments")
    args = ap.parse_args()

    runs = load_runs(args.outputs)
    os.makedirs(args.outdir, exist_ok=True)

    views = {}
    source_files = []
    for view in sorted({r["view"] for r in runs}):
        vruns = sorted([r for r in runs if r["view"] == view], key=lambda r: r["seed"])
        metric, direction = VIEW_METRIC[view]
        seeds = [r["seed"] for r in vruns]
        # per-seed method -> metric value
        per_seed = []
        for r in vruns:
            table = {e["method"]: e for e in r["results"]}
            per_seed.append(table)
            source_files.append({"file": r["rel"], "sha256": r["sha256"],
                                 "view": view, "seed": r["seed"]})
        methods = [m for m in per_seed[0].keys()]
        # keep only methods present in every seed
        methods = [m for m in methods if all(m in t for t in per_seed)]

        mrows = {}
        for m in methods:
            vals = [t[m][metric] for t in per_seed]
            rnd = [t["random"][metric] for t in per_seed]
            ful = [t["full"][metric] for t in per_seed]
            ns = [t[m]["n"] for t in per_seed]
            row = {
                "n_selected_per_seed": ns,
                "metric_per_seed": vals,
                "seeds": seeds,
            }
            if metric == "mase":
                d_abs = [rnd[i] - vals[i] for i in range(len(vals))]
                d_rel = [100.0 * (rnd[i] - vals[i]) / rnd[i] for i in range(len(vals))]
                g_full = [vals[i] - ful[i] for i in range(len(vals))]
                mu, sd, ci = mean_std_ci(d_abs)
                row["delta_vs_random_abs"] = {"per_seed": d_abs, "mean": mu, "std": sd, "ci95": ci}
                mu, sd, ci = mean_std_ci(d_rel)
                row["delta_vs_random_rel_pct"] = {"per_seed": d_rel, "mean": mu, "std": sd, "ci95": ci}
                mu, sd, ci = mean_std_ci(g_full)
                row["gap_vs_full_abs"] = {"per_seed": g_full, "mean": mu, "std": sd, "ci95": ci}
            else:
                d_pp = [100.0 * (vals[i] - rnd[i]) for i in range(len(vals))]
                g_pp = [100.0 * (vals[i] - ful[i]) for i in range(len(vals))]
                mu, sd, ci = mean_std_ci(d_pp)
                row["delta_vs_random_pp"] = {"per_seed": d_pp, "mean": mu, "std": sd, "ci95": ci}
                mu, sd, ci = mean_std_ci(g_pp)
                row["gap_vs_full_pp"] = {"per_seed": g_pp, "mean": mu, "std": sd, "ci95": ci}
            mrows[m] = row
        views[view] = {
            "label": VIEW_LABEL[view],
            "metric": metric,
            "direction": direction,
            "n_seeds": len(vruns),
            "methods": mrows,
        }

    out = {
        "task": ("pubcore main-table paired gains (audit item 4; scope = the four pubcore "
                 "main-table views ONLY - vision/etth1/tep/electricity; NOT an all-paper "
                 "unified gain table; text views join after QZ3+controls close)"),
        "generated_by": "scripts/downstream_gain_table.py",
        "disclosure": DISCLOSURE,
        "text_view_status": TEXT_EXCLUSION,
        "pairing": "all deltas are paired within the same seed (same pool, same noise realization, same fit_seed)",
        "n_seeds": 3,
        "t_crit_df2_0975": T_CRIT_DF2,
        "formulas": FORMULAS,
        "source_files": source_files,
        "views": views,
    }
    json_path = os.path.join(args.outdir, "downstream_gain_table.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=1)

    # markdown
    def fmt(x, nd=2):
        return ("{:+." + str(nd) + "f}").format(x)

    lines = ["# Unified downstream gain table (pubcore-paired, 3 seeds)", "",
             "Paired per-seed deltas. " + DISCLOSURE + ".",
             "", "Text view excluded (QZ3 + text-controls lanes still running).", ""]
    for view in ["vision", "timeseries", "tep", "tabular"]:
        if view not in views:
            continue
        v = views[view]
        lines.append("## " + v["label"])
        lines.append("")
        if v["metric"] == "mase":
            lines.append("| method | dMASE vs random (abs, +=better) mean+-std [95% CI] | dMASE vs random (rel %) | gap vs full (abs, +=worse) |")
            lines.append("|---|---|---|---|")
            for m, row in v["methods"].items():
                if m == "full":
                    continue
                a = row["delta_vs_random_abs"]; rl = row["delta_vs_random_rel_pct"]; g = row["gap_vs_full_abs"]
                lines.append("| {} | {} +- {:.3f} [{} , {}] | {} +- {:.2f} | {} |".format(
                    m, fmt(a["mean"], 3), a["std"], fmt(a["ci95"][0], 3), fmt(a["ci95"][1], 3),
                    fmt(rl["mean"], 2), rl["std"], fmt(g["mean"], 3)))
        else:
            lines.append("| method | d{} vs random (pp, +=better) mean+-std [95% CI] | gap vs full (pp) |".format(v["metric"]))
            lines.append("|---|---|---|")
            for m, row in v["methods"].items():
                if m == "full":
                    continue
                a = row["delta_vs_random_pp"]; g = row["gap_vs_full_pp"]
                lines.append("| {} | {} +- {:.2f} [{} , {}] | {} |".format(
                    m, fmt(a["mean"]), a["std"], fmt(a["ci95"][0]), fmt(a["ci95"][1]), fmt(g["mean"])))
        lines.append("")
    md_path = os.path.join(args.outdir, "downstream_gain_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
