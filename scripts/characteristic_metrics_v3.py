"""Characteristic metrics v3 (statistics-corrected revision).

DISCLOSURE (unchanged): SECONDARY / EXPLORATORY, proposed post-hoc; primary metrics remain
the standard task metrics. v2 JSON is kept as a diagnostic; this file is the paper source.

Fixes over v2:
  1. TTR removed. Replaced by (a) DESCRIPTIVE per-seed rank statistics (fraction of
     (view,seed) cells where the controller is best / top-2, median rank percentile) with
     no per-seed max-competitor re-selection, and (b) PAIRED dominance tests against the
     PRE-FIXED competitor set (the 6 strategies comparable on all views), hierarchical
     bootstrap CI per competitor.
  2. No degenerate 3-seed per-dataset bootstrap intervals are used for claims.
  3. AG uses a HIERARCHICAL (cluster) bootstrap: resample datasets (views) with
     replacement, then resample seeds within each sampled view, B=10000. Plus
     leave-one-view-out (LOVO) sensitivity.
  4. Scope is explicit: AG/WCR/dominance cover ONLY the 6 strategies present on all
     views ("fully-comparable set"). Partial-coverage methods (dmf, density, semdedup,
     ccs, el2n, grand, quadmix, ...) get a symmetric applicability-masked report (mean
     utility / mean regret / worst regret over the views where they ran, with n_views).
     No "beats all baselines" claim is licensed by this file.
"""
import json
import os
import random
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = os.path.join(ROOT, "experiments", "results_matrix.json")
OUT = os.path.join(ROOT, "experiments", "characteristic_metrics_v3.json")
MIN_SEEDS = 3
CONTROLLER = "mmds_adapt"
B = 10000
random.seed(20260716)


def load():
    mx = json.load(open(MATRIX))
    data = {}
    for ds, e in mx.items():
        if "__" in ds:
            continue
        pool = {m: sv for m, sv in e["methods"].items() if len(sv) >= MIN_SEEDS}
        if CONTROLLER not in pool or len(pool) < 4:
            continue
        seeds = sorted(set.intersection(*[set(sv) for sv in pool.values()]))[:MIN_SEEDS]
        if len(seeds) < MIN_SEEDS:
            continue
        data[ds] = {"hib": e["higher_is_better"],
                    "vals": {m: [sv[s] for s in seeds] for m, sv in pool.items()}}
    return data


def utilities(data, kind):
    u = {}
    for ds, e in data.items():
        methods = list(e["vals"])
        u[ds] = {m: [] for m in methods}
        for i in range(MIN_SEEDS):
            sign = 1.0 if e["hib"] else -1.0
            sx = {m: sign * e["vals"][m][i] for m in methods}
            if kind == "rank":
                order = sorted(sx.values())
                n = len(order)
                for m in methods:
                    r = sum(1 for v in order if v < sx[m]) + 0.5 * (sum(1 for v in order if v == sx[m]) - 1)
                    u[ds][m].append(r / (n - 1) if n > 1 else 0.5)
            else:
                vals = sorted(sx.values())
                med = st.median(vals)
                q1, q3 = vals[len(vals) // 4], vals[(3 * len(vals)) // 4]
                iqr = (q3 - q1) or (st.pstdev(vals) or 1.0)
                for m in methods:
                    u[ds][m].append((sx[m] - med) / iqr)
    return u


def hier_boot(views, stat, B=B):
    """Hierarchical bootstrap: resample views with replacement, then seeds within each.
    stat(sample) where sample = list of (view_name, [seed_idx,...])."""
    base = stat([(v, list(range(MIN_SEEDS))) for v in views])
    reps = []
    for _ in range(B):
        vs = [views[random.randrange(len(views))] for _ in views]
        sample = [(v, [random.randrange(MIN_SEEDS) for _ in range(MIN_SEEDS)]) for v in vs]
        reps.append(stat(sample))
    reps.sort()
    return base, (reps[int(0.025 * B)], reps[int(0.975 * B)])


def analyze(u, data):
    views = list(u)
    fixed = sorted(set.intersection(*[set(mm) for mm in u.values()]) - {CONTROLLER})

    def mean_u(m, sample):
        vals = [u[v][m][i] for v, idxs in sample for i in idxs]
        return sum(vals) / len(vals)

    # --- AG with hierarchical bootstrap + LOVO ---
    def ag_stat(sample):
        return mean_u(CONTROLLER, sample) - max(mean_u(m, sample) for m in fixed)
    AG, ag_ci = hier_boot(views, ag_stat)
    lovo = {}
    for drop in views:
        keep = [v for v in views if v != drop]
        lovo[drop] = round(ag_stat([(v, list(range(MIN_SEEDS))) for v in keep]), 4)

    # --- paired dominance vs each PRE-FIXED competitor ---
    dominance = {}
    for m in fixed:
        def d_stat(sample, m=m):
            vals = [u[v][CONTROLLER][i] - u[v][m][i] for v, idxs in sample for i in idxs]
            return sum(vals) / len(vals)
        d, ci = hier_boot(views, d_stat)
        dominance[m] = {"mean_diff": round(d, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)],
                        "significantly_better": ci[0] > 0}
    n_sig = sum(1 for v in dominance.values() if v["significantly_better"])

    # --- descriptive per-seed ranks (raw values, no competitor re-selection) ---
    best_cells = top2_cells = 0
    pctls = []
    for v in views:
        e = data[v]
        sign = 1.0 if e["hib"] else -1.0
        for i in range(MIN_SEEDS):
            sx = {m: sign * e["vals"][m][i] for m in e["vals"]}
            c = sx[CONTROLLER]
            others = sorted((val for m, val in sx.items() if m != CONTROLLER), reverse=True)
            if c >= others[0]:
                best_cells += 1
            if c >= others[1]:
                top2_cells += 1
            pctls.append(u[v][CONTROLLER][i])
    n_cells = len(views) * MIN_SEEDS

    # --- WCR over the fully-comparable set (scope-labeled) ---
    regret = {}
    for v in views:
        methods = list(u[v])
        oracle = [max(u[v][m][i] for m in methods) for i in range(MIN_SEEDS)]
        regret[v] = {m: sum(oracle[i] - u[v][m][i] for i in range(MIN_SEEDS)) / MIN_SEEDS
                     for m in methods}
    WCR = {m: round(max(regret[v][m] for v in views), 4) for m in [CONTROLLER] + fixed}

    # --- symmetric applicability-masked report for ALL methods (incl. partial coverage) ---
    union = sorted(set(m for v in views for m in u[v]))
    masked = {}
    for m in union:
        vs = [v for v in views if m in u[v]]
        masked[m] = {"n_views": len(vs),
                     "mean_utility": round(sum(u[v][m][i] for v in vs for i in range(MIN_SEEDS)) / (len(vs) * MIN_SEEDS), 4),
                     "mean_regret": round(sum(regret[v][m] for v in vs) / len(vs), 4),
                     "worst_regret": round(max(regret[v][m] for v in vs), 4)}

    return {"views": views, "fully_comparable_set": fixed,
            "scope_note": "AG/WCR/dominance cover ONLY the %d strategies present on all %d views; partial-coverage methods are in applicability_masked" % (len(fixed), len(views)),
            "AG": round(AG, 4), "AG_ci95_hier": [round(ag_ci[0], 4), round(ag_ci[1], 4)],
            "AG_lovo": lovo, "AG_lovo_range": [min(lovo.values()), max(lovo.values())],
            "dominance_vs_prefixed": dominance,
            "n_significantly_dominated": "%d/%d" % (n_sig, len(fixed)),
            "descriptive_ranks": {"frac_best": round(best_cells / n_cells, 3),
                                  "frac_top2": round(top2_cells / n_cells, 3),
                                  "median_rank_percentile": round(st.median(pctls), 3),
                                  "n_cells": n_cells},
            "WCR_fully_comparable": dict(sorted(WCR.items(), key=lambda kv: kv[1])),
            "applicability_masked": masked}


def main():
    data = load()
    out = {"disclosure": "secondary/exploratory, post-hoc; primary = standard task metrics; "
                         "v2 JSON retained as diagnostic only",
           "min_seeds": MIN_SEEDS, "bootstrap": "hierarchical (views then seeds), B=%d" % B}
    for kind in ("rank", "effect"):
        u = utilities(data, kind)
        out[kind] = analyze(u, data)
        r = out[kind]
        print("[%s] AG=%s hierCI=%s LOVO_range=%s" % (kind, r["AG"], r["AG_ci95_hier"], r["AG_lovo_range"]))
        print("  dominance: %s significantly beaten; ranks: best %.0f%%, top2 %.0f%% of %d cells, median pctl %.2f"
              % (r["n_significantly_dominated"], 100 * r["descriptive_ranks"]["frac_best"],
                 100 * r["descriptive_ranks"]["frac_top2"], r["descriptive_ranks"]["n_cells"],
                 r["descriptive_ranks"]["median_rank_percentile"]))
        print("  WCR(fully-comparable): %s" % r["WCR_fully_comparable"])
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved -> %s" % OUT)


if __name__ == "__main__":
    main()
