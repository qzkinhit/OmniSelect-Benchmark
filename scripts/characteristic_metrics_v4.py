"""Characteristic metrics v4 (frozen metric gate).

DISCLOSURE: SECONDARY / EXPLORATORY, post-hoc; primary metrics remain the standard task
metrics. Supersedes v3 (kept as diagnostic).

Changes over v3:
  1. Bootstrap clusters by UNDERLYING DATASET, not by view: ETTh1-DLinear and
     ETTh1-Chronos are the same cluster; CIFAR-100 and CIFAR-100N share the CIFAR-100
     image cluster. Resample clusters -> (all views of a sampled cluster) -> seeds.
     Leave-one-CLUSTER-out sensitivity.
  2. Utilities (rank percentile / robust effect), the regret oracle, and AG/dominance/WCR
     are computed WITHIN THE COMMON 7-METHOD POOL ONLY (6 fixed + controller). Partial-
     coverage methods are excluded from that pool and reported separately: each is ranked
     inside {common 7 + itself} on its applicable views.
  3. The 6 dominance tests get HOLM correction (bootstrap one-sided p-values, step-down);
     both raw and Holm-adjusted verdicts reported.
"""
import json
import os
import random
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = os.path.join(ROOT, "experiments", "results_matrix.json")
OUT = os.path.join(ROOT, "experiments", "characteristic_metrics_v4.json")
MIN_SEEDS = 3
CONTROLLER = "mmds_adapt"
B = 10000
random.seed(20260716)

# view -> underlying-dataset cluster
CLUSTER = {
    "vision_cifar100": "cifar100", "vision_cifar100n": "cifar100",
    "time_etth1": "etth1", "time_etth1_chronos": "etth1",
    "time_etth2": "etth2", "time_etth2_chronos": "etth2",
    "time_ettm1_chronos": "ettm1", "time_ettm1": "ettm1",
    "time_daisy_cstr": "daisy_cstr", "time_daisy_cstr_chronos": "daisy_cstr",
    "time_daisy_steamgen": "daisy_steamgen", "time_daisy_steamgen_chronos": "daisy_steamgen",
    "process_tep": "tep", "tabular_electricity": "electricity",
}


def load():
    mx = json.load(open(MATRIX))
    # controller rows use the same v5-canonical priority as the paper tables so the
    # analysis and the manuscripts share one source (canonical_paper_tables.py rule)
    from canonical_paper_tables import v5_controller_cells
    v5c = v5_controller_cells()
    # quadmix style proxy is PROTOCOL_INVALID_DUPLICATE_IDS (withdrawn) - it must not
    # pollute the common pool, utilities, oracle or dominance tests (audit 1215 item 3)
    for e in mx.values():
        if isinstance(e, dict):
            (e.get("methods") or {}).pop("quadmix", None)
    data = {}
    for ds, e in mx.items():
        if "__" in ds or ds not in CLUSTER:
            continue
        pool = {m: sv for m, sv in e["methods"].items() if len(sv) >= MIN_SEEDS}
        if ds in v5c and len(v5c[ds]) >= MIN_SEEDS:
            pool[CONTROLLER] = v5c[ds]
        if CONTROLLER not in pool or len(pool) < 4:
            continue
        seeds = sorted(set.intersection(*[set(sv) for sv in pool.values()]))[:MIN_SEEDS]
        if len(seeds) < MIN_SEEDS:
            continue
        data[ds] = {"hib": e["higher_is_better"],
                    "vals": {m: [sv[s] for s in seeds] for m, sv in pool.items()}}
    return data


def rank_util(sx, m):
    order = sorted(sx.values())
    n = len(order)
    r = sum(1 for v in order if v < sx[m]) + 0.5 * (sum(1 for v in order if v == sx[m]) - 1)
    return r / (n - 1) if n > 1 else 0.5


def effect_util(sx, m):
    vals = sorted(sx.values())
    med = st.median(vals)
    q1, q3 = vals[len(vals) // 4], vals[(3 * len(vals)) // 4]
    iqr = (q3 - q1) or (st.pstdev(vals) or 1.0)
    return (sx[m] - med) / iqr


def utilities(data, kind, pool_methods):
    """Utilities computed WITHIN pool_methods only (common-7 discipline)."""
    fn = rank_util if kind == "rank" else effect_util
    u = {}
    for ds, e in data.items():
        methods = [m for m in pool_methods if m in e["vals"]]
        if len(methods) < len(pool_methods):
            continue  # common pool must be fully present
        u[ds] = {m: [] for m in methods}
        for i in range(MIN_SEEDS):
            sign = 1.0 if e["hib"] else -1.0
            sx = {m: sign * e["vals"][m][i] for m in methods}
            for m in methods:
                u[ds][m].append(fn(sx, m))
    return u


def cluster_boot(views, stat, B=B):
    """Cluster bootstrap: resample underlying-dataset clusters, keep all views of each
    sampled cluster, resample seeds within each view. stat(sample) with sample =
    [(view, [seed_idx,...]), ...]."""
    clusters = sorted(set(CLUSTER[v] for v in views))
    byc = {c: [v for v in views if CLUSTER[v] == c] for c in clusters}
    base = stat([(v, list(range(MIN_SEEDS))) for v in views])
    reps = []
    for _ in range(B):
        cs = [clusters[random.randrange(len(clusters))] for _ in clusters]
        sample = [(v, [random.randrange(MIN_SEEDS) for _ in range(MIN_SEEDS)])
                  for c in cs for v in byc[c]]
        reps.append(stat(sample))
    reps.sort()
    return base, (reps[int(0.025 * B)], reps[int(0.975 * B)]), reps


def holm(pvals):
    """Holm step-down; returns adjusted p-values in original order."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def analyze(data, kind, common):
    u = utilities(data, kind, common)
    views = list(u)
    fixed = [m for m in common if m != CONTROLLER]

    def mean_u(m, sample):
        vals = [u[v][m][i] for v, idxs in sample for i in idxs]
        return sum(vals) / len(vals)

    def ag_stat(sample):
        return mean_u(CONTROLLER, sample) - max(mean_u(m, sample) for m in fixed)
    AG, ag_ci, _ = cluster_boot(views, ag_stat)

    clusters = sorted(set(CLUSTER[v] for v in views))
    loco = {}
    for drop in clusters:
        keep = [v for v in views if CLUSTER[v] != drop]
        loco[drop] = round(ag_stat([(v, list(range(MIN_SEEDS))) for v in keep]), 4)

    dominance, pvals = {}, []
    for m in fixed:
        def d_stat(sample, m=m):
            vals = [u[v][CONTROLLER][i] - u[v][m][i] for v, idxs in sample for i in idxs]
            return sum(vals) / len(vals)
        d, ci, reps = cluster_boot(views, d_stat)
        p = max(1, sum(1 for r in reps if r <= 0)) / len(reps)  # one-sided, floor 1/B
        pvals.append(p)
        dominance[m] = {"mean_diff": round(d, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)],
                        "p_raw": round(p, 5)}
    for m, p_adj in zip(fixed, holm(pvals)):
        dominance[m]["p_holm"] = round(p_adj, 5)
        dominance[m]["significant_holm_05"] = p_adj < 0.05
    n_sig = sum(1 for v in dominance.values() if v["significant_holm_05"])

    best = top2 = 0
    pctls = []
    for v in views:
        e = data[v]
        sign = 1.0 if e["hib"] else -1.0
        for i in range(MIN_SEEDS):
            sx = {m: sign * e["vals"][m][i] for m in common}
            others = sorted((val for m, val in sx.items() if m != CONTROLLER), reverse=True)
            best += sx[CONTROLLER] >= others[0]
            top2 += sx[CONTROLLER] >= others[1]
            pctls.append(u[v][CONTROLLER][i])
    n_cells = len(views) * MIN_SEEDS

    regret = {}
    for v in views:
        oracle = [max(u[v][m][i] for m in common) for i in range(MIN_SEEDS)]
        regret[v] = {m: sum(oracle[i] - u[v][m][i] for i in range(MIN_SEEDS)) / MIN_SEEDS
                     for m in common}
    WCR = {m: round(max(regret[v][m] for v in views), 4) for m in common}

    # partial-coverage methods: ranked inside {common 7 + itself} on applicable views
    partial = {}
    all_methods = set(m for v in views for m in data[v]["vals"])
    fn = rank_util if kind == "rank" else effect_util
    for pm in sorted(all_methods - set(common)):
        vs = [v for v in views if pm in data[v]["vals"]]
        if not vs:
            continue
        us, wins = [], 0
        for v in vs:
            e = data[v]
            sign = 1.0 if e["hib"] else -1.0
            for i in range(MIN_SEEDS):
                sx = {m: sign * e["vals"][m][i] for m in common + [pm]}
                us.append(fn(sx, pm))
                wins += sx[pm] >= sx[CONTROLLER]
        partial[pm] = {"n_views": len(vs), "mean_utility_in_common7_pool": round(sum(us) / len(us), 4),
                       "beats_controller_cells": "%d/%d" % (wins, len(vs) * MIN_SEEDS)}

    return {"views": views, "n_clusters": len(clusters), "common_pool": common,
            "scope_note": "AG/dominance/WCR computed strictly within the common %d-method pool on %d views (%d underlying-dataset clusters); partial-coverage methods reported separately" % (len(common), len(views), len(clusters)),
            "AG": round(AG, 4), "AG_ci95_cluster": [round(ag_ci[0], 4), round(ag_ci[1], 4)],
            "AG_leave_one_cluster_out": loco,
            "AG_loco_range": [min(loco.values()), max(loco.values())],
            "dominance_holm": dominance,
            "n_significant_holm": "%d/%d" % (n_sig, len(fixed)),
            "descriptive_ranks": {"frac_best": round(best / n_cells, 3),
                                  "frac_top2": round(top2 / n_cells, 3),
                                  "median_rank_percentile": round(st.median(pctls), 3),
                                  "n_cells": n_cells},
            "WCR_common_pool": dict(sorted(WCR.items(), key=lambda kv: kv[1])),
            "partial_coverage_separate": partial}


def main():
    data = load()
    # common pool = methods present on ALL usable views
    common_set = set.intersection(*[set(e["vals"]) for e in data.values()])
    common = sorted(common_set - {CONTROLLER}) + [CONTROLLER]
    out = {"disclosure": "secondary/exploratory, post-hoc; primary = standard task metrics; "
                         "v3 kept as diagnostic; dominance Holm-corrected",
           "min_seeds": MIN_SEEDS,
           "bootstrap": "cluster (underlying dataset -> views -> seeds), B=%d" % B}
    for kind in ("rank", "effect"):
        r = analyze(data, kind, common)
        out[kind] = r
        print("[%s] AG=%s clusterCI=%s LOCO_range=%s" % (kind, r["AG"], r["AG_ci95_cluster"], r["AG_loco_range"]))
        print("  dominance(Holm): %s; ranks: best %.0f%% top2 %.0f%% of %d cells" %
              (r["n_significant_holm"], 100 * r["descriptive_ranks"]["frac_best"],
               100 * r["descriptive_ranks"]["frac_top2"], r["descriptive_ranks"]["n_cells"]))
        print("  WCR(common pool): %s" % r["WCR_common_pool"])
        print("  partial(separate): %s" % {k: v["beats_controller_cells"] for k, v in r["partial_coverage_separate"].items()})
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved -> %s" % OUT)


if __name__ == "__main__":
    main()
