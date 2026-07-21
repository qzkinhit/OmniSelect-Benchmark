"""Characteristic metrics v2 (rebuilt per CODEX_AUDIT_20260716_1230 item 4).

DISCLOSURE: these are SECONDARY / EXPLORATORY metrics, proposed post-hoc after initial
results were seen. They never replace the primary standard task metrics (top-1 / MASE /
macro-F1 / AUC / PPL). They are reported with bootstrap CIs and normalization sensitivity.

Design (all defects of the retracted v1 fixed):
  - Input: per-seed raw values from experiments/results_matrix.json (built by
    build_results_matrix.py from real logs, each with source log SHA). No hand copying.
  - Dimensionless utility, two normalizations reported for sensitivity:
      rank: within (dataset, seed), utility = rank percentile in [0,1] (1 = best).
      effect: within (dataset, seed), utility = (x - median) / IQR, sign-corrected so
              higher = better (robust standardized effect; falls back to std if IQR=0).
  - Regret(dataset) = oracle_utility - method_utility, oracle = best method in that
    (dataset, seed). Lower is better. This IS a regret, unlike v1's min-gain.
  - AG (adaptivity gain) = mean_utility(controller) - max over fixed methods of
    mean_utility(fixed uniformly applied), paired over (dataset, seed) cells; bootstrap
    95% CI over cells.
  - TTR (top-tier recovery, replaces v1 OSRR): fraction of datasets where the controller
    is statistically in the top tier - paired bootstrap over seeds of (best_method -
    controller) utility difference; top-tier if the 95% CI of the difference includes 0
    or controller is best. Descriptive of test results; no validation/test mixing because
    it makes no method decision.
  - WCR = worst-case (max over datasets) regret. Lower is better.
Only methods with >= MIN_SEEDS on a dataset enter that dataset's pool; coverage printed.
"""
import json
import os
import random
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = os.path.join(ROOT, "experiments", "results_matrix.json")
MIN_SEEDS = 3
CONTROLLER = "mmds_adapt"
B = 2000
random.seed(20260716)  # fixed for reproducibility of the CI itself


def load():
    mx = json.load(open(MATRIX))
    data = {}  # ds -> {method: [v_seed0, v_seed1, v_seed2]}, seeds intersected
    for ds, e in mx.items():
        if "__" in ds:  # rerun/aux views excluded from the cross-modal aggregate
            continue
        pool = {m: sv for m, sv in e["methods"].items() if len(sv) >= MIN_SEEDS}
        if CONTROLLER not in pool or len(pool) < 4:
            continue
        seeds = sorted(set.intersection(*[set(sv) for sv in pool.values()]))[:MIN_SEEDS]
        if len(seeds) < MIN_SEEDS:
            continue
        data[ds] = {"hib": e["higher_is_better"],
                    "vals": {m: [sv[s] for s in seeds] for m, sv in pool.items()},
                    "seeds": seeds, "source": e["source_log"]}
    return data


def utilities(data, kind):
    """u[ds][method][seed_idx] = dimensionless utility, higher better."""
    u = {}
    for ds, e in data.items():
        methods = list(e["vals"])
        u[ds] = {m: [] for m in methods}
        for i in range(MIN_SEEDS):
            xs = {m: e["vals"][m][i] for m in methods}
            sign = 1.0 if e["hib"] else -1.0
            sx = {m: sign * v for m, v in xs.items()}
            if kind == "rank":
                order = sorted(sx.values())
                n = len(order)
                for m in methods:
                    r = sum(1 for v in order if v < sx[m]) + 0.5 * (sum(1 for v in order if v == sx[m]) - 1)
                    u[ds][m].append(r / (n - 1) if n > 1 else 0.5)
            else:  # robust effect
                vals = sorted(sx.values())
                med = st.median(vals)
                q1 = vals[len(vals) // 4]
                q3 = vals[(3 * len(vals)) // 4]
                iqr = (q3 - q1) or (st.pstdev(vals) or 1.0)
                for m in methods:
                    u[ds][m].append((sx[m] - med) / iqr)
    return u


def common_methods(u):
    sets = [set(mm) for mm in u.values()]
    return set.intersection(*sets) if sets else set()


def analyze(u):
    ds_list = list(u)
    fixed = sorted(m for m in common_methods(u) if m != CONTROLLER)
    cells = [(ds, i) for ds in ds_list for i in range(MIN_SEEDS)]

    def mean_util(m, cs):
        return sum(u[ds][m][i] for ds, i in cs) / len(cs)

    def ag_of(cs):
        best = max(mean_util(m, cs) for m in fixed)
        return mean_util(CONTROLLER, cs) - best

    AG = ag_of(cells)
    boots = []
    for _ in range(B):
        cs = [cells[random.randrange(len(cells))] for _ in cells]
        boots.append(ag_of(cs))
    boots.sort()
    ag_ci = (boots[int(0.025 * B)], boots[int(0.975 * B)])

    # regret + WCR + TTR
    regret, ttr_detail = {}, {}
    for ds in ds_list:
        methods = list(u[ds])
        oracle = [max(u[ds][m][i] for m in methods) for i in range(MIN_SEEDS)]
        regret[ds] = {m: sum(oracle[i] - u[ds][m][i] for i in range(MIN_SEEDS)) / MIN_SEEDS
                      for m in methods}
        # paired bootstrap over seeds: diff = best_fixed_or_any - controller
        diffs = [max(u[ds][m][i] for m in methods if m != CONTROLLER) - u[ds][CONTROLLER][i]
                 for i in range(MIN_SEEDS)]
        bs = []
        for _ in range(B):
            sample = [diffs[random.randrange(MIN_SEEDS)] for _ in range(MIN_SEEDS)]
            bs.append(sum(sample) / MIN_SEEDS)
        bs.sort()
        lo, hi = bs[int(0.025 * B)], bs[int(0.975 * B)]
        ttr_detail[ds] = {"diff_mean": round(sum(diffs) / MIN_SEEDS, 4),
                          "ci": [round(lo, 4), round(hi, 4)],
                          "top_tier": lo <= 0 or sum(diffs) <= 0}
    TTR = sum(1 for d in ttr_detail.values() if d["top_tier"]) / len(ds_list)
    WCR = {m: round(max(regret[ds].get(m, float("nan")) for ds in ds_list), 4)
           for m in ([CONTROLLER] + fixed)
           if all(m in regret[ds] for ds in ds_list)}
    return {"datasets": ds_list, "n_fixed_common": len(fixed), "fixed_common": fixed,
            "AG": round(AG, 4), "AG_ci95": [round(ag_ci[0], 4), round(ag_ci[1], 4)],
            "TTR": round(TTR, 3), "TTR_detail": ttr_detail,
            "WCR": dict(sorted(WCR.items(), key=lambda kv: kv[1]))}


def main():
    data = load()
    cov = ", ".join("%s:%dm" % (ds, len(e["vals"])) for ds, e in data.items())
    print("coverage: %d datasets usable (%s)" % (len(data), cov))
    out = {"disclosure": "secondary/exploratory, proposed post-hoc; primary = standard task metrics",
           "min_seeds": MIN_SEEDS, "bootstrap_B": B}
    for kind in ("rank", "effect"):
        u = utilities(data, kind)
        out[kind] = analyze(u)
        r = out[kind]
        print(f"\n[{kind} normalization] AG={r['AG']} CI95={r['AG_ci95']}  TTR={r['TTR']}")
        print(f"  WCR (lower=better): {r['WCR']}")
    d = os.path.join(ROOT, "experiments", "characteristic_metrics_v2.json")
    json.dump(out, open(d, "w"), indent=1)
    print(f"\nsaved -> {d}")


if __name__ == "__main__":
    main()
