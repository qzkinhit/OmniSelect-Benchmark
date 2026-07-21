"""Characteristic metrics v5 (canonical-parity revision).

DISCLOSURE: SECONDARY / EXPLORATORY, post-hoc. Supersedes v4's dominance inference
(whose percentile-bootstrap p-values over only 8 clusters are approximate; v4's 10/10
is relabeled exploratory). Standard task metrics remain primary. The per-cell FINAL
main-table numbers come from the pubcore-paired same-code-state batch, not from here.

Inference design for 8 underlying-dataset clusters:
  EXACT tests, no bootstrap approximation:
  - per fixed method m: cluster-level paired difference D_c = mean over (views in
    cluster c, seeds) of u(controller) - u(m);
  - exact sign test: p = P[Binomial(n_clusters, 1/2) >= #positive] (one-sided);
  - exact sign-flip randomization: enumerate ALL 2^8 = 256 sign assignments of the
    D_c, p = fraction with mean(flipped) >= mean(observed) (one-sided, includes
    identity, so p >= 1/256);
  - Holm step-down over the fixed-method family, both tests reported.
Utilities and the common pool follow v4 (rank percentile / robust effect within the
common pool, v5-canonical controller values).
"""
import itertools
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import characteristic_metrics_v4 as v4  # noqa: E402  (load/utilities/CLUSTER reused)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "characteristic_metrics_v5.json")
CONTROLLER = v4.CONTROLLER
MIN_SEEDS = v4.MIN_SEEDS


def holm(pvals):
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def analyze(data, kind, common):
    u = v4.utilities(data, kind, common)
    views = list(u)
    clusters = sorted(set(v4.CLUSTER[v] for v in views))
    byc = {c: [v for v in views if v4.CLUSTER[v] == c] for c in clusters}
    fixed = [m for m in common if m != CONTROLLER]

    res, p_sign_list, p_flip_list = {}, [], []
    for m in fixed:
        D = []
        for c in clusters:
            diffs = [u[v][CONTROLLER][i] - u[v][m][i] for v in byc[c] for i in range(MIN_SEEDS)]
            D.append(sum(diffs) / len(diffs))
        n = len(D)
        npos = sum(1 for d in D if d > 0)
        # exact one-sided sign test (ties counted against us: zeros are not positives)
        from math import comb
        p_sign = sum(comb(n, k) for k in range(npos, n + 1)) / (2 ** n)
        # exact sign-flip randomization
        t_obs = sum(D) / n
        ge = 0
        for signs in itertools.product((1, -1), repeat=n):
            t = sum(s * d for s, d in zip(signs, D)) / n
            if t >= t_obs - 1e-15:
                ge += 1
        p_flip = ge / (2 ** n)
        res[m] = {"cluster_diffs": [round(d, 4) for d in D], "n_positive": "%d/%d" % (npos, n),
                  "mean_diff": round(t_obs, 4), "p_sign_exact": round(p_sign, 5),
                  "p_signflip_exact": round(p_flip, 5)}
        p_sign_list.append(p_sign)
        p_flip_list.append(p_flip)
    for m, ps, pf in zip(fixed, holm(p_sign_list), holm(p_flip_list)):
        res[m]["p_sign_holm"] = round(ps, 5)
        res[m]["p_signflip_holm"] = round(pf, 5)
        res[m]["significant_holm_05"] = ps < 0.05 and pf < 0.05
    n_sig = sum(1 for v in res.values() if v["significant_holm_05"])
    return {"n_clusters": len(clusters), "clusters": clusters, "common_pool": common,
            "dominance_exact": res,
            "n_significant_holm_both_tests": "%d/%d" % (n_sig, len(fixed)),
            "note": "with 8 clusters the minimum attainable exact p is 1/256=0.0039 "
                    "(sign-flip) and 1/256 (sign test at 8/8); Holm over %d methods means "
                    "the strongest attainable Holm-adjusted p is %.4f - results clearing "
                    "0.05 are exact, not approximations" % (len(fixed), (1 / 256) * len(fixed))}


def main():
    data = v4.load()
    common_set = set.intersection(*[set(e["vals"]) for e in data.values()])
    common = sorted(common_set - {CONTROLLER}) + [CONTROLLER]
    out = {"disclosure": "secondary/exploratory, post-hoc; primary = standard task metrics; "
                         "v4 dominance (percentile bootstrap) relabeled exploratory and "
                         "superseded by these EXACT cluster-level tests; FINAL per-cell "
                         "numbers come from the pubcore-paired same-code-state batch"}
    for kind in ("rank", "effect"):
        r = analyze(data, kind, common)
        out[kind] = r
        sig = {m: (v["p_sign_holm"], v["p_signflip_holm"]) for m, v in r["dominance_exact"].items()}
        print("[%s] exact dominance (Holm both tests): %s" % (kind, r["n_significant_holm_both_tests"]))
        for m, v in r["dominance_exact"].items():
            print("  vs %-15s D+=%s mean=%+.4f p_sign(H)=%.4f p_flip(H)=%.4f %s"
                  % (m, v["n_positive"], v["mean_diff"], v["p_sign_holm"],
                     v["p_signflip_holm"], "SIG" if v["significant_holm_05"] else "ns"))
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved -> %s" % OUT)


if __name__ == "__main__":
    main()
