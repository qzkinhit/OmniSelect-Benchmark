"""Theorem-6 scaling scatter: measured performance gap vs sensitivity-flip magnitude.

Theorem 6 predicts the gap between measured allocation and ANY single global fixed
weighting equals the angular dispersion of the per-arm sensitivities (the flip
magnitude): arms whose best-channel profile is far from the global fixed direction
should show a large measured gap, arms aligned with it should tie.

x-axis: 1 - cos(theta) between the arm's measured channel-gain direction and the
        single global fixed direction (the average direction across arms, i.e. the
        best one-size-fits-all weighting the fixed paradigm could hope for).
y-axis: measured gap on that arm = (controller - fixed-weight fusion), normalized by
        the arm's (best - random) range so heterogeneous metrics are comparable.

Numbers are the frozen multi-seed means from experiments/*.log (the same values in
the paper's tables); each entry names its source log. New server lanes append rows.
Usage: .venv/bin/python scripts/thm6_scatter.py  -> prints table + writes
       papers/mmdataselect/submissions/aaai2027/figures/fig_thm6_scatter.pdf
"""
from __future__ import annotations

import os

import numpy as np

# (auth_only, influence_only, coverage_only, random, fixed_fusion, controller, higher_better, source)
ARMS = {
    "image/CIFAR-100 (linear head)": dict(auth=0.424, infl=0.358, cov=0.365, rand=0.366,
                                          fixed=0.339, ctrl=0.428, hib=True,
                                          src="vision_full_3seed.log"),
    "image/CIFAR-10 (linear head)": dict(auth=0.888, infl=0.902, cov=0.921, rand=0.913,
                                         fixed=0.905, ctrl=0.919, hib=True,
                                         src="fidelity_clean_cifar.log"),
    "process/TEP (MLP)": dict(auth=0.394, infl=0.358, cov=0.303, rand=0.324,
                              fixed=0.341, ctrl=0.417, hib=True,
                              src="tep_full_3seed.log"),
    "tabular/electricity (TabPFN)": dict(auth=0.845, infl=0.816, cov=0.873, rand=0.871,
                                         fixed=0.860, ctrl=0.874, hib=True,
                                         src="tabular_full_3seed.log"),
    "time/ETTh1 (DLinear)": dict(auth=0.956, infl=1.249, cov=1.082, rand=1.066,
                                 fixed=1.018, ctrl=0.995, hib=False,
                                 src="timeseries_full_3seed.log"),
    "time/DaISy CSTR (DLinear)": dict(auth=1.026, infl=1.115, cov=1.089, rand=1.106,
                                      fixed=1.061, ctrl=1.014, hib=False,
                                      src="daisy_cstr_3seed.log"),
}


def _gain_vec(a):
    """Per-channel gain over random, oriented so higher = better, L2-normalized."""
    sgn = 1.0 if a["hib"] else -1.0
    g = np.array([sgn * (a["auth"] - a["rand"]),
                  sgn * (a["infl"] - a["rand"]),
                  sgn * (a["cov"] - a["rand"])])
    n = np.linalg.norm(g)
    return g / n if n > 1e-12 else g


def main():
    names = list(ARMS)
    G = np.stack([_gain_vec(ARMS[k]) for k in names])
    beta0 = G.mean(axis=0)
    beta0 = beta0 / (np.linalg.norm(beta0) + 1e-12)
    xs, ys = [], []
    print(f"{'arm':38} {'1-cos':>7} {'gap(norm)':>10}  source")
    for k, g in zip(names, G):
        a = ARMS[k]
        x = float(1.0 - g @ beta0)
        sgn = 1.0 if a["hib"] else -1.0
        best = max(sgn * v for v in (a["auth"], a["infl"], a["cov"], a["rand"], a["ctrl"]))
        rng_ = best - sgn * a["rand"]
        y = float(sgn * (a["ctrl"] - a["fixed"]) / (abs(rng_) + 1e-12))
        xs.append(x); ys.append(y)
        print(f"{k:38} {x:7.3f} {y:10.3f}  {a['src']}")
    r = float(np.corrcoef(xs, ys)[0, 1])
    print(f"\nPearson r = {r:.3f}  (Theorem 6 predicts positive scaling)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.scatter(xs, ys, s=42, color="#2C6FBB", zorder=3)
    for k, x, y in zip(names, xs, ys):
        ax.annotate(k.split(" (")[0], (x, y), fontsize=6.5,
                    xytext=(4, 3), textcoords="offset points")
    cf = np.polyfit(xs, ys, 1)
    xr = np.linspace(min(xs), max(xs), 20)
    ax.plot(xr, np.polyval(cf, xr), "--", color="#999999", lw=1, zorder=2)
    ax.set_xlabel(r"flip magnitude  $1-\cos\bar\theta$")
    ax.set_ylabel("measured gap (normalized)")
    ax.set_title(f"Theorem 6 scaling (r = {r:.2f})", fontsize=9)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "papers/mmdataselect/submissions/aaai2027/figures/fig_thm6_scatter.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"figure -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
