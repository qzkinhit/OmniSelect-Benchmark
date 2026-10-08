"""Rebuild both scale-up columns from the three published runs of each batch."""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path.cwd()
SEEDS = json.loads((Path(__file__).resolve().parents[1] / "seeds_reported.json").read_text())["scale_up_seeds"]
PATHS = {
    "IN-100": {"main": "scale_up/imagenet224/native/imagenet100/resnet18_scratch",
               "nocoop": "scale_up/imagenet224_nc/native/imagenet100/resnet18_scratch"},
    "Text-100k": {"main": "scale_up/text100k_adult_v2/text/five_domain/smollm2_360m",
                  "nocoop": "scale_up/text100k_adult_v2_nc/text/five_domain/smollm2_360m",
                  "adapt": "scale_up/text100k_adapt_v2/text/five_domain/smollm2_360m"},
}
ALIASES = {"coverage_text": "coreset", "fixed_fusion": "mmdataselect", "herding_text": "herding", "density_text": "density",
           "el2n_adapt": "el2n", "grand_adapt": "grand", "ccs_adapt": "ccs"}
DISPLAYED = ["random", "influence_only", "coreset", "mmdataselect", "auth_only", "herding", "el2n", "grand", "ccs", "density", "quadmix_pub", "dmf_pub"]


def read_cell(rel, seed, lower):
    path = ROOT / rel / f"seed_{seed}"
    met = json.loads((path / "metrics.json").read_text())
    raw_rows = met["rows"]
    if isinstance(raw_rows, list):
        raw_rows = {r.get("name", r.get("strategy")): r for r in raw_rows}
    sign = -1 if lower else 1
    rows = {ALIASES.get(k, k): sign * v["u_test"] for k, v in raw_rows.items() if v.get("u_test") is not None}
    lb = json.loads((path / "leaderboard.json").read_text())
    lb = lb if isinstance(lb, list) else lb["rows"]
    for r in lb:
        key = ALIASES.get(r["name"], r["name"])
        if key in DISPLAYED and key not in rows and r.get("u_test") is not None:
            rows[key] = sign * r["u_test"]
    dec = json.loads((path / "decision.json").read_text())
    elected = next((r for r in lb if r["name"] == dec["elected"]), None)
    value = dec.get("elected_test_utility")
    if value is None and elected is not None:
        value = elected["u_test"]
    if value is not None:
        rows["mmds_adapt"] = sign * value
    return rows, dec["elected"]


def summarize(per):
    values = list(per.values())
    assert len(values) == 3
    return {"n": len(values), "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)), "per_seed": per}


def paired(main, baseline, lower):
    sign = -1 if lower else 1
    d = [sign * (main[s] - baseline[s]) for s in main]
    m = float(np.mean(d)); half = float(stats.t.ppf(.975, len(d) - 1) * np.std(d, ddof=1) / math.sqrt(len(d)))
    return {"n": len(d), "mean": m, "lo": m - half, "hi": m + half,
            "verdict": "W" if m - half > 0 else "L" if m + half < 0 else "T"}


out = {"main": {}, "nocoop": {}, "summary": {}}
for task, paths in PATHS.items():
    lower = task == "Text-100k"
    cells = {kind: {str(s): read_cell(rel, s, lower) for s in SEEDS[task]} for kind, rel in paths.items()}
    for kind in ("main", "nocoop"):
        values = {}
        for seed, (rows, _) in cells[kind].items():
            for method, value in rows.items():
                if method in DISPLAYED + ["mmds_adapt", "full"]:
                    values.setdefault(method, {})[seed] = value
        if task == "Text-100k" and kind == "main":
            for method in ("el2n", "grand", "ccs"):
                values[method] = {seed: r[0][method] for seed, r in cells["adapt"].items()}
        result = {"n_seeds": len(SEEDS[task]), "seeds": SEEDS[task], "source_batch": paths[kind],
                  "utility": "neg_gmean_ppl" if lower else "accuracy", "rows": {k: summarize(v) for k, v in values.items()},
                  "elected": {s: r[1] for s, r in cells[kind].items()}}
        if kind == "main":
            result["verdicts"] = {k: paired(values["mmds_adapt"], values[k], lower) for k in DISPLAYED}
            if lower:
                for method in ("el2n", "grand", "ccs"):
                    result["rows"][method]["source_batch"] = paths["adapt"]
        out[kind][task] = result
    rows = out["main"][task]["rows"]
    sign = -1 if lower else 1
    best = max(DISPLAYED[5:], key=lambda k: sign * rows[k]["mean"])
    om, rnd = rows["mmds_adapt"], rows["random"]
    out["summary"][task] = {"gain_over_random_pct": sign * (om["mean"] - rnd["mean"]) / abs(rnd["mean"]) * 100,
                            "runs_above_random": sum(sign * (om["per_seed"][s] - rnd["per_seed"][s]) > 0 for s in om["per_seed"]),
                            "strongest_external": best,
                            "runs_below_strongest_external": sum(sign * (om["per_seed"][s] - rows[best]["per_seed"][s]) < 0 for s in om["per_seed"]),
                            "vs_full": paired(om["per_seed"], rows["full"]["per_seed"], lower)}
Path("scale_up_stats.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
print(json.dumps(out["summary"], indent=1))
