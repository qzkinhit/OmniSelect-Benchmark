"""Appendix table of paired verdicts and election on each main-table task.

Per task: the baselines that OmniSelect beats (W) or ties (T) in the paired test (it loses to none), the baselines
whose mean falls below Random (Below), the number m of frozen candidates of a run (Random included), whether a
challenger is adopted in all, some, or none of the runs, the kinds of the elected strategies (Fu fusion, Co cooperative, Cs consensus, Li library,
Ex external), and the gain over Random of the run means of the best reference, of the elected strategy, and of the
best frozen candidate of each run in hindsight. The gain definition is the one of the main table.

Usage: python3 adjudication_by_task.py <directory holding main/> <paper_main_stats.json> <paper_numbers.json> <output json>
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

REC, STATS, NUMBERS, OUT = map(Path, sys.argv[1:5])
MAP = {"C-100": "vision/cifar100/clip_vitb32", "C-100N": "vision/cifar100n/clip_vitb32",
       "C-10": "native/cifar10/resnet18_scratch", "ETTh1": "timeseries/ETTh1/dlinear", "ETTm1": "timeseries/ETTm1/dlinear",
       "ETTh2": "timeseries/ETTh2/dlinear", "CSTR": "timeseries/daisy_cstr/dlinear",
       "Steam": "timeseries/daisy_steamgen/dlinear", "TEP21": "process/tep21/mlp", "Elec": "tabular/electricity/tabpfn",
       "Text": "text/five_domain/smollm2_135m"}
EXTERNAL = {"ccs", "herding", "herding_text", "density", "quadmix_pub", "dmf_pub", "el2n", "grand"}


def kind(name):
    if name.startswith("fuse") or name == "mmdataselect":
        return "Fu"
    if name.startswith("coop"):
        return "Co"
    if name.startswith("vote_ensemble"):
        return "Cs"
    if name in EXTERNAL:
        return "Ex"
    return "Li"


def main():
    stats = json.loads(STATS.read_text())["tasks"]
    below = json.loads(NUMBERS.read_text())["displayed_below_random"]
    out = {}
    for t, rel in MAP.items():
        ref, el, best, rnd, m, adopted, kinds = [], [], [], [], set(), [], []
        for d in sorted(glob.glob(str(REC / "main" / rel / "seed_*"))):
            dec = json.loads(Path(d, "decision.json").read_text())
            rows = {r["name"]: r for r in json.loads(Path(d, "leaderboard.json").read_text())["rows"]}
            u = {n: r.get("u_test") for n, r in rows.items()}
            e = dec.get("elected_test_utility")
            e = u[dec["elected"]] if e is None else e
            cand = [r["u_test"] for n, r in rows.items()
                    if r.get("u_test") is not None and r["role"] != "screened_out" and n not in ("random", "full")]
            ref.append(u[dec["reference"]]); el.append(e); best.append(max(cand)); rnd.append(u["random"])
            m.add(len(cand) + 1); adopted.append(bool(dec["adopted"])); kinds.append(kind(dec["elected"]))
        z = np.mean(rnd)
        gain = lambda v: float(100 * (np.mean(v) - z) / abs(z))   # utilities are stored larger-is-better
        verdicts = [x["verdict"] for x in stats[t]["verdicts"].values()]
        order = sorted(set(kinds), key=lambda k: (-kinds.count(k), kinds.index(k)))
        assert verdicts.count("L") == 0
        out[t] = {"W": verdicts.count("W"), "T": verdicts.count("T"),
                  "below": sum(t in v for v in below.values()), "m": min(m) if len(m) == 1 else sorted(m),
                  "adoption": "all" if all(adopted) else "none" if not any(adopted) else "some",
                  "kinds": order, "gain_ref": gain(ref), "gain_elected": gain(el), "gain_best": gain(best)}
    mean = {k: float(np.mean([o[k] for o in out.values()])) for k in ("gain_ref", "gain_elected", "gain_best")}
    res = {"tasks": out, "mean": mean, **{k: sum(o[k] for o in out.values()) for k in ("W", "T", "below")}}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(mean), "W", res["W"], "T", res["T"], "below", res["below"])
    for t, o in out.items():
        print(t, o["W"], o["T"], o["below"], o["m"], o["adoption"], ",".join(o["kinds"]),
              f'{o["gain_ref"]:.1f} {o["gain_elected"]:.1f} {o["gain_best"]:.1f}')


if __name__ == "__main__":
    main()
