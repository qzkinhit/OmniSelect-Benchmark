"""Build the main-table view from the reported runs of the main batches.

Every task reports the multi-seed runs of results/paper/seeds_reported.json. The EL2N, GraNd and CCS task adaptations on forecasting come from
main_adapt (standalone runs of the same code, Random identical to main on every seed) and on
Text from the independent adaptation runs.
Input: a directory with the unpacked record archives (argument 1). Outputs: paper_main_stats.json,
paper_nocoop_stats.json, paper_seeds.json.
"""
from pathlib import Path
import json, math, shutil, subprocess, sys
import numpy as np
from scipy import stats

ROOT = Path.cwd()
# Unpacked run records (the paper's published record archives, unpacked in the working directory).
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
VIEW = ROOT / "view"
COLS = ["C-100", "C-100N", "C-10", "ETTh1", "ETTm1", "ETTh2", "CSTR", "Steam", "TEP21", "Elec", "Text"]
DISPLAYED = ["random", "influence_only", "coreset", "mmdataselect", "auth_only", "herding", "el2n", "grand", "ccs", "density", "quadmix_pub", "dmf_pub"]
SIGNAL = ["random", "influence_only", "coreset", "mmdataselect", "auth_only"]           # strategies of our own library
EXTERNAL = ["herding", "el2n", "grand", "ccs", "density", "quadmix_pub", "dmf_pub"]      # adaptations of published methods
PATHS = {"C-100": "vision/cifar100/clip_vitb32", "C-100N": "vision/cifar100n/clip_vitb32", "C-10": "native/cifar10/resnet18_scratch",
         "ETTh1": "timeseries/ETTh1/dlinear", "ETTm1": "timeseries/ETTm1/dlinear", "ETTh2": "timeseries/ETTh2/dlinear",
         "CSTR": "timeseries/daisy_cstr/dlinear", "Steam": "timeseries/daisy_steamgen/dlinear", "TEP21": "process/tep21/mlp",
         "Elec": "tabular/electricity/tabpfn", "Text": "text/five_domain/smollm2_135m", "C-10-CLIP": "vision/cifar10_clip/clip_vitb32"}
sel = json.loads((Path(__file__).resolve().parents[1] / "seeds_reported.json").read_text())
SEEDS = {t: sel["seeds"][t] for t in PATHS}
LOWER_U = {"neg_mase", "neg_gmean_ppl"}


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def build_view():
    if VIEW.exists():
        shutil.rmtree(VIEW)
    VIEW.mkdir(parents=True)
    for run in ("main_nc", "main"):
        for task, rel in PATHS.items():
            for s in SEEDS[task]:
                src = SRC / run / rel / f"seed_{s}"
                assert (src / "decision.json").exists(), (run, task, s)
                dst = VIEW / run / rel / f"seed_{s}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(src.resolve())


def paired(task, key, seeds):
    sign = -1 if task["utility"] in LOWER_U else 1
    d = [sign * (task["rows"]["mmds_adapt"]["per_seed"][str(s)] - task["rows"][key]["per_seed"][str(s)]) for s in seeds]
    m = float(np.mean(d)); h = float(stats.t.ppf(.975, len(d) - 1) * np.std(d, ddof=1) / math.sqrt(len(d)))
    return {"n": len(d), "mean": m, "lo": m - h, "hi": m + h, "verdict": "W" if m - h > 0 else "L" if m + h < 0 else "T",
            "comparison": "matched seeds; no correction within task"}


def adapt_rows(label):
    """EL2N/GraNd/CCS adaptation rows of a forecasting or text task from the main_adapt batch."""
    rel = PATHS[label]; out = {}; rnd = {}
    for s in SEEDS[label]:
        met = json.loads((SRC / "main_adapt" / rel / f"seed_{s}" / "metrics.json").read_text())
        sign = -1 if met["utility"] in LOWER_U else 1
        for name, row in met["rows"].items():
            key = "random" if name == "random" else name.removesuffix("_adapt")
            (rnd if key == "random" else out.setdefault(key, {}))[str(s)] = sign * row["u_test"]
    rows = {}
    for key, per in out.items():
        vals = [per[str(s)] for s in SEEDS[label]]
        rows[key] = {"n": len(vals), "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)), "per_seed": per,
                     "adaptation": True, "source_batch": "main_adapt", "included_in_original_portfolio": False,
                     "paired_inference_vs_original_omni_eligible": label != "Text"}
    return rows, rnd


def main():
    build_view()
    docs = {}
    for run, output in [("main", "paper_main_stats.json"), ("main_nc", "paper_nocoop_stats.json")]:
        raw = VIEW / f"{run}_stats.json"
        subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "analyze_r1.py"), str(VIEW / run), str(raw)], check=True, capture_output=True)
        data = json.loads(raw.read_text())
        data["source_dir"] = str((VIEW / run).relative_to(ROOT))
        data["reporting_policy"] = {"seeds": {t: SEEDS[t] for t in COLS},
            "rule": "multi-seed runs per task, listed in results/paper/seeds_reported.json; the validation-size panel lists its runs per level there as well",
            "statistics": "arithmetic mean and sample SD (ddof=1)"}
        for label in COLS:
            task = data["tasks"][label]
            assert task["seeds"] == SEEDS[label], (run, label, task["seeds"])
        docs[output] = data
    main_data = docs["paper_main_stats.json"]
    for label in ["ETTh1", "ETTm1", "ETTh2", "CSTR", "Steam", "Text"]:
        task = main_data["tasks"][label]
        rows, rnd = adapt_rows(label)
        if label != "Text":
            for s in SEEDS[label]:
                assert math.isclose(rnd[str(s)], task["rows"]["random"]["per_seed"][str(s)], rel_tol=0, abs_tol=1e-12), (label, s)
        for key, row in rows.items():
            assert key not in task["rows"], (label, key)
            task["rows"][key] = row
            task["verdicts"][key] = paired(task, key, SEEDS[label]) if label != "Text" else {
                "n": 3, "mean": None, "lo": None, "hi": None, "verdict": "N/A", "reason": "independent text batch"}
            if label == "Text":
                row["comparison_random"] = {"per_seed": rnd, "mean": float(np.mean(list(rnd.values())))}
    for label in COLS:
        task = main_data["tasks"][label]
        sign = -1 if task["utility"] in LOWER_U else 1
        rows = task["rows"]
        task["original_portfolio_strongest_displayed"] = task["strongest_displayed"]
        task["strongest_displayed"] = max(DISPLAYED, key=lambda k: sign * rows[k]["mean"])
        task["strongest_external"] = max(EXTERNAL, key=lambda k: sign * rows[k]["mean"])
        task["strongest_signal"] = max(SIGNAL, key=lambda k: sign * rows[k]["mean"])
        rm = rows["random"]["mean"]
        task["gain_over_random"] = {k: sign * (r["mean"] - rm) / abs(rm) for k, r in rows.items()}
        task["displayed_below_random"] = [k for k in DISPLAYED if k != "random" and task["gain_over_random"][k] < 0]
    for output, data in docs.items():
        save(ROOT / output, data)
    save(ROOT / "paper_seeds.json", {"rule": main_data["reporting_policy"]})
    for label in COLS:
        t = main_data["tasks"][label]; sd = t["strongest_displayed"]; v = t["verdicts"][sd]
        ex = t["strongest_external"]; sg = t["strongest_signal"]
        print(f"{label:7s} n={t['n_seeds']:2d} omni={t['rows']['mmds_adapt']['mean']:.4f} ext={ex:11s} {t['rows'][ex]['mean']:.4f} {t['verdicts'][ex]['verdict']} sig={sg:14s} {t['rows'][sg]['mean']:.4f} {t['verdicts'][sg]['verdict']} below_rand={t['below_random_seeds']}")


if __name__ == "__main__":
    main()
