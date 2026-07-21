"""AdaptiveController re-validation v4.

IMPACT LEDGER: every paper-facing controller row produced under the pre-fix code is
either in PLAN (replayed) or in EXCLUDED (with a construction-level justification).

Modes:
  --dry-run   parse-only: prove old picked/val/test are extractable for EVERY planned
              trial; prints a table and exits nonzero on any gap. No GPU, no writes.
  (default)   replay phase, gated per trial:
              exit0, isolated artifact exists, old+new picked exist, old_val/old_test
              exist, math.isfinite on new val/test, sel_sha12 present, stderr clean,
              picked_match true AND |delta_test|<=tol  -> trial UNCHANGED.
              all trials gate-pass & unchanged  -> CONTROLLER_REVALIDATED_OK
              all trials gate-pass, some changed -> REPLAY_COMPLETE_WITH_CHANGES
              anything else                      -> no marker, nonzero exit.
"""
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys

R = "/root/autodl-tmp/OmniSelect"
PY = os.path.join(R, ".venv/bin/python")
LOGDIR = os.path.join(R, "experiments/controller_reval")
os.makedirs(LOGDIR, exist_ok=True)
DRY = "--dry-run" in sys.argv
TOL = 0.005
VAL_PAT = r"\(val(?:_[a-zA-Z0-9]+)?=(-?[0-9.]+)\)"

EXCLUDED = {
    "text-pilot & text-scaleup ctrl": "portfolio = baseline strategies only (no fusion grid, no gate); q>0 candidates never exist",
    "original-protocol ctrl (E)": "argmax over five fixed subsets; no fusion grid, no gate",
    "G drop=auth": "prefilter_grid=(0.0,) when auth dropped; q is identically 0",
    "SH/GRPO negative-result rows": "docs-only archived negatives, not paper tables; same config class as replayed base rows",
    "H calib ctrl rows": "identical controller config to tep-base (calib only adds metrics); covered by tep-base replay",
}

# (tag, env, script, old_log, block_regexes[fmt with {s}], metric)
PLAN = [
    ("vision-base", {"METHODS": "mmds_adapt", "VIS_DATASET": "uoft-cs/cifar100"},
     "scripts/run_vision_experiment.py", "experiments/split_protocol_3seed.log",
     [r"##### GRPO=0 SEED={s} VISION #####"], "test_acc"),
    ("tep-base", {"METHODS": "mmds_adapt", "MODEL": "mlp"},
     "scripts/run_tep_experiment.py", "experiments/split_protocol_3seed.log",
     [r"##### GRPO=0 SEED={s} TEP #####"], "f1"),
    ("tabular-base", {"METHODS": "mmds_adapt", "MODEL": "tabpfn"},
     "scripts/run_tabular_experiment.py", "experiments/split_protocol_3seed.log",
     [r"##### GRPO=0 SEED={s} TAB #####"], "auc"),
    ("ts-ETTh1", {"METHODS": "mmds_adapt", "TS_DATASET": "ETTh1"},
     "scripts/run_timeseries_experiment.py", "experiments/split_protocol_3seed.log",
     [r"##### GRPO=0 SEED={s} ETTH1 #####"], "mase"),
    ("ts-ETTh2", {"METHODS": "mmds_adapt", "TS_DATASET": "ETTh2"},
     "scripts/run_timeseries_experiment.py", "experiments/etth2_3seed.log",
     [r"##### SEED {s}[^\n]*#####"], "mase"),
    ("ts-daisy_cstr", {"METHODS": "mmds_adapt", "TS_DATASET": "daisy_cstr"},
     "scripts/run_timeseries_experiment.py", "experiments/daisy_cstr_3seed.log",
     [r"##### SEED {s}[^\n]*#####"], "mase"),
    ("ts-daisy_steamgen", {"METHODS": "mmds_adapt", "TS_DATASET": "daisy_steamgen"},
     "scripts/run_timeseries_experiment.py", "experiments/daisy_steamgen_3seed.log",
     [r"##### SEED {s}[^\n]*#####"], "mase"),
    ("chronos-ETTh1", {"METHODS": "mmds_adapt", "TS_DATASET": "ETTh1", "TS_MODEL": "chronos"},
     "scripts/run_timeseries_experiment.py", "experiments/chronos_fm_3seed.log",
     [r"##### CHRONOS ds=ETTh1 SEED={s} #####"], "mase"),
    ("chronos-ETTh2", {"METHODS": "mmds_adapt", "TS_DATASET": "ETTh2", "TS_MODEL": "chronos"},
     "scripts/run_timeseries_experiment.py", "experiments/chronos_fm_3seed.log",
     [r"##### CHRONOS ds=ETTh2 SEED={s} #####"], "mase"),
    ("chronos-daisy_cstr", {"METHODS": "mmds_adapt", "TS_DATASET": "daisy_cstr", "TS_MODEL": "chronos"},
     "scripts/run_timeseries_experiment.py", "experiments/chronos_fm_3seed.log",
     [r"##### CHRONOS ds=daisy_cstr SEED={s} #####"], "mase"),
    ("chronos-daisy_steamgen", {"METHODS": "mmds_adapt", "TS_DATASET": "daisy_steamgen", "TS_MODEL": "chronos"},
     "scripts/run_timeseries_experiment.py", "experiments/chronos_fm_3seed.log",
     [r"##### CHRONOS ds=daisy_steamgen SEED={s} #####"], "mase"),
    ("chronos-ETTm1", {"METHODS": "mmds_adapt", "TS_DATASET": "ETTm1", "TS_MODEL": "chronos"},
     "scripts/run_timeseries_experiment.py", "experiments/chronos_ettm1_3seed.log",
     [r"##### CHRONOS ds=ETTm1 SEED={s} #####"], "mase"),
    ("vision-nf0.2", {"METHODS": "mmds_adapt", "VIS_DATASET": "uoft-cs/cifar100", "NOISE_FRAC": "0.2"},
     "scripts/run_vision_experiment.py", "experiments/noise_ratio_ablation_3seed.log",
     [r"##### VIS-NOISE nf=0.2 SEED={s} #####"], "test_acc"),
    ("vision-nf0.6", {"METHODS": "mmds_adapt", "VIS_DATASET": "uoft-cs/cifar100", "NOISE_FRAC": "0.6"},
     "scripts/run_vision_experiment.py", "experiments/noise_ratio_ablation_3seed.log",
     [r"##### VIS-NOISE nf=0.6 SEED={s} #####"], "test_acc"),
    ("tep-nf0.2", {"METHODS": "mmds_adapt", "MODEL": "mlp", "NOISE_FRAC": "0.2"},
     "scripts/run_tep_experiment.py", "experiments/noise_ratio_ablation_3seed.log",
     [r"##### TEP-NOISE nf=0.2 SEED={s} #####"], "f1"),
    ("tep-nf0.6", {"METHODS": "mmds_adapt", "MODEL": "mlp", "NOISE_FRAC": "0.6"},
     "scripts/run_tep_experiment.py", "experiments/noise_ratio_ablation_3seed.log",
     [r"##### TEP-NOISE nf=0.6 SEED={s} #####"], "f1"),
    ("vision-drop-infl", {"METHODS": "mmds_adapt", "VIS_DATASET": "uoft-cs/cifar100", "DROP_CHANNEL": "infl"},
     "scripts/run_vision_experiment.py", "experiments/channel_drop_ablation_3seed.log",
     [r"##### VIS-DROP ch=infl SEED={s} #####"], "test_acc"),
    ("vision-drop-red", {"METHODS": "mmds_adapt", "VIS_DATASET": "uoft-cs/cifar100", "DROP_CHANNEL": "red"},
     "scripts/run_vision_experiment.py", "experiments/channel_drop_ablation_3seed.log",
     [r"##### VIS-DROP ch=red SEED={s} #####"], "test_acc"),
    ("tep-drop-infl", {"METHODS": "mmds_adapt", "MODEL": "mlp", "DROP_CHANNEL": "infl"},
     "scripts/run_tep_experiment.py", "experiments/channel_drop_ablation_3seed.log",
     [r"##### TEP-DROP ch=infl SEED={s} #####"], "f1"),
    ("tep-drop-red", {"METHODS": "mmds_adapt", "MODEL": "mlp", "DROP_CHANNEL": "red"},
     "scripts/run_tep_experiment.py", "experiments/channel_drop_ablation_3seed.log",
     [r"##### TEP-DROP ch=red SEED={s} #####"], "f1"),
    ("vision-realnoise", {"METHODS": "mmds_adapt", "VIS_DATASET": "uoft-cs/cifar100", "VIS_NOISE": "real"},
     "scripts/run_vision_experiment.py", "experiments/real_noise_cifar100n_3seed.log",
     [r"##### REALN SEED={s} #####"], "test_acc"),
]


def block_for(logpath, regexes, seed):
    fp = os.path.join(R, logpath)
    if not os.path.exists(fp):
        return None
    txt = open(fp, errors="ignore").read()
    for rx in regexes:
        pat = rx.replace("{s}", str(seed))
        m = re.search(pat + r"(.*?)(?=##### |\Z)", txt, re.S)
        if m:
            return m.group(1)
    return None


def old_evidence(logpath, regexes, seed, metric):
    body = block_for(logpath, regexes, seed)
    if body is None:
        return None, None, None
    picks = re.findall(r"picked '([^']+)'", body)
    vals = re.findall(VAL_PAT, body)
    pat = {"test_acc": r"mmds_adapt\s+(?:n=\s*\d+ clean%=[0-9.]+ )?(?:test_)?acc=([0-9.]+)",
           "f1": r"mmds_adapt\s+F1=([0-9.]+)",
           "auc": r"mmds_adapt\s+auc=([0-9.]+)",
           "mase": r"mmds_adapt\s+(?:n=\s*\d+ clean%=[0-9.]+ )?MASE=([0-9.]+)"}[metric]
    tests = re.findall(pat, body)
    return (picks[-1] if picks else None,
            float(vals[-1]) if vals else None,
            float(tests[-1]) if tests else None)


if DRY:
    bad = 0
    print(f"{'config':22} {'seed':>4} {'picked':>7} {'val':>7} {'test':>7}")
    for tag, env, script, oldlog, rxs, metric in PLAN:
        for seed in (0, 1, 2):
            p, v, t = old_evidence(oldlog, rxs, seed, metric)
            ok = all(x is not None for x in (p, v, t))
            print(f"{tag:22} {seed:>4} {str(p is not None):>7} {str(v is not None):>7} {str(t is not None):>7}"
                  + ("" if ok else f"   <-- GAP ({oldlog})"))
            bad += (not ok)
    print(f"\nEXCLUDED ({len(EXCLUDED)}):")
    for k, why in EXCLUDED.items():
        print(f"  - {k}: {why}")
    print(f"\nDRY_RUN_{'OK' if bad == 0 else f'GAPS={bad}'}")
    sys.exit(0 if bad == 0 else 1)

report = {"replays": [], "changed": [], "excluded": EXCLUDED}
for tag, env, script, oldlog, rxs, metric in PLAN:
    for seed in (0, 1, 2):
        e = dict(os.environ)
        e.update(env)
        e["SEED"] = str(seed)
        e["RUN_ID"] = "controller-reval"
        outp = os.path.join(LOGDIR, f"{tag}_seed{seed}.out.log")
        errp = os.path.join(LOGDIR, f"{tag}_seed{seed}.err.log")
        with open(outp, "w") as fo, open(errp, "w") as fe:
            r = subprocess.run([PY, "-u", os.path.join(R, script)], env=e, cwd=R,
                               stdout=fo, stderr=fe, timeout=7200)
        out = open(outp, errors="ignore").read()
        picks = re.findall(r"picked '([^']+)'", out)
        new_pick = picks[-1] if picks else None
        g = glob.glob(os.path.join(R, "outputs", "*", "*", "run_id=controller-reval-*", f"seed_{seed}", "results.json"))
        # 找与本配置匹配的最新产物
        jp, newval, newtest, selh = "", None, None, None
        cand = sorted(g, key=os.path.getmtime, reverse=True)
        if cand:
            jp = cand[0]
            d = json.load(open(jp))
            man = d.get("adapt_manifest") or {}
            selh = man.get("sel_sha12")
            newval = (man.get("chosen") or {}).get("val_gain")
            for row in d.get("results", []):
                if row.get("method") == "mmds_adapt":
                    newtest = row.get(metric) or row.get("acc") or row.get("mase") or row.get("f1") or row.get("auc")
        oldp, oldv, oldt = old_evidence(oldlog, rxs, seed, metric)
        errtxt = open(errp, errors="ignore").read()
        picked_match = (new_pick == oldp) if (new_pick and oldp) else None
        dtest = abs(newtest - oldt) if (isinstance(newtest, float) and isinstance(oldt, float)) else None
        gate = {
            "exit0": r.returncode == 0,
            "artifact": bool(jp),
            "old_picked": oldp is not None,
            "new_picked": new_pick is not None,
            "old_val": oldv is not None,
            "old_test": oldt is not None,
            "new_val_finite": isinstance(newval, (int, float)) and math.isfinite(newval),
            "new_test_finite": isinstance(newtest, (int, float)) and math.isfinite(newtest),
            "sel_hash": bool(selh),
            "stderr_clean": not re.search(r"Traceback|CUDA out of memory|Killed", errtxt),
        }
        unchanged = bool(picked_match) and dtest is not None and dtest <= TOL
        entry = {"config": tag, "seed": seed, "gate": gate, "gate_pass": all(gate.values()),
                 "old_picked": oldp, "new_picked": new_pick, "picked_match": picked_match,
                 "old_val": oldv, "new_val": newval, "old_test": oldt, "new_test": newtest,
                 "delta_test": dtest, "unchanged": unchanged, "sel_sha12": selh,
                 "stdout": outp, "stderr": errp, "artifact": jp}
        report["replays"].append(entry)
        if all(gate.values()) and not unchanged:
            report["changed"].append(entry)
        print(f"[replay] {tag} seed{seed} gate={all(gate.values())} unchanged={unchanged} "
              f"old={oldp!r} new={new_pick!r} dtest={dtest}")

n = len(PLAN) * 3
gates_ok = sum(1 for x in report["replays"] if x["gate_pass"])
unchanged_n = sum(1 for x in report["replays"] if x["unchanged"])
report["summary"] = {"planned": n, "ran": len(report["replays"]), "gate_pass": gates_ok,
                     "unchanged": unchanged_n, "changed": len(report["changed"])}
json.dump(report, open(os.path.join(R, "experiments/controller_reval_report.json"), "w"), indent=2)
print(json.dumps(report["summary"]))
if gates_ok == n == len(report["replays"]):
    if unchanged_n == n:
        open("/root/CONTROLLER_REVALIDATED_OK", "w").write(json.dumps(report["summary"]))
        print("CONTROLLER_REVALIDATED_OK")
        sys.exit(0)
    open("/root/REPLAY_COMPLETE_WITH_CHANGES", "w").write(json.dumps(report["summary"]))
    print("REPLAY_COMPLETE_WITH_CHANGES (targeted adoption required before final OK)")
    sys.exit(2)
print("NO MARKER (gates incomplete)")
sys.exit(1)
