"""Runtime ledger (read-only): aggregate per-method wall-clock seconds from the
tracked run logs into experiments/runtime_ledger.json.

Every runner prints one row per method per seed ending in "(X.Xs)". Absolute
seconds are hardware-specific (logs span server generations), so the honest
statistics are WITHIN-LOG: per-method mean/std seconds and the controller's
overhead ratio versus the median standalone baseline of the same log. No new
runs, no result files touched.
"""
import glob
import json
import os
import re
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "runtime_ledger.json")
ROW = re.compile(r"^\s{2}(\S+)\s+n=\s*\d+\s.*\((\d+(?:\.\d+)?)s\)\s*$")

ledger = {"_meta": {"source": "tracked experiments/*.log per-method wall-clock rows",
                    "discipline": "absolute seconds are hardware-specific; only within-log "
                                  "method-vs-method comparisons are meaningful; no reruns",
                    "controller_key": "mmds_adapt"},
          "logs": {}}

for path in sorted(glob.glob(os.path.join(ROOT, "experiments", "*.log"))):
    rows = {}
    for line in open(path, errors="replace"):
        m = ROW.match(line)
        if m:
            rows.setdefault(m.group(1), []).append(float(m.group(2)))
    if not rows:
        continue
    entry = {"methods": {}}
    for meth, secs in sorted(rows.items()):
        entry["methods"][meth] = {
            "n_rows": len(secs),
            "mean_s": round(sum(secs) / len(secs), 2),
            "std_s": round(st.stdev(secs), 2) if len(secs) > 1 else None,
            "min_s": min(secs), "max_s": max(secs)}
    base = [v["mean_s"] for k, v in entry["methods"].items()
            if k not in ("mmds_adapt", "full", "noselect") and v["mean_s"] > 0]
    ctrl = entry["methods"].get("mmds_adapt")
    if ctrl and base:
        med = st.median(base)
        entry["controller_overhead"] = {
            "controller_mean_s": ctrl["mean_s"],
            "median_standalone_baseline_mean_s": round(med, 2),
            "ratio": round(ctrl["mean_s"] / med, 1) if med else None}
    ledger["logs"][os.path.basename(path)] = entry

with open(OUT, "w") as fh:
    json.dump(ledger, fh, indent=1)

print("logs aggregated:", len(ledger["logs"]))
for name, e in ledger["logs"].items():
    ov = e.get("controller_overhead")
    if ov:
        print(f"{name}: controller {ov['controller_mean_s']}s vs median baseline "
              f"{ov['median_standalone_baseline_mean_s']}s -> x{ov['ratio']}")
