"""Strict validator for the TEP calibrated 3-seed lane.

Checks (all must pass, exit nonzero otherwise; marker is written only by the lane wrapper
after this validator exits 0):
  1. results JSON exists for every (method, seed) under outputs/tep/*/tep-calib*/seed_*/;
  2. every method entry carries calib with the primary operating point, empirical
     val_far@far5 <= 0.05, thresholds, balacc, auroc, auprc, confusion@far5;
  3. log error scan: no Traceback / CUDA error / NaN-metric lines;
  4. exit codes: every seed has "TEPCALIB SEED=s python_exit=0" in the log.
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "experiments",
                   os.environ.get("TEP_CALIB_LOG", "tep_calibrated_3seed.log"))
RUN_GLOB = os.environ.get("TEP_CALIB_RUNGLOB", "*tep-calib*")
SEED_PREFIX = os.environ.get("TEP_CALIB_PREFIX", "TEPCALIB")
EXPECTED_METHODS = os.environ.get(
    "TEP_CALIB_METHODS",
    "full,random,coreset,auth_only,influence_only,mmdataselect,mmds_adapt,herding,"
    "kcenter,el2n,grand,ccs,semdedup,density,quadmix,dmf,d4,dsdm").split(",")
SEEDS = ("0", "1", "2")
CAP5 = 0.05

fails = []

# 1+2: per-seed JSONs and calib fields
for s in SEEDS:
    paths = glob.glob(os.path.join(ROOT, "outputs", "tep", "*", RUN_GLOB, f"seed_{s}", "results.json"))
    if not paths:
        fails.append(f"seed {s}: no tep-calib results.json")
        continue
    rows = json.load(open(sorted(paths)[-1]))
    if isinstance(rows, dict):  # _trial_dump wrapper: metadata + "results" list
        rows = rows["results"]
    have = {r["method"]: r for r in rows}
    for m in EXPECTED_METHODS:
        r = have.get(m)
        if r is None:
            fails.append(f"seed {s}: method {m} missing from results")
            continue
        c = r.get("calib") or {}
        if c.get("na_reason"):
            # documented structural N/A (e.g. no class-0 in selected training set) is a
            # valid cell; an EMPTY calib without na_reason is not.
            continue
        need = ("primary_operating_point", "threshold@far5", "val_far@far5", "val_fp@far5",
                "fdr@far5", "far@far5", "balacc@far5", "auroc", "auprc", "confusion@far5")
        miss = [k for k in need if k not in c]
        if miss:
            fails.append(f"seed {s} {m}: calib missing {miss}")
            continue
        if c["val_far@far5"] > CAP5 + 1e-9:
            fails.append(f"seed {s} {m}: val_far@far5={c['val_far@far5']} exceeds cap {CAP5}")
        for k in ("f1", "acc"):
            v = r.get(k)
            if v is None or v != v:
                fails.append(f"seed {s} {m}: metric {k} is missing/NaN")

# 3: log error scan
if os.path.exists(LOG):
    bad = re.compile(r"Traceback|CUDA error|RuntimeError|nan(?![a-z])", re.IGNORECASE)
    for i, line in enumerate(open(LOG, errors="replace"), 1):
        if bad.search(line):
            fails.append(f"log error line {i}: {line.strip()[:120]}")
else:
    fails.append(f"log missing: {LOG}")

# 4: exit codes
if os.path.exists(LOG):
    txt = open(LOG, errors="replace").read()
    for s in SEEDS:
        if not re.search(rf"{SEED_PREFIX} SEED={s} python_exit=0", txt):
            fails.append(f"seed {s}: python_exit=0 marker not found in log")

if fails:
    print("TEP_CALIB VALIDATION FAILED (%d):" % len(fails))
    for f in fails[:40]:
        print("  -", f)
    sys.exit(1)
print("TEP_CALIB VALIDATION OK: %d methods x %d seeds, val FAR cap enforced, no log errors"
      % (len(EXPECTED_METHODS), len(SEEDS)))
