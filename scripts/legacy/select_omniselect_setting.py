#!/usr/bin/env python3
"""SUPERSEDED - the two-switch-setting (regular/calibrated) design this script
implements has been abandoned in favor of a single controller. Kept for audit-trail
provenance only; not called by any current runner or reproduce script. See
scripts/build_canonical_seed0.py for the current headline-table pipeline.

--- original docstring below ---

Unified OmniSelect result per (dataset, seed): pick the better of the two switch
settings by the VALIDATION main metric only, then report the chosen setting's test
result. Never reads a test field to choose. Produces a machine-readable manifest so
the canonical table is rebuilt mechanically, not by hand-copying the larger value.

Two settings (same signals, candidates, budget, downstream training; differ only in
the final switch condition):
  - regular    : fixed-fraction regularized switch (higher selection sensitivity when
                 validation evidence is ample)
  - calibrated : task-specific finite-sample statistical radius (keeps the reference
                 unless the paired advantage exceeds the statistical fluctuation)

Selection rule per seed:
  read chosen.val_gain from both settings (a validation-only held-out gain on the same
  split); adopt the higher; on a tie or when indistinguishable, default to calibrated.
  Then report that setting's test metric as the OmniSelect value for the seed.

Inputs are two result trees keyed identically by (dataset, seed). Outputs
experiments/omniselect_setting_selection.json.
"""
import argparse
import glob
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIE_EPS = 1e-9


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load(path):
    r = json.load(open(path))
    ch = (r.get("adapt_manifest") or {}).get("chosen") or {}
    row = next((x for x in r["results"]
                if isinstance(x, dict) and x.get("method") == "mmds_adapt"), {})
    pm = r.get("pairing_manifest") or {}
    metric = next((k for k in ("acc", "auc", "f1", "macro_f1", "mase")
                   if k in row and row[k] is not None), None)
    return {"path": path, "val_gain": ch.get("val_gain"),
            "test_metric_name": metric, "test_metric": row.get(metric) if metric else None,
            "sel_sha12": row.get("sel_sha12"), "pool_sha256": pm.get("pool_sha256"),
            "results_sha256": sha256(path)}


def pick(regular, calibrated):
    """Validation-only choice; default calibrated on tie/missing."""
    gr, gc = regular.get("val_gain"), calibrated.get("val_gain")
    if gr is None and gc is None:
        return "calibrated", "no validation gain on either; default calibrated"
    if gr is None:
        return "calibrated", "regular missing validation gain"
    if gc is None:
        return "regular", "calibrated missing validation gain"
    if abs(gr - gc) <= TIE_EPS:
        return "calibrated", f"validation tie ({gr:.6g}=={gc:.6g}); default calibrated"
    if gc > gr:
        return "calibrated", f"calibrated validation gain higher ({gc:.6g} > {gr:.6g})"
    return "regular", f"regular validation gain higher ({gr:.6g} > {gc:.6g})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regular-glob", required=True,
                    help="glob for the fixed-fraction setting results.json (with {dataset}/{seed} structure)")
    ap.add_argument("--calibrated-glob", required=True,
                    help="glob for the calibrated setting results.json")
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "omniselect_setting_selection.json"))
    args = ap.parse_args()

    def index(pattern):
        d = {}
        for p in glob.glob(pattern, recursive=True):
            # key by (dataset dir, seed dir) — the two segments that identify a cell
            parts = p.split(os.sep)
            seed = next((s for s in parts if s.startswith("seed_")), None)
            dataset = parts[parts.index(seed) - 2] if seed and parts.index(seed) >= 2 else parts[-3]
            d[(dataset, seed)] = p
        return d

    reg, cal = index(args.regular_glob), index(args.calibrated_glob)
    keys = sorted(set(reg) & set(cal))
    out = {"_meta": {"rule": "per seed: adopt the setting with the higher validation held-out "
                             "gain (val_gain); test field never read for selection; tie/default = calibrated",
                     "settings": ["regular (fixed-fraction switch)", "calibrated (statistical radius)"],
                     "n_paired_cells": len(keys)},
           "cells": {}, "unpaired": {"regular_only": sorted(set(reg) - set(cal)),
                                     "calibrated_only": sorted(set(cal) - set(reg))}}
    for k in keys:
        r, c = load(reg[k]), load(cal[k])
        paired = r["pool_sha256"] == c["pool_sha256"] and r["pool_sha256"] is not None
        setting, reason = pick(r, c)
        chosen = c if setting == "calibrated" else r
        out["cells"]["/".join(k)] = {
            "dataset": k[0], "seed": k[1], "same_pool_verified": paired,
            "regular": {"val_gain": r["val_gain"], "test": r["test_metric"],
                        "sel_sha12": r["sel_sha12"], "results_sha256": r["results_sha256"]},
            "calibrated": {"val_gain": c["val_gain"], "test": c["test_metric"],
                           "sel_sha12": c["sel_sha12"], "results_sha256": c["results_sha256"]},
            "chosen_setting": setting, "reason": reason,
            "metric_name": chosen["test_metric_name"],
            "omniselect_test": chosen["test_metric"]}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"paired cells: {len(keys)}  ->  {args.out}")
    for kk, v in out["cells"].items():
        flag = "" if v["same_pool_verified"] else "  [WARN pool mismatch]"
        print(f"  {kk}: {v['chosen_setting']:10s} omniselect={v['omniselect_test']} ({v['metric_name']}){flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
