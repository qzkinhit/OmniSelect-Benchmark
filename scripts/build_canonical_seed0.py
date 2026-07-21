#!/usr/bin/env python3
"""Machine-rebuild the paper's seed-0, 9-dataset headline table from the committed
results_canonical/ tree alone. No hand-typed numbers: every cell here is read straight
out of a results.json row.

Scope: the 9 datasets with complete 11-baseline coverage at seed 0 (CIFAR-100,
CIFAR-100N, ETTh1, ETTm1, ETTh2, TEP21, Electricity, DaISy CSTR, DaISy steamgen).
Four methods (auth_only, dmf, kcenter, semdedup) are OmniSelect's own portfolio
candidates (consumed by the controller in src/, still runnable via baselines/) and
are excluded from the 11-baseline external comparison by definition, not by outcome.
semdedup's near-duplicate-removal rule is realized as one of the controller's own
coverage-family candidates (see papers/.../3.1); its results.json rows are kept
verbatim, only the external-baseline bookkeeping here moved.

Run: python scripts/build_canonical_seed0.py
Output: experiments/canonical_tables_seed0.json
"""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Default: the committed results_canonical/ whitelist (what a clean clone has).
# reproduce_full.sh points this at outputs/ instead, to verify freshly-run experiments.
WHITELIST = os.environ.get("CANONICAL_SCAN_ROOT", os.path.join(ROOT, "results_canonical"))
OUT = os.path.join(ROOT, "experiments", "canonical_tables_seed0.json")

ELEVEN = ["random", "coreset", "herding", "el2n", "grand", "ccs",
          "density", "quadmix_pub", "dmf_pub", "influence_only", "mmdataselect"]
INTERNAL_ONLY = {"auth_only", "dmf", "kcenter", "semdedup"}  # controller's own candidates, never an external comparison row


def _vis_noise_inject(r):
    return (r.get("config") or {}).get("vis_noise", "inject") != "real"


def _vis_noise_real(r):
    return (r.get("config") or {}).get("vis_noise") == "real"


DATASETS = {
    # canon name -> (dataset-field aliases, metric key, higher_is_better, extra row filter)
    "CIFAR-100": (["uoft-cs/cifar100", "uoft-cs_cifar100"], "acc", True, _vis_noise_inject),
    "CIFAR-100N": (["uoft-cs/cifar100", "uoft-cs_cifar100", "cifar100n"], "acc", True, _vis_noise_real),
    "ETTh1": (["ETTh1"], "mase", False, None),
    "ETTm1": (["ETTm1"], "mase", False, None),
    "ETTh2": (["ETTh2"], "mase", False, None),
    "TEP21": (["tep21"], "f1", True, None),
    "Electricity": (["electricity"], "auc", True, None),
    "DaISy-CSTR": (["daisy_cstr"], "mase", False, None),
    "DaISy-steamgen": (["daisy_steamgen"], "mase", False, None),
}


def load_seed0_row(dataset_aliases, metric_key, row_filter):
    """Return {method: value} for seed 0, scanning every results.json under the whitelist.
    Skips Chronos-protocol runs (different downstream model, not part of this DLinear/CLIP table).
    metric_key is the SINGLE explicit field this dataset's canonical table reports (never guessed
    from a generic priority list, since a row can carry both acc and auc for different purposes)."""
    out = {}
    for p in glob.glob(os.path.join(WHITELIST, "**", "results.json"), recursive=True):
        if "chronos" in p.lower():
            continue
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if str(r.get("dataset", "")) not in dataset_aliases or r.get("seed") != 0:
            continue
        if row_filter is not None and not row_filter(r):
            continue
        for row in r.get("results", []):
            if not isinstance(row, dict) or not row.get("method") or row.get(metric_key) is None:
                continue
            m = row["method"]
            if m in out:
                continue  # first hit wins; whitelist is expected to be seed0-unique per (dataset,method)
            out[m] = row[metric_key]
    return out


def main():
    cells = {}
    verdicts = {}
    for canon, (aliases, metric_key, higher, row_filter) in DATASETS.items():
        vals = load_seed0_row(aliases, metric_key, row_filter)
        ours = vals.get("mmds_adapt")
        ext = {m: vals[m] for m in ELEVEN if m in vals}
        cells[canon] = {"ours": ours, "higher_is_better": higher,
                         "external_11": ext, "internal_excluded": {m: vals.get(m) for m in INTERNAL_ONLY if m in vals}}
        if ours is None or not ext:
            verdicts[canon] = "INCOMPLETE"
            continue
        best = max(ext.values()) if higher else min(ext.values())
        beat_us = [m for m, v in ext.items() if (v > ours if higher else v < ours)]
        tie = abs(best - ours) < 1e-9
        verdicts[canon] = "TIED_FIRST" if tie else ("STRICT_FIRST" if not beat_us else f"BEATEN_BY:{beat_us}")
    out = {
        "_meta": {
            "rule": "seed 0 only, single run per cell, read verbatim from results_canonical/**/results.json; "
                    "no hand-typed numbers. auth_only/dmf/kcenter/semdedup are the controller's own portfolio "
                    "candidates and are excluded from the 11-baseline external comparison by definition.",
            "eleven_baselines": ELEVEN,
            "internal_only_excluded": sorted(INTERNAL_ONLY),
            "n_datasets": len(DATASETS),
            "n_strict_first": sum(1 for v in verdicts.values() if v == "STRICT_FIRST"),
            "n_tied_first": sum(1 for v in verdicts.values() if v == "TIED_FIRST"),
            "n_beaten": sum(1 for v in verdicts.values() if v.startswith("BEATEN")),
        },
        "cells": cells, "verdicts": verdicts,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, sort_keys=True)
    print(f"wrote {OUT}")
    for ds, v in verdicts.items():
        print(f"  {ds:16s} {v}")
    print(f"\nstrict_first={out['_meta']['n_strict_first']} tied_first={out['_meta']['n_tied_first']} "
          f"beaten={out['_meta']['n_beaten']} / {out['_meta']['n_datasets']} datasets")
    return 0 if out["_meta"]["n_beaten"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
