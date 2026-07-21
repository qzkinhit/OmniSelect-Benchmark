"""Aggregate the stratified multi-seed sweep into a per-(method, modality) mean +-std
table and mark the best (lowest-PPL) method per modality. Reads the per-seed result
JSONs written by run_experiment.py.

    python scripts/aggregate_stratified.py [outputs/experiment]
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np


def main() -> int:
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "experiment")
    files = sorted(f for f in os.listdir(d) if f.startswith("results_seed") and f.endswith(".json"))
    if not files:
        print(f"no results_seed*.json in {d}")
        return 1

    # ppl[method][modality] = [per-seed values]; hi[method] = [per-seed high_frac]
    ppl = defaultdict(lambda: defaultdict(list))
    hi = defaultdict(list)
    seeds = []
    for f in files:
        with open(os.path.join(d, f)) as fh:
            blob = json.load(fh)
        seeds.append(blob.get("config", {}).get("seed", f))
        for r in blob["results"]:
            for mod, v in r["ppl"].items():
                ppl[r["method"]][mod].append(v)
            hi[r["method"]].append(r.get("high_frac", 0.0))

    mods = sorted({m for meth in ppl.values() for m in meth})
    methods = list(ppl.keys())
    print(f"seeds={seeds}  ({len(files)} files)\n")
    # best (min mean ppl) per modality
    best = {}
    for mod in mods:
        means = {m: np.mean(ppl[m][mod]) for m in methods if ppl[m][mod]}
        if means:
            best[mod] = min(means, key=means.get)

    hdr = f"{'method':<14}{'hi%':>6}  " + "".join(f"{mod:>16}" for mod in mods)
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        cells = []
        for mod in mods:
            vals = ppl[m][mod]
            if not vals:
                cells.append(f"{'-':>16}")
                continue
            mean, std = np.mean(vals), np.std(vals)
            star = "*" if best.get(mod) == m else " "
            cells.append(f"{mean:8.1f}±{std:5.1f}{star}".rjust(16))
        hiv = f"{np.mean(hi[m]):.2f}" if hi[m] else "-"
        print(f"{m:<14}{hiv:>6}  " + "".join(cells))
    print("\n(* = best/lowest mean PPL per modality)")

    # --- decision-ready verdict: ours vs the best *baseline* per modality ---
    ours = "mmdataselect"
    baselines = [m for m in methods if m not in (ours, "noselect")]
    if ours in ppl:
        print(f"\n=== verdict: {ours} vs best baseline (per modality) ===")
        wins = 0
        for mod in mods:
            if not ppl[ours][mod]:
                continue
            om, osd = np.mean(ppl[ours][mod]), np.std(ppl[ours][mod])
            base_means = {m: np.mean(ppl[m][mod]) for m in baselines if ppl[m][mod]}
            if not base_means:
                continue
            bb = min(base_means, key=base_means.get)         # best baseline (lowest ppl)
            bm = base_means[bb]
            bsd = np.std(ppl[bb][mod])
            gap = 100.0 * (bm - om) / bm                       # +% = ours lower/better
            pooled = (osd + bsd) / 2 + 1e-9
            sig = abs(bm - om) > pooled                        # rough: gap exceeds avg std
            tag = "WIN " if om < bm else "lose"
            if om < bm:
                wins += 1
            sigtag = "sig" if sig else "~noise"
            print(f"  {mod:9} {tag} vs {bb:<12} ours={om:8.1f}±{osd:5.1f}  best_base={bm:8.1f}±{bsd:5.1f}  gap={gap:+5.1f}%  ({sigtag})")
        print(f"  -> wins on {wins}/{len([m for m in mods if ppl[ours][m]])} modalities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
