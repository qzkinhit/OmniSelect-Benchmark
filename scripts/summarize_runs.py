#!/usr/bin/env python3
"""汇总 experiments/*_3seed.log 为各模态 mean±std 表(runall.sh local 跑完自动调用)。"""
import os
import re
import sys
from collections import defaultdict

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOGS = [
    ("vision_full_3seed.log", r'^\s{2}([a-z_0-9]+)\s+acc=([0-9.]+)', "VISION CIFAR-100 top-1 (higher better)"),
    ("tep_full_3seed.log", r'^\s{2}([a-z_0-9]+)\s+.*?F1=([0-9.]+)', "TEP macro-F1 (higher better)"),
    ("tabular_full_3seed.log", r'^\s{2}([a-z_0-9]+)\s+auc=([0-9.]+)', "TABULAR electricity ROC AUC (higher better)"),
    ("timeseries_full_3seed.log", r'^\s{2}([a-z_0-9]+)\s+MASE=([0-9.]+)', "TIMESERIES ETTh1 MASE (lower better)"),
    ("daisy_cstr_3seed.log", r'^\s{2}([a-z_0-9]+)\s+MASE=([0-9.]+)', "DAISY CSTR MASE (lower better)"),
]


def main():
    for fname, pat, title in LOGS:
        path = os.path.join(_REPO, "experiments", fname)
        if not os.path.exists(path):
            continue
        vals = defaultdict(list)
        for m in re.finditer(pat, open(path).read(), re.M):
            vals[m.group(1)].append(float(m.group(2)))
        if not vals:
            continue
        print(f"\n==== {title} ====")
        for name in sorted(vals, key=lambda x: -np.mean(vals[x])):
            v = vals[name]
            print(f"  {name:16} {np.mean(v):.4f} ± {np.std(v):.4f}  (n={len(v)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
