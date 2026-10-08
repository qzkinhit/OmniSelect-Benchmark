"""Paired-interval diagnostics of Figure 3, computed from the compact ranking-unit files.

Same formulas, pairs, seeds and resampling streams as the figure script of the paper; the only change
is the reader. Argument 1: the directory that holds the unpacked paired_units tree. Argument 2: output json.
"""
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("paired_units")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("paired_diagnostics.json")
DELTA, RESAMPLES, FRAC = 0.05, 2000, 0.5

SEEDS = {task: [0, 1, 2] for task in ("C-100", "C-100N", "C-10-CLIP", "C-10", "IN-100", "Elec")}
MAIN = [
    ("C-100", "vision", "main/vision/cifar100/clip_vitb32", "top1"),
    ("C-100N", "vision", "main/vision/cifar100n/clip_vitb32", "top1"),
    ("C-10-CLIP", "vision", "main/vision/cifar10_clip/clip_vitb32", "top1"),
    ("C-10", "vision", "main/native/cifar10/resnet18_scratch", "top1"),
    ("IN-100", "vision", "scale_up/imagenet224/native/imagenet100/resnet18_scratch", "top1"),
    ("Elec", "tabular", "main/tabular/electricity/tabpfn", "auc"),
]
CONDITIONS = [
    *[(f"C-100 p={r}", "vision", f"robustness/ratio_{r}/vision/cifar100/clip_vitb32", "top1", "C-100")
      for r in ("0.0", "0.1", "0.2", "0.3", "0.5", "0.6")],
    ("C-100 unseen-corruption", "vision", "robustness/unseen/vision/cifar100/clip_vitb32", "top1", "C-100"),
    ("C-100 learner-dinov2", "vision", "robustness/learner/vision/cifar100/dinov2_vits14", "top1", "C-100"),
    ("C-100 val-natural", "vision", "robustness/val_natural/vision/cifar100/clip_vitb32", "top1", "C-100"),
    ("C-100 val-prior-0.5", "vision", "robustness/val_prior_shift_0.5/vision/cifar100/clip_vitb32", "top1", "C-100"),
    *[(f"C-100 val-noise-{r}", "vision", f"robustness/val_sym_{r}/vision/cifar100/clip_vitb32", "top1", "C-100")
      for r in ("0.1", "0.2", "0.3", "0.4", "0.5")],
    ("Elec unseen-corruption", "tabular", "robustness/unseen/tabular/electricity/tabpfn", "auc", "Elec"),
    ("Elec learner-xgboost", "tabular", "robustness/learner/tabular/electricity/xgboost", "auc", "Elec"),
]
GROUPS = [(*m, m[0]) for m in MAIN] + list(CONDITIONS)


def paired_radius(qhat, n, m):
    L = math.log(3 * m * (m - 1) / 2 / DELTA)
    Q = min(1.0, qhat + math.sqrt(2 * qhat * L / n) + 2 * L / n)
    return math.sqrt(2 * Q * L / n) + 2 * L / (3 * n), 2 * math.sqrt(math.log(2 * m / DELTA) / (2 * n))


def paired_radius_vec(qhat, n, m):
    L = math.log(3 * m * (m - 1) / 2 / DELTA)
    Q = np.minimum(1.0, qhat + np.sqrt(2 * qhat * L / n) + 2 * L / n)
    return np.sqrt(2 * Q * L / n) + 2 * L / (3 * n), 2 * math.sqrt(math.log(2 * m / DELTA) / (2 * n))


def load_pair(units, meta, name_a, name_b, kind):
    row = {c["name"]: c for c in meta["candidates"]}
    if name_a not in row or name_b not in row:
        return None
    a, b = row[name_a], row[name_b]
    sha = [a["rank_npz_sha256"], b["rank_npz_sha256"]]
    y = units["target"]
    if kind == "top1":
        D = units["correct"][a["row"]].astype(float) - units["correct"][b["row"]].astype(float)
        return D, len(D), float((D * D).mean()), float(D.mean()), sha
    pa, pb = units["proba_pos"][a["row"]], units["proba_pos"][b["row"]]
    pos, neg = pa[y == 1], pa[y == 0]
    qb, qbb = pb[y == 1], pb[y == 0]
    ka = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    kb = (qb[:, None] > qbb[None, :]) + 0.5 * (qb[:, None] == qbb[None, :])
    d = ka - kb
    D = ka.mean(axis=1) - kb.mean(axis=1)
    return D, int(min(len(pos), len(neg))), float((d * d).mean()), float(d.mean()), sha


def coverage(D, m, seed_key):
    n_pop = len(D)
    n2 = n_pop // 2
    g = float(D.mean())
    rng = np.random.default_rng(zlib.crc32(seed_key.encode()))
    idx = np.argsort(rng.random((RESAMPLES, n_pop)), axis=1)[:, :n2]
    d = D[idx]
    est = d.mean(axis=1)
    qhat = (d * d).mean(axis=1)
    B, r1 = paired_radius_vec(qhat, n2, m)
    dev = np.abs(est - g)
    return {"n_sub": n2, "trials": int(RESAMPLES),
            "cov_paired": float((dev <= B).mean()), "cov_range": float((dev <= r1).mean()),
            "max_ratio_paired": float((dev / B).max()), "max_ratio_range": float((dev / r1).max()),
            "min_margin_paired": float((B - dev).min()), "min_margin_range": float((r1 - dev).min())}


def main():
    rows_b, groups, missing = [], {}, []
    for label, family, rel, kind, parent in GROUPS:
        for seed in SEEDS[parent]:
            cell = BASE / rel / f"seed_{seed}"
            if not (cell / "decision.json").exists():
                missing.append([label, seed])
                continue
            dec = json.loads((cell / "decision.json").read_text())
            lb = json.loads((cell / "leaderboard.json").read_text())
            meta = json.loads((cell / "rank_units.json").read_text())
            units = np.load(cell / "rank_units.npz")
            frozen = [r for r in lb["rows"] if r["role"] != "screened_out"]
            m = len(set(r["sel_sha12"] for r in frozen))
            refs = [r["name"] for r in frozen if r["role"] == "reference"]
            el = dec["elected"]
            for b in refs:
                if b == el:
                    continue
                got = load_pair(units, meta, el, b, kind)
                if got is None:
                    missing.append([label, seed, b])
                    continue
                D, n, qhat, gain, sha = got
                B, R2 = paired_radius(qhat, n, m)
                rows_b.append(dict(group=label, family=parent, condition=(label != parent), seed=seed, elected=el, baseline=b,
                                   is_random=(b == "random"), n=n, m=m, qhat=qhat, rank_gain=gain, B=B, hoeffding_2r1=R2,
                                   ratio=B / R2, sha256=sha))
                cov = coverage(D, m, f"{label}|{seed}|{el}|{b}")
                g = groups.setdefault(label, {"family": parent, "condition": label != parent, "trials": 0, "pairs": 0,
                                              "cov_paired_sum": 0.0, "cov_range_sum": 0.0, "max_ratio_paired": 0.0,
                                              "max_ratio_range": 0.0, "min_margin_paired": math.inf, "min_margin_range": math.inf})
                g["trials"] += cov["trials"]
                g["pairs"] += 1
                g["cov_paired_sum"] += cov["cov_paired"]
                g["cov_range_sum"] += cov["cov_range"]
                g["max_ratio_paired"] = max(g["max_ratio_paired"], cov["max_ratio_paired"])
                g["max_ratio_range"] = max(g["max_ratio_range"], cov["max_ratio_range"])
                g["min_margin_paired"] = min(g["min_margin_paired"], cov["min_margin_paired"])
                g["min_margin_range"] = min(g["min_margin_range"], cov["min_margin_range"])
    for g in groups.values():
        g["cov_paired"] = g.pop("cov_paired_sum") / g["pairs"]
        g["cov_range"] = g.pop("cov_range_sum") / g["pairs"]
    out = {"batch": "the runs of the main table (main, scale_up) and the robustness conditions on the same runs (robustness)",
           "confidence": 1 - DELTA, "resamples": RESAMPLES, "subsample_frac": FRAC,
           "comparison": "elected versus every other reference-eligible candidate on ranking units",
           "groups": groups, "rows": rows_b, "missing": missing}
    assert not missing, missing
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"groups {len(groups)} pairs {len(rows_b)} trials {sum(g['trials'] for g in groups.values())} missing {len(missing)}")
    print(f"tighter pairs: {sum(r['ratio'] < 1 for r in rows_b)}/{len(rows_b)}")


if __name__ == "__main__":
    main()
