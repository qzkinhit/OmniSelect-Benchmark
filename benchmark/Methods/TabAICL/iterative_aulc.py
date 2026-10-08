"""Tab-AICL iterative protocol: cold-start active in-context learning for TabPFN, reporting AULC.

    python -m benchmark.Methods.TabAICL.iterative_aulc --dataset ionosphere --nmax 100 --batch 10

Starts from one labelled example per class. Each round fits TabPFN on the context, scores the
pool, acquires ``batch`` records by the rule (random, margin, coreset, hybrid) and records test
accuracy. AULC is the mean accuracy over the trajectory up to ``nmax`` labels. This reproduces the
source protocol on its own datasets and is not a same-protocol comparison with one-shot selection.
"""
from __future__ import annotations

import argparse

import numpy as np


def load(name: str):
    """Standardized numeric features and encoded labels of an OpenML dataset (version 1)."""
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    d = fetch_openml(name, version=1, as_frame=True)
    X = d.data.select_dtypes(include=[np.number]).fillna(0.0).to_numpy(float)
    y = LabelEncoder().fit_transform(d.target.astype(str).to_numpy())
    return StandardScaler().fit_transform(X), y


def kcenter_init(X, pool, k, seed):
    """Farthest-point traversal of ``pool`` from a seeded start."""
    rng = np.random.default_rng(seed)
    chosen = [int(rng.choice(pool))]
    dist = np.linalg.norm(X[pool] - X[chosen[0]][None], axis=1)
    while len(chosen) < k and len(chosen) < len(pool):
        i = int(np.argmax(dist))
        c = pool[i]
        chosen.append(int(c))
        dist = np.minimum(dist, np.linalg.norm(X[pool] - X[c][None], axis=1))
    return chosen[:k]


def acquire(rule, clf, X, ctx, pool, b, seed):
    """Records added to the context by one acquisition rule."""
    if rule == "random":
        return list(np.random.default_rng(seed).permutation(pool)[:b])
    if rule == "coreset":
        return kcenter_init(X, np.array(pool), b, seed)
    p = clf.predict_proba(X[pool])
    s = np.sort(p, axis=1)
    margin = s[:, -1] - (s[:, -2] if p.shape[1] > 1 else 0.0)
    if rule == "margin":
        return [int(pool[i]) for i in np.argsort(margin, kind="stable")[:b]]
    if rule == "hybrid":
        mm = [int(pool[i]) for i in np.argsort(margin, kind="stable")[:b // 2]]
        rest = [c for c in pool if c not in set(mm)]
        return mm + kcenter_init(X, np.array(rest), b - len(mm), seed)
    raise ValueError(rule)


def run_rule(rule, X, y, tr, te, seed, nmax, batch):
    """AULC and the (context size, accuracy) curve of one rule."""
    from tabpfn import TabPFNClassifier

    rng = np.random.default_rng(seed)
    ctx = [int(rng.choice(tr[y[tr] == c])) for c in np.unique(y[tr])]
    pool = [int(i) for i in tr if i not in set(ctx)]
    curve = []
    while True:
        clf = TabPFNClassifier.create_default_for_version("v2", device="cpu", ignore_pretraining_limits=True)
        clf.fit(X[ctx], y[ctx])
        curve.append((len(ctx), float((clf.predict(X[te]) == y[te]).mean())))
        if len(ctx) >= nmax or not pool:
            break
        add = acquire(rule, clf, X, ctx, pool, min(batch, len(pool), nmax - len(ctx)), seed + len(ctx))
        add = [a for a in add if a in set(pool)][:min(batch, len(pool))]
        if not add:
            add = [pool[0]]
        ctx += add
        pool = [p for p in pool if p not in set(add)]
    return float(np.mean([a for _, a in curve])), curve


def main(argv=None) -> int:
    """Print the AULC of each rule averaged over seeds."""
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(description="Tab-AICL iterative AULC protocol")
    parser.add_argument("--dataset", default="ionosphere")
    parser.add_argument("--nmax", type=int, default=100)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args(argv)
    X, y = load(args.dataset)
    rules = ["random", "margin", "coreset", "hybrid"]
    agg = {r: [] for r in rules}
    for seed in args.seeds:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.4, random_state=seed, stratify=y)
        for r in rules:
            agg[r].append(run_rule(r, X, y, tr, te, seed, args.nmax, args.batch)[0])
    base = np.mean(agg["random"])
    for r in rules:
        m = np.mean(agg[r])
        print(f"{args.dataset} {r:8} AULC={m:.4f} std={np.std(agg[r]):.4f} minus_random={m - base:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
