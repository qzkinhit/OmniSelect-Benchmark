"""GRAD-MATCH (Killamsetty et al., ICML 2021): orthogonal matching pursuit on per-record last-layer gradients.

For each class c (one group for regression) the budget share round(k n_c / n) is filled by
non-negative OMP that approximates the summed training gradient of the class (``target="train"``,
the paper's default) or the summed V_con gradient of the class (``target="val"``) with a weighted
sum of member gradients, with ridge regularization ``lam`` (the authors' OrthogonalMP_REG). OMP
runs in kernel form on <g_i, g_j> = (phi_i . phi_j)(e_i . e_j). A group larger than 2000 records
adds ``ceil(budget / 50)`` atoms per iteration. Unfilled slots take seeded random records, as in the
authors' code. The OMP weights are recorded by ``gradmatch`` but the run record trains unweighted.
"""
from __future__ import annotations

import math

import numpy as np

from benchmark.Methods._gradients import LastLayer, gram
from omniselect.core.portfolio.registry import SelectionContext, register


def omp_kernel(K: np.ndarray, b: np.ndarray, b_norm2: float, budget: int, *, lam: float = 0.5,
               tol: float = 1e-4, atoms_per_iter: int = 1) -> tuple[list[int], list[float]]:
    """Non-negative regularized OMP given K = A^T A, b = A^T y and ||y||^2. Returns (support, weights)."""
    n = len(b)
    support: list[int] = []
    banned: set[int] = set()            # atoms removed for a negative weight are not added again
    x = np.zeros(0)
    budget = min(int(budget), n)
    while len(support) < budget:
        resid_corr = b - (K[:, support] @ x if support else 0.0)
        resid2 = b_norm2 - 2 * float(x @ b[support]) + float(x @ K[np.ix_(support, support)] @ x) if support \
            else b_norm2
        if b_norm2 <= 0 or resid2 / b_norm2 < tol ** 2:
            break
        corr = np.array(resid_corr, dtype=float)
        corr[support] = -np.inf
        corr[list(banned)] = -np.inf
        take = min(atoms_per_iter, budget - len(support))
        new = [int(i) for i in np.argsort(-corr, kind="stable")[:take] if corr[i] > 0]
        if not new:
            break
        support.extend(new)
        while True:
            A = K[np.ix_(support, support)] + lam * np.eye(len(support))
            x = np.linalg.lstsq(A, b[support], rcond=None)[0]
            if len(x) == 0 or x.min() >= 0:
                break
            drop = int(np.argmin(x))
            banned.add(support.pop(drop))
            if not support:
                x = np.zeros(0)
                break
        if not support:
            break
    return support, [float(v) for v in x]


def _allocate(counts: np.ndarray, k: int) -> np.ndarray:
    """Largest-remainder allocation of k slots proportional to ``counts``."""
    raw = counts / counts.sum() * k
    base = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - base), kind="stable")[: k - base.sum()]:
        base[i] += 1
    return np.minimum(base, counts)


def gradmatch(model: LastLayer, k: int, *, target: str = "train", lam: float = 0.5, seed: int = 0,
              per_class: bool = True) -> tuple[list[int], np.ndarray]:
    """GRAD-MATCH selection of exactly k records and their OMP weights (1 for random fill)."""
    n = len(model.phi)
    k = min(int(k), n)
    errors = model.errors(model.phi, model.target)
    e_val = model.errors(model.phi_val, model.target_val) if target == "val" else None
    if per_class and model.labels is not None:
        labels = np.asarray(model.labels)
        classes = np.unique(labels)
        groups = [np.flatnonzero(labels == c) for c in classes]
        val_labels = np.asarray(model.target_val) if model.kind == "softmax" else None
    else:
        classes, groups, val_labels = np.array([0]), [np.arange(n)], None
    shares = _allocate(np.array([len(g) for g in groups], dtype=float), k)
    chosen: list[int] = []
    weights: dict[int, float] = {}
    for c, idx, share in zip(classes, groups, shares):
        if share <= 0:
            continue
        K = gram(model.phi[idx], errors[idx], model.phi[idx], errors[idx])
        if target == "val":
            vmask = np.ones(len(model.phi_val), dtype=bool) if val_labels is None else val_labels == c
            if not vmask.any():
                continue
            cross = gram(model.phi[idx], errors[idx], model.phi_val[vmask], e_val[vmask])
            b = cross.sum(axis=1)
            b_norm2 = float(gram(model.phi_val[vmask], e_val[vmask], model.phi_val[vmask], e_val[vmask]).sum())
        else:
            b = K.sum(axis=1)
            b_norm2 = float(K.sum())
        atoms = 1 if len(idx) <= 2000 else int(math.ceil(share / 50))
        support, x = omp_kernel(K, b, b_norm2, int(share), lam=lam, atoms_per_iter=atoms)
        for s, w in zip(support, x):
            chosen.append(int(idx[s]))
            weights[int(idx[s])] = w
    rng = np.random.default_rng(seed)
    if len(chosen) < k:
        rest = np.setdiff1d(np.arange(n), np.asarray(chosen, dtype=int))
        fill = [int(i) for i in rng.choice(rest, size=k - len(chosen), replace=False)]
        chosen.extend(fill)
        weights.update({i: 1.0 for i in fill})
    chosen = chosen[:k]
    return chosen, np.array([weights[i] for i in chosen])


@register("gradmatch", method_dir="GRADMATCH", family="new", fidelity="published-core transfer",
          requires=("last_layer",), display="GRAD-MATCH")
def gradmatch_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """Per-class GRAD-MATCH toward the summed training gradient on the track's warm-started last layer."""
    model = ctx.extras["last_layer"]
    model = model() if callable(model) else model
    chosen, _ = gradmatch(model, k, target=str(ctx.extras.get("gradmatch_target", "train")), seed=ctx.seed)
    return chosen
