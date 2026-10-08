"""GraNd (Paul et al., NeurIPS 2021), disclosed last-layer gradient-norm proxy.

``grand_expected`` trains five linear heads from random init for eight full-batch steps and
averages ||p - onehot(y)|| sqrt(||phi||^2 + 1) over them (the canonical row). ``grand`` is the
single-probe proxy ||p - onehot(y)|| ||phi||. Neither is a full-network reproduction.
"""
from __future__ import annotations

import numpy as np

from benchmark.Methods._common import as2d, softmax
from omniselect.core.portfolio.registry import SelectionContext, register


def grand(probs_or_logits: np.ndarray, labels: np.ndarray, features: np.ndarray, k: int,
          *, is_logits: bool = True) -> list:
    """Top k of ||p - onehot(y)|| * ||phi|| from one probe."""
    P = softmax(np.asarray(probs_or_logits, float)) if is_logits else np.asarray(probs_or_logits, float)
    y = np.asarray(labels).astype(int)
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    err = np.linalg.norm(P - onehot, axis=1)
    phi = np.linalg.norm(as2d(features), axis=1)
    score = err * phi
    return [int(i) for i in np.argsort(-score, kind="stable")[:min(k, len(score))]]


def grand_expected(features: np.ndarray, labels: np.ndarray, k: int, *,
                   K: int = 5, steps: int = 8, lr: float = 0.5, seed: int = 0) -> list:
    """Top k of the gradient-norm proxy averaged over K early-training linear heads."""
    X = as2d(np.asarray(features, float))
    n, d = X.shape
    y = np.asarray(labels).astype(int)
    C = int(y.max()) + 1
    onehot = np.zeros((n, C))
    onehot[np.arange(n), y] = 1.0
    phinorm = np.sqrt((X ** 2).sum(1) + 1.0)
    score = np.zeros(n)
    for r in range(K):
        rng = np.random.default_rng(seed * 1000 + r)
        W = rng.standard_normal((d, C)) * 0.01
        b = np.zeros(C)
        for _ in range(steps):
            P = softmax(X @ W + b)
            G = P - onehot
            W -= lr * (X.T @ G) / n
            b -= lr * G.mean(0)
        P = softmax(X @ W + b)
        score += np.linalg.norm(P - onehot, axis=1) * phinorm
    score /= K
    return [int(i) for i in np.argsort(-score, kind="stable")[:min(k, n)]]


@register("grand", method_dir="GraNd", family="external", fidelity="proxy", requires=("labels",),
          display="GraNd")
def grand_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """grand_expected on the track representation with the run seed."""
    return grand_expected(ctx.features, ctx.labels, k, seed=ctx.seed)
