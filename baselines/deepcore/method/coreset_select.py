"""DeepCore coreset-selection baselines (self-contained, someone else's methods).

Faithful re-implementations of the standard image data-selection / coreset baselines
collected by the DeepCore benchmark (Guo, Zhao & Bai, 2022), kept independent of our
method layer so this directory stays a clean reproduction of the original algorithms.

Methods
-------
geometric coresets (feature-space):
  - herding         Welling, "Herding Dynamical Weights to Learn", ICML 2009. Greedily pick
                    the point that pulls the running selected mean toward the full-set mean.
  - kcenter_greedy  Sener & Savarese, "Active Learning for CNNs: A Core-Set Approach",
                    ICLR 2018. Farthest-point traversal (greedy k-center).
score-based pruning (downstream per-sample signal):
  - el2n            Paul, Ganguli & Dziugaite, "Deep Learning on a Data Diet", NeurIPS 2021.
                    EL2N score = ||softmax(logits) - onehot(y)||_2, keep the top-k hardest.
  - grand           Same paper. GraNd = expected loss-gradient norm; for a linear head this is
                    exactly ||softmax - onehot|| * ||phi||, which we use.

These ship with ``run_deepcore.py``, which reproduces the image-coreset setting (CIFAR + a
frozen-encoder probe) so the behaviour can be checked against the original papers: on CLEAN
data the score-based scores keep informative hard examples and match-or-beat random at moderate
budgets, while under label noise EL2N/GraNd select the highest-error = mislabelled samples and
collapse, the documented failure mode that motivates our adaptive controller.
"""
from __future__ import annotations

import numpy as np

__all__ = ["herding", "kcenter_greedy", "el2n", "grand", "select"]


def _as2d(features: np.ndarray) -> np.ndarray:
    f = np.asarray(features, dtype=np.float64)
    return f.reshape(f.shape[0], -1)


def herding(features: np.ndarray, k: int) -> list:
    """Herding coreset (Welling 2009)."""
    X = _as2d(features)
    n = X.shape[0]; k = min(k, n)
    target = X.mean(axis=0)
    chosen, running = [], np.zeros_like(target)
    mask = np.ones(n, dtype=bool)
    for t in range(k):
        cand = (running[None, :] + X) / (t + 1)
        d = np.linalg.norm(cand - target[None, :], axis=1)
        d[~mask] = np.inf
        i = int(np.argmin(d))
        chosen.append(i); running = running + X[i]; mask[i] = False
    return chosen


def kcenter_greedy(features: np.ndarray, k: int, seed: int = 0) -> list:
    """k-center greedy / coreset (Sener & Savarese 2018)."""
    X = _as2d(features)
    n = X.shape[0]; k = min(k, n)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(n))
    chosen = [start]
    dist = np.linalg.norm(X - X[start][None, :], axis=1)
    for _ in range(1, k):
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(X - X[i][None, :], axis=1))
    return chosen


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def el2n(probs_or_logits: np.ndarray, labels: np.ndarray, k: int, *, is_logits: bool = True) -> list:
    """EL2N (Paul et al. 2021): keep the top-k highest ||p - onehot(y)||_2 (hardest)."""
    P = _softmax(np.asarray(probs_or_logits, float)) if is_logits else np.asarray(probs_or_logits, float)
    y = np.asarray(labels).astype(int)
    onehot = np.zeros_like(P); onehot[np.arange(len(y)), y] = 1.0
    score = np.linalg.norm(P - onehot, axis=1)
    return [int(i) for i in np.argsort(-score)[:min(k, len(score))]]


def grand(probs_or_logits: np.ndarray, labels: np.ndarray, features: np.ndarray, k: int,
          *, is_logits: bool = True) -> list:
    """GraNd (Paul et al. 2021): last-layer gradient-norm proxy = EL2N * ||phi||, keep top-k."""
    P = _softmax(np.asarray(probs_or_logits, float)) if is_logits else np.asarray(probs_or_logits, float)
    y = np.asarray(labels).astype(int)
    onehot = np.zeros_like(P); onehot[np.arange(len(y)), y] = 1.0
    err = np.linalg.norm(P - onehot, axis=1)
    phi = np.linalg.norm(_as2d(features), axis=1)
    return [int(i) for i in np.argsort(-(err * phi))[:min(k, len(err))]]


def select(method: str, k: int, *, features=None, probs=None, labels=None, seed: int = 0) -> list:
    """Dispatch one DeepCore method by name. ``probs`` are per-sample class probabilities."""
    if method == "herding":
        return herding(features, k)
    if method == "kcenter":
        return kcenter_greedy(features, k, seed=seed)
    if method == "el2n":
        return el2n(probs, labels, k, is_logits=False)
    if method == "grand":
        return grand(probs, labels, features, k, is_logits=False)
    raise ValueError(f"unknown DeepCore method: {method}")
