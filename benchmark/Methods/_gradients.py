"""Last-layer gradient model shared by GLISTER and GRAD-MATCH.

A ``LastLayer`` is a linear map Theta (d x m) on features phi (the penultimate activations with a
constant column) followed by softmax cross-entropy (classification) or half squared error
(regression). The gradient of one record's loss with respect to Theta is phi e^T, where e is
p - onehot(y) or y_hat - y. Inner products of two such gradients are (phi_i . phi_j)(e_i . e_j), so
GRAD-MATCH runs in this kernel form and GLISTER evaluates gains as rows of (Phi G_V) * E. Each
track builds the model from a warm-started learner (``extras['last_layer']``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class LastLayer:
    """Features, targets and warm-start parameters of a last-layer model on the pool and on V_con."""

    kind: str                       # softmax | squared
    phi: np.ndarray                 # (n, d) pool features with a constant column
    target: np.ndarray              # (n,) labels (softmax) or (n, m) targets (squared)
    theta: np.ndarray               # (d, m) warm-start parameters
    phi_val: np.ndarray             # (n_val, d)
    target_val: np.ndarray
    labels: Optional[np.ndarray] = None   # (n,) class of each pool record for per-class matching
    source: str = ""

    def outputs(self, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Class probabilities or predictions."""
        z = phi @ theta
        return _softmax(z) if self.kind == "softmax" else z

    def errors(self, phi: np.ndarray, target: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
        """e = dLoss / dOutput per record: p - onehot(y) or y_hat - y."""
        out = self.outputs(phi, self.theta if theta is None else theta)
        if self.kind == "softmax":
            onehot = np.zeros_like(out)
            onehot[np.arange(len(target)), np.asarray(target).astype(int)] = 1.0
            return out - onehot
        return out - np.asarray(target, dtype=float)

    def step_size(self) -> float:
        """1 / mean ||phi||^2, the inverse of the smoothness bound of the per-record loss."""
        return 1.0 / max(float(np.mean(np.sum(self.phi ** 2, axis=1))), 1e-12)


def append_bias(X: np.ndarray) -> np.ndarray:
    """[X, 1]."""
    X = np.asarray(X, dtype=float)
    return np.hstack([X, np.ones((len(X), 1))])


def softmax_theta(weights: np.ndarray, intercept: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    """(d + 1) x C parameters from output weights of shape (d, C_present) or (d, 1) for a binary logistic unit.

    Classes absent from the fit get weight 0 and bias -30. A single output column is the logit of
    ``classes[1]`` against ``classes[0]``.
    """
    W = np.asarray(weights, dtype=float)
    b = np.asarray(intercept, dtype=float).reshape(-1)
    cls = np.asarray(classes).astype(int)
    d = W.shape[0]
    theta = np.zeros((d + 1, int(n_classes)))
    theta[-1, :] = -30.0
    if W.shape[1] == 1 and len(cls) == 2:
        theta[:, cls[0]] = 0.0
        theta[:d, cls[1]] = W[:, 0]
        theta[-1, cls[1]] = b[0]
        return theta
    theta[:d, cls] = W
    theta[-1, cls] = b
    return theta


def gram(phi_a: np.ndarray, e_a: np.ndarray, phi_b: np.ndarray, e_b: np.ndarray) -> np.ndarray:
    """<g_i, g_j> = (phi_i . phi_j)(e_i . e_j) for every pair (i in a, j in b)."""
    return (phi_a @ phi_b.T) * (e_a @ e_b.T)
