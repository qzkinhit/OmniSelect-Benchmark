"""Helpers shared by the method implementations: 2-D float view, row softmax, top-k indices.

Every method returns plain Python lists of distinct integer indices into the pool, ordered as
the method ranks them. The strategy adapters in each method.py translate a SelectionContext
into the arguments of the pure function and return exactly k indices (or a full order on
token-budgeted tracks).
"""
from __future__ import annotations

import numpy as np


def as2d(features: np.ndarray) -> np.ndarray:
    """float64 view with one row per record."""
    f = np.asarray(features, dtype=np.float64)
    return f.reshape(f.shape[0], -1)


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax with max subtraction."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def top_k(scores: np.ndarray, k: int) -> list[int]:
    """Indices of the k largest scores. Ties keep ascending pool index (stable sort of the negated scores)."""
    return [int(i) for i in np.argsort(-np.asarray(scores), kind="stable")[:k]]
