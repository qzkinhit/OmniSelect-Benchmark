"""Similarity spaces of the budget selector's redundancy penalty (``coverage.space``).

``feature``: the dot product of the track representation rows (L2-normalized by the tracks), as the
selector has always used. ``gradient``: the cosine of per-record last-layer gradients
g_i = phi_i e_i^T, which factorizes as cos(phi_i, phi_j) cos(e_i, e_j), so it is computed from the
normalized factors without forming d x m gradients (on text the factors are the LESS sketches and a
unit error). ``concat``: the mean of the two, the cosine of the concatenated unit vectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

SPACES = ("feature", "gradient", "concat")


def _unit(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


@dataclass
class Similarity:
    """Pairwise similarities of the records of one pool (or a subset of it)."""

    space: str
    features: np.ndarray                  # rows as given (the feature space uses them unchanged)
    phi: Optional[np.ndarray] = None      # unit-normalized gradient factors (gradient and concat spaces)
    err: Optional[np.ndarray] = None

    @classmethod
    def build(cls, space: str, features: np.ndarray, phi: Optional[np.ndarray] = None,
              err: Optional[np.ndarray] = None) -> "Similarity":
        """Similarity of ``space``. The gradient factors are normalized here."""
        if space not in SPACES:
            raise ValueError(f"coverage.space must be one of {SPACES}")
        if space != "feature" and (phi is None or err is None):
            raise ValueError(f"coverage.space={space} needs gradient factors")
        return cls(space, np.asarray(features), None if phi is None else _unit(phi),
                   None if err is None else _unit(err))

    def column(self, j: int) -> np.ndarray:
        """Similarity of every record to record j."""
        if self.space == "feature":
            return self.features @ self.features[j]
        grad = (self.phi @ self.phi[j]) * (self.err @ self.err[j])
        if self.space == "gradient":
            return grad
        return 0.5 * (self.features @ self.features[j] + grad)

    def block(self, rows: np.ndarray) -> np.ndarray:
        """Similarities of the records ``rows`` to every record, shape (len(rows), n)."""
        rows = np.asarray(rows, dtype=int)
        if self.space == "feature":
            return self.features[rows] @ self.features.T
        grad = (self.phi[rows] @ self.phi.T) * (self.err[rows] @ self.err.T)
        if self.space == "gradient":
            return grad
        return 0.5 * (self.features[rows] @ self.features.T + grad)

    def subset(self, idx: np.ndarray) -> "Similarity":
        """The similarity restricted to the records ``idx`` (in that order)."""
        idx = np.asarray(idx, dtype=int)
        return Similarity(self.space, self.features[idx], None if self.phi is None else self.phi[idx],
                          None if self.err is None else self.err[idx])
