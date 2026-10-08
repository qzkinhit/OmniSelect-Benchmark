"""Hoeffding lower-confidence-bound gate on weighted paired differences in [-1, 1].

For differences d_i in [-1, 1] with weights w_i summing to one, the one-sided Hoeffding radius
at level delta / K is r = sqrt(2 log(K / delta) sum_i w_i^2). The challenger is adopted iff the
weighted mean minus r is strictly positive. With K = 1 and text weights this equals the
canonical text gate (core/gates/paired_text_logloss.py) after dividing by the clip C.
"""
from __future__ import annotations

import math

import numpy as np

from omniselect.core.gates.base import GateResult


def hoeffding_radius(weights: np.ndarray, delta: float, k: int = 1) -> float:
    """sqrt(2 log(k / delta) sum w^2) for differences bounded in [-1, 1]."""
    w = np.asarray(weights, dtype=float)
    return float(math.sqrt(2.0 * math.log(max(int(k), 1) / float(delta)) * float(np.sum(w ** 2))))


def lcb_gate(
    reference: str,
    challenger: str,
    differences: np.ndarray,
    weights: np.ndarray,
    delta: float,
    k: int = 1,
) -> GateResult:
    """Adopt iff sum_i w_i d_i - hoeffding_radius(w, delta, k) > 0."""
    d = np.asarray(differences, dtype=float)
    w = np.asarray(weights, dtype=float)
    if d.shape != w.shape or d.ndim != 1 or len(d) == 0:
        raise ValueError("differences and weights must be non-empty vectors of equal length")
    if np.any(np.abs(d) > 1.0 + 1e-12) or not math.isclose(float(w.sum()), 1.0, rel_tol=1e-9):
        raise ValueError("differences must lie in [-1, 1] and weights must sum to one")
    mean = float(np.dot(w, d))
    radius = hoeffding_radius(w, delta, k)
    lcb = mean - radius
    return GateResult(
        "lcb",
        reference,
        challenger,
        bool(lcb > 0.0),
        {
            "mean": mean,
            "radius": radius,
            "lcb": lcb,
            "delta": float(delta),
            "K": int(k),
            "n_units": int(len(d)),
            "sum_w2": float(np.sum(w ** 2)),
        },
    )
