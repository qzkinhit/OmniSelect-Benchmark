"""Betting e-process gate against H0: mean paired difference <= -eps.

The wealth is W_0 = 1 and W_t = W_{t-1} (1 + lambda_t (z_t + eps)) over a reading sequence z_t of
paired differences in [-1, 1]. The stake lambda_t is the aGRAPA rule mu / (sigma2 + mu^2) of the
values z_s + eps seen before step t, clipped to [0, 1 / (2 (1 - eps))]. The default sequence is
one shuffle of the units from ``default_rng(seed)``. A stratified sequence (core/gates/paired.py)
can be passed instead. The e-value is sup_t W_t. The challenger is adopted iff e >= K / delta and
the weighted mean difference is strictly positive.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from omniselect.core.gates.base import GateResult

_LOG_CAP = 690.0   # exp(690) is finite in float64, the stored e_value is capped there


@dataclass
class WealthPath:
    """Log-wealth after each unit, the stakes, and the running supremum of the wealth."""

    log_wealth: np.ndarray          # log W_t for t = 1..n
    stakes: np.ndarray              # lambda_t for t = 1..n
    order: np.ndarray               # position in the reading sequence
    log_sup: float                  # log sup_{0 <= t <= n} W_t (W_0 = 1)

    def stop_index(self, log_threshold: float) -> Optional[int]:
        """First t (1-based number of units consumed) with log W_t >= log_threshold, else None."""
        hits = np.flatnonzero(self.log_wealth >= log_threshold - 1e-12)
        return int(hits[0]) + 1 if len(hits) else None


def agrapa_wealth(y: np.ndarray, eps: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Log-wealth and stakes of the aGRAPA betting sequence on ordered observations y in [-(1-eps), 1+eps].

    lambda_1 = 0. For t >= 2 the stake uses the mean mu and the variance sigma2 of y_1..y_{t-1}
    and equals clip(mu / (sigma2 + mu^2), 0, 1 / (2 (1 - eps))).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    cap = 1.0 / (2.0 * (1.0 - float(eps)))
    csum = np.concatenate([[0.0], np.cumsum(y)[:-1]])
    csq = np.concatenate([[0.0], np.cumsum(y ** 2)[:-1]])
    seen = np.arange(n, dtype=float)
    mu = np.divide(csum, seen, out=np.zeros(n), where=seen > 0)
    second = np.divide(csq, seen, out=np.zeros(n), where=seen > 0)   # sigma2 + mu^2
    stakes = np.divide(mu, second, out=np.zeros(n), where=second > 1e-15)
    stakes = np.clip(stakes, 0.0, cap)
    log_wealth = np.cumsum(np.log1p(stakes * y))
    return log_wealth, stakes


def wealth_path(sequence: np.ndarray, *, eps: float = 0.0) -> WealthPath:
    """aGRAPA wealth over the sequence in the given order."""
    y = np.asarray(sequence, dtype=float) + float(eps)
    log_wealth, stakes = agrapa_wealth(y, eps)
    log_sup = float(max(0.0, log_wealth.max())) if len(log_wealth) else 0.0
    return WealthPath(log_wealth, stakes, np.arange(len(y)), log_sup)


def eprocess_gate(
    reference: str,
    challenger: str,
    differences: np.ndarray,
    weights: np.ndarray,
    *,
    delta: float = 0.05,
    eps: float = 0.0,
    k: int = 1,
    seed: int = 0,
    sequence: Optional[np.ndarray] = None,
    design: str = "uniform",
) -> GateResult:
    """Betting e-process gate.

    differences d_i in [-1, 1] (challenger minus reference) and weights w_i summing to one give the
    estimate. ``sequence`` is the reading sequence (default: d shuffled by default_rng(seed)) and
    ``design`` its name. ``delta`` is the family error level, ``eps`` the harm tolerance of H0,
    ``k`` the number of challengers tested in the ordered family. Returns a GateResult with kind
    "eprocess", adopted = (e >= k / delta and mean > 0), and statistics {e_value, log_e_value,
    threshold = k / delta, mean, eps, delta, K, seed, n_units, n_read, stop_index,
    final_log_wealth, max_stake, reading}. stop_index is the number of units read when the wealth
    first reached the threshold (None when it never did).
    """
    d = np.asarray(differences, dtype=float)
    w = np.asarray(weights, dtype=float)
    if d.shape != w.shape or d.ndim != 1 or len(d) == 0:
        raise ValueError("differences and weights must be non-empty vectors of equal length")
    if np.any(np.abs(d) > 1.0 + 1e-12) or not math.isclose(float(w.sum()), 1.0, rel_tol=1e-9):
        raise ValueError("differences must lie in [-1, 1] and weights must sum to one")
    if not 0.0 < float(delta) < 1.0 or not 0.0 <= float(eps) < 1.0:
        raise ValueError("delta must lie in (0, 1) and eps in [0, 1)")
    k = max(int(k), 1)
    d = np.clip(d, -1.0, 1.0)
    mean = float(np.dot(w, d))
    if sequence is None:
        sequence = d[np.random.default_rng(seed).permutation(len(d))]
        design = "uniform"
    sequence = np.clip(np.asarray(sequence, dtype=float), -1.0, 1.0)
    path = wealth_path(sequence, eps=eps)
    threshold = k / float(delta)
    log_threshold = math.log(threshold)
    adopted = path.log_sup >= log_threshold - 1e-12 and mean > 0.0
    return GateResult(
        "eprocess",
        reference,
        challenger,
        bool(adopted),
        {
            "e_value": float(math.exp(min(path.log_sup, _LOG_CAP))),
            "log_e_value": float(path.log_sup),
            "threshold": float(threshold),
            "mean": mean,
            "eps": float(eps),
            "delta": float(delta),
            "K": k,
            "seed": int(seed),
            "n_units": int(len(d)),
            "n_read": int(len(sequence)),
            "stop_index": path.stop_index(log_threshold),
            "final_log_wealth": float(path.log_wealth[-1]) if len(sequence) else 0.0,
            "max_stake": float(path.stakes.max()) if len(sequence) else 0.0,
            "reading": design,
        },
    )
