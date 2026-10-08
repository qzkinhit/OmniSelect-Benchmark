"""Paired bootstrap gate with a Bonferroni threshold over K challengers.

The gate draws ``n_boot`` resamples of the unit indices with replacement from
``default_rng(seed)``. Each resample gives the weighted mean of the paired differences with the
weights of the drawn units renormalized to sum to one. Stratified replicates (core/gates/paired.py)
can be passed instead. p_win is the fraction of replicates whose mean is strictly positive. The
challenger is adopted iff the full-sample weighted mean is strictly positive and
p_win >= 1 - (1 - p_beat_min) / K. No finite-sample error bound is attached.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from omniselect.core.gates.base import GateResult

_CHUNK_CELLS = 4_000_000   # resamples x units drawn per chunk, bounds the memory of one draw


def bootstrap_means(differences: np.ndarray, weights: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Weighted means of ``n_boot`` paired resamples (indices drawn uniformly with replacement)."""
    d = np.asarray(differences, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = len(d)
    rng = np.random.default_rng(seed)
    rows = max(1, _CHUNK_CELLS // max(n, 1))
    out = np.empty(int(n_boot), dtype=float)
    done = 0
    while done < n_boot:
        size = min(rows, int(n_boot) - done)
        idx = rng.integers(0, n, size=(size, n))
        wi = w[idx]
        out[done:done + size] = (wi * d[idx]).sum(axis=1) / np.maximum(wi.sum(axis=1), 1e-300)
        done += size
    return out


def bootstrap_gate(
    reference: str,
    challenger: str,
    differences: np.ndarray,
    weights: np.ndarray,
    *,
    n_boot: int = 1000,
    p_beat_min: float = 0.9,
    k: int = 1,
    seed: int = 0,
    boot_means: Optional[np.ndarray] = None,
    design: str = "uniform",
) -> GateResult:
    """Paired bootstrap gate.

    differences d_i in [-1, 1] (challenger minus reference) and weights w_i summing to one over the
    same units. ``k`` is the number of challengers tested in the ordered family. ``boot_means``
    replaces the plain resampling by precomputed replicates, ``design`` names them. Returns a
    GateResult with kind "bootstrap", adopted = (mean > 0 and p_win >= 1 - (1 - p_beat_min) / k),
    and statistics {mean, p_win, threshold, n_boot, p_beat_min, K, seed, n_units, boot_q05,
    boot_q95, resampling}.
    """
    d = np.asarray(differences, dtype=float)
    w = np.asarray(weights, dtype=float)
    if d.shape != w.shape or d.ndim != 1 or len(d) == 0:
        raise ValueError("differences and weights must be non-empty vectors of equal length")
    if np.any(np.abs(d) > 1.0 + 1e-12) or not math.isclose(float(w.sum()), 1.0, rel_tol=1e-9):
        raise ValueError("differences must lie in [-1, 1] and weights must sum to one")
    if int(n_boot) < 1 or not 0.0 < float(p_beat_min) < 1.0:
        raise ValueError("n_boot must be positive and p_beat_min must lie in (0, 1)")
    k = max(int(k), 1)
    mean = float(np.dot(w, d))
    if boot_means is None:
        means = bootstrap_means(d, w, int(n_boot), int(seed))
        design = "uniform"
    else:
        means = np.asarray(boot_means, dtype=float)
        n_boot = len(means)
    p_win = float(np.mean(means > 0.0))
    threshold = 1.0 - (1.0 - float(p_beat_min)) / k
    adopted = mean > 0.0 and p_win >= threshold
    return GateResult(
        "bootstrap",
        reference,
        challenger,
        bool(adopted),
        {
            "mean": mean,
            "p_win": p_win,
            "threshold": float(threshold),
            "n_boot": int(n_boot),
            "p_beat_min": float(p_beat_min),
            "K": k,
            "seed": int(seed),
            "n_units": int(len(d)),
            "boot_q05": float(np.quantile(means, 0.05)),
            "boot_q95": float(np.quantile(means, 0.95)),
            "resampling": design,
        },
    )
