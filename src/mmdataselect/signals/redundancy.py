"""Redundancy signal (Entropy-Law style) — model-free, pure CPU.

Two model-free, deterministic views of *information density*:

* per-record ``RedundancySignal.score`` — each record's own byte compression ratio
  (incompressible text = high information density = high value);
* set-level :func:`set_redundancy` — ``R(S) = 1 - compress(S)/raw(S)`` (higher =
  the set repeats itself more), used as a diagnostic and in the selection objective;
* :func:`hashed_features` — stable char n-gram hashing vectors used by the selector
  to measure marginal coverage (diversity) without any learned model.
"""
from __future__ import annotations

import zlib
from typing import Sequence

import numpy as np

from ..datatypes import UnifiedRecord
from .base import Signal


def _compress_ratio(blob: bytes) -> float:
    if not blob:
        return 0.0
    return min(1.0, len(zlib.compress(blob, 6)) / len(blob))


def set_redundancy(records: Sequence[UnifiedRecord]) -> float:
    """Entropy-Law set redundancy ``R(S) = 1 - compress(S)/raw(S)`` (higher = more redundant)."""
    blob = "\n".join(r.text for r in records).encode("utf-8")
    if not blob:
        return 0.0
    return float(1.0 - len(zlib.compress(blob, 6)) / len(blob))


def hashed_features(records: Sequence[UnifiedRecord], dim: int = 256, ngram: int = 4) -> np.ndarray:
    """Stable (crc32-hashed) char n-gram features, L2-normalized per row.

    Deterministic across runs/processes (unlike ``hash()``), so selection is reproducible.
    """
    X = np.zeros((len(records), dim), dtype=float)
    for i, r in enumerate(records):
        t = r.text or " "
        upper = max(1, len(t) - ngram + 1)
        for j in range(upper):
            h = zlib.crc32(t[j : j + ngram].encode("utf-8")) % dim
            X[i, h] += 1.0
        nrm = np.linalg.norm(X[i])
        if nrm > 0:
            X[i] /= nrm
    return X


class RedundancySignal(Signal):
    """Per-record information density = self byte compression ratio (model-free, CPU)."""

    name = "redundancy"

    def score(self, records: Sequence[UnifiedRecord]) -> np.ndarray:
        return np.array([_compress_ratio((r.text or "").encode("utf-8")) for r in records], dtype=float)
