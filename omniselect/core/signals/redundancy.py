"""Redundancy channels on text: per-record compression ratio, set redundancy and hashed n-gram features.

``RedundancySignal.score`` is len(zlib(text)) / len(text) per record, clipped to 1.
``set_redundancy`` is 1 - compressed / raw size of the concatenated set. ``hashed_features`` maps
each text to an L2-normalized vector of crc32-hashed character 4-grams (256 dimensions by default),
which the budget selector uses as its similarity space on text.
"""
from __future__ import annotations

import zlib
from typing import Sequence

import numpy as np

from omniselect.core.datatypes import UnifiedRecord
from omniselect.core.signals.base import Signal


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
        """Compression ratio of each record's UTF-8 text, clipped to [0, 1]."""
        return np.array([_compress_ratio((r.text or "").encode("utf-8")) for r in records], dtype=float)
