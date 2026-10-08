"""Validation splits: construction (con), ranking (rank), confirmation (conf).

``split_validation`` permutes the validation indices with ``default_rng(seed + offset)`` and
cuts them into con, rank and conf. ``two_way`` reproduces the canonical halves (V1 = con,
V2 = rank). ``time_block_zones`` partitions a forecasting time axis into contiguous zones in the
order con, rank, conf, test with a gap of at least ``gap`` steps between zones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from omniselect.config.config import SplitConfig

SPLIT_NAMES = ("con", "rank", "conf")


@dataclass
class ValidationSplits:
    """Index arrays into the validation set. Empty arrays mark an unused split."""

    con: np.ndarray
    rank: np.ndarray
    conf: np.ndarray
    blocks: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> np.ndarray:
        """Return the index array of ``name`` (con, rank or conf)."""
        if name not in SPLIT_NAMES:
            raise KeyError(f"unknown validation split {name!r}")
        return getattr(self, name)

    def sizes(self) -> dict[str, int]:
        """Number of validation records in each split."""
        return {name: int(len(self.get(name))) for name in SPLIT_NAMES}

    def check_disjoint(self) -> None:
        """Raise ValueError when two splits share an index."""
        seen: set[int] = set()
        for name in SPLIT_NAMES:
            values = set(int(i) for i in self.get(name))
            if seen & values:
                raise ValueError(f"validation split {name} overlaps an earlier split")
            seen |= values


def split_validation(n_val: int, seed: int, cfg: SplitConfig) -> ValidationSplits:
    """Cut ``range(n_val)`` into con, rank and conf according to ``cfg``.

    two_way: con = perm[: n // 2], rank = perm[n // 2 :], identical to the canonical runners.
    three_way: con, rank, conf take ``round(f * n)`` records in that order, conf takes the rest.
    rank_only: every index goes to rank in its original order.
    """
    if n_val <= 0:
        raise ValueError("the validation set is empty")
    empty = np.array([], dtype=np.int64)
    if cfg.mode == "rank_only":
        return ValidationSplits(con=empty, rank=np.arange(n_val, dtype=np.int64), conf=empty)
    perm = np.random.default_rng(seed + cfg.perm_seed_offset).permutation(n_val)
    if cfg.mode == "two_way":
        half = len(perm) // 2
        return ValidationSplits(con=perm[:half], rank=perm[half:], conf=empty)
    f_con, f_rank, _ = cfg.fractions
    n_con = int(round(f_con * n_val))
    n_rank = int(round(f_rank * n_val))
    n_con = min(n_con, n_val)
    n_rank = min(n_rank, n_val - n_con)
    out = ValidationSplits(
        con=perm[:n_con], rank=perm[n_con : n_con + n_rank], conf=perm[n_con + n_rank :]
    )
    out.check_disjoint()
    return out


def time_block_zones(
    first: int, last: int, gap: int, fractions: tuple[float, float, float], test_fraction: float = 0.5
) -> dict[str, tuple[int, int]]:
    """Contiguous start-position zones in time order con, rank, conf, test.

    ``[first, last)`` is the range of window starts after the pool segment. The second
    ``test_fraction`` of it is the test zone, as in the canonical forecasting runner. The first
    part is divided among con, rank and conf by ``fractions``. Consecutive zones are separated
    by ``gap`` start positions, so windows of length ``gap`` from different zones never overlap.
    Returns ``{name: (lo, hi)}`` half-open ranges. A zone may be empty (lo == hi).
    """
    if last <= first:
        raise ValueError("the post-pool range is empty")
    total = last - first
    test_lo = first + int(np.floor((1.0 - test_fraction) * total))
    val_hi = max(first, test_lo - gap)
    usable = max(0, val_hi - first - 2 * gap)
    widths = [int(np.floor(f * usable)) for f in fractions]
    zones: dict[str, tuple[int, int]] = {}
    cursor = first
    for name, width in zip(SPLIT_NAMES, widths):
        zones[name] = (cursor, cursor + width)
        cursor = cursor + width + gap
    zones["test"] = (test_lo, last)
    return zones


def draw_starts(rng: np.random.Generator, zone: tuple[int, int], size: int) -> np.ndarray:
    """Draw ``min(size, zone width)`` distinct start positions from a zone without replacement."""
    lo, hi = zone
    if hi <= lo:
        return np.array([], dtype=np.int64)
    positions = np.arange(lo, hi)
    return rng.choice(positions, size=min(size, len(positions)), replace=False)
