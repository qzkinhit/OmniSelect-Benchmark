"""Validation splits: canonical halves, three-way fractions, and forecasting time blocks."""
from __future__ import annotations

import numpy as np
import pytest

from omniselect.config.config import SplitConfig
from omniselect.core.adjudication.splits import draw_starts, split_validation, time_block_zones


def test_two_way_matches_canonical_halves():
    n, seed = 800, 3
    perm = np.random.default_rng(seed + 41).permutation(n)
    s = split_validation(n, seed, SplitConfig(mode="two_way"))
    assert np.array_equal(s.con, perm[:400]) and np.array_equal(s.rank, perm[400:])
    assert len(s.conf) == 0


def test_three_way_keeps_canonical_rank_size():
    for n, n_rank in ((800, 400), (1000, 500), (2000, 1000), (2500, 1250)):
        s = split_validation(n, 0, SplitConfig(mode="three_way", fractions=(0.3, 0.5, 0.2)))
        assert s.sizes()["rank"] == n_rank
        assert sum(s.sizes().values()) == n
        s.check_disjoint()


def test_time_blocks_are_ordered_and_separated():
    first, last, gap = 12000, 17300, 120
    zones = time_block_zones(first, last, gap, (0.3, 0.5, 0.2))
    order = ["con", "rank", "conf", "test"]
    for left, right in zip(order, order[1:]):
        assert zones[left][1] + gap <= zones[right][0], (left, right, zones)
    assert zones["test"] == (first + (last - first) // 2, last)
    rng = np.random.default_rng(0)
    starts = {name: draw_starts(rng, zones[name], 300) for name in order}
    for left, right in zip(order, order[1:]):
        # the last window of an earlier zone ends before the first window of the next zone starts
        assert starts[left].max() + gap <= starts[right].min()


def test_rank_only_puts_everything_in_rank():
    s = split_validation(10, 0, SplitConfig(mode="rank_only"))
    assert np.array_equal(s.rank, np.arange(10)) and len(s.con) == 0


def test_empty_validation_raises():
    with pytest.raises(ValueError):
        split_validation(0, 0, SplitConfig())
