"""Track-side split helpers: validation index splits and forecasting start positions.

``validation_splits`` wraps omniselect.core.adjudication.splits.split_validation with the
config of the run. ``forecast_starts`` draws pool, validation and test window starts either in
the canonical layout (random validation and test zones after the pool segment) or in the v2
layout (contiguous con, rank, conf, test zones separated by L + H steps).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from omniselect.config.config import OmniSelectConfig
from omniselect.core.adjudication.splits import ValidationSplits, draw_starts, split_validation, time_block_zones


def validation_splits(n_val: int, seed: int, ocfg: OmniSelectConfig) -> ValidationSplits:
    """con, rank and conf index arrays into the validation set of ``n_val`` records."""
    return split_validation(n_val, seed, ocfg.splits)


def forecast_starts(
    rng: np.random.Generator,
    series_len: int,
    L: int,
    H: int,
    pool_n: int,
    val_n: int,
    test_n: int,
    ocfg: OmniSelectConfig,
    recent_validation: bool = False,
) -> dict[str, Any]:
    """Window start positions for pool, validation (con, rank, conf) and test.

    Canonical: pool starts come from [0, 0.7 max_start - gap). The rest splits into a validation
    zone (first half minus gap) and a test zone (second half). The draws follow one rng stream in
    the order pool, validation, test. Time blocks: the validation zone becomes three contiguous
    zones con, rank, conf with fractions ``ocfg.splits.fractions`` and gaps of L + H.
    """
    max_start = series_len - L - H
    split = int(0.7 * max_start)
    gap = L + H
    pool_starts = rng.choice(np.arange(max(1, split - gap)), size=min(pool_n, max(1, split - gap)), replace=False)
    if not (ocfg.splits.time_blocks and ocfg.splits.mode == "three_way"):
        rest = np.arange(split, max_start)
        half = len(rest) // 2
        val_zone = rest[: max(1, half - gap)]
        test_zone = rest[half:]
        if recent_validation:
            val_zone = val_zone[-max(1, len(val_zone) // 3):]
        val_starts = rng.choice(val_zone, size=min(val_n, len(val_zone)), replace=False)
        test_starts = rng.choice(test_zone, size=min(test_n, len(test_zone)), replace=False)
        return {"pool": pool_starts, "val": val_starts, "test": test_starts, "blocks": None,
                "layout": "canonical"}
    zones = time_block_zones(split, max_start, gap, ocfg.splits.fractions)
    sizes = [int(round(f * val_n)) for f in ocfg.splits.fractions]
    parts = {name: draw_starts(rng, zones[name], size) for name, size in zip(("con", "rank", "conf"), sizes)}
    val_starts = np.concatenate([parts["con"], parts["rank"], parts["conf"]]).astype(np.int64)
    test_starts = draw_starts(rng, zones["test"], test_n)
    counts = {name: int(len(parts[name])) for name in ("con", "rank", "conf")}
    return {"pool": pool_starts, "val": val_starts, "test": test_starts, "layout": "time_blocks",
            "blocks": {"zones": {k: [int(a), int(b)] for k, (a, b) in zones.items()}, "gap": gap,
                       "counts": counts, "pool_zone": [0, int(max(1, split - gap))]}}


def block_splits(counts: dict[str, int]) -> ValidationSplits:
    """con, rank, conf index arrays for validation starts concatenated in that order."""
    a = counts["con"]
    b = a + counts["rank"]
    c = b + counts["conf"]
    return ValidationSplits(con=np.arange(0, a), rank=np.arange(a, b), conf=np.arange(b, c))
