"""Clean oracle: a uniform draw of the budget from the records whose corruption tag is 'high'.

Diagnostic row that reads the injector's tags (``extras['tags']``), never a portfolio member. The
clean records are shuffled with default_rng(seed) and taken first, and the corrupted records follow
in a second shuffle when the clean ones do not fill the budget. On token-budgeted tracks the
whole order is returned and the track cuts it to the budget.
"""
from __future__ import annotations

import numpy as np

from omniselect.core.portfolio.registry import SelectionContext, register


def clean_oracle(tags, k: int, seed: int = 0, full_order: bool = False) -> list[int]:
    """Clean records in random order first, then the others, truncated to k unless ``full_order``."""
    tags = np.asarray(tags).astype(str)
    rng = np.random.default_rng(seed)
    clean = rng.permutation(np.flatnonzero(tags == "high"))
    other = rng.permutation(np.flatnonzero(tags != "high"))
    order = [int(i) for i in np.concatenate([clean, other])]
    return order if full_order else order[: int(k)]


@register("clean_oracle", method_dir="CleanOracle", family="control", fidelity="diagnostic",
          requires=("tags",), display="Clean oracle")
def clean_oracle_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """Budget drawn uniformly from tag-clean records (diagnostic, reads the corruption tags)."""
    full = ctx.extras.get("budget_cut") is not None
    return clean_oracle(ctx.extras["tags"], k, seed=ctx.seed, full_order=full)
