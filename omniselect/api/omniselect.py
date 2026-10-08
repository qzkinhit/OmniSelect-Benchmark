"""Public entry point: OmniSelect.adjudicate(task) -> Election and OmniSelect.select(...).

``AdjudicationTask`` bundles what the controller needs for one pool: records, channel scores,
features, the budget, executed reference strategies and the utility callbacks per split.
``OmniSelect`` holds one OmniSelectConfig and a fusion grid and forwards to
omniselect.core.adjudication.controller.adjudicate. ``select`` is the one-call path without a
learner (the earlier ``select_pool``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from omniselect.api.select import SelectionResult, select_pool
from omniselect.config.config import OmniSelectConfig
from omniselect.core.adjudication.controller import Election, adjudicate
from omniselect.core.gates.surrogate import UnitScores
from omniselect.core.selection.budget import Budget
from omniselect.core.selection.fusion_grid import FusionGrid


@dataclass
class AdjudicationTask:
    """Inputs of one adjudication. See ``adjudicate`` for the meaning of each field."""

    records: Sequence
    scores: np.ndarray
    features: np.ndarray
    k: int
    references: Sequence[tuple[str, list]]
    gain_rank: Callable[[list], float]
    gain_con: Optional[Callable[[list], float]] = None
    cheap_gain: Optional[Callable[[list], float]] = None
    challengers: Sequence[tuple[str, list]] = field(default_factory=list)
    units: Optional[Callable[[str, list], UnitScores]] = None
    full_selection: Optional[list] = None
    random_name: str = "random"


class OmniSelect:
    """Controller facade bound to one configuration and fusion grid."""

    def __init__(self, config: Optional[OmniSelectConfig] = None, grid: Optional[FusionGrid] = None) -> None:
        self.config = config or OmniSelectConfig.preset("canonical")
        self.grid = grid if grid is not None else FusionGrid()

    def adjudicate(self, task: AdjudicationTask) -> Election:
        """Run Algorithm 1 on ``task`` and return the Election with its full trace."""
        return adjudicate(
            records=task.records, scores=task.scores, features=task.features, k=task.k,
            cfg=self.config, grid=self.grid, references=task.references, gain_rank=task.gain_rank,
            gain_con=task.gain_con, cheap_gain=task.cheap_gain, challengers=task.challengers,
            units=task.units, full_selection=task.full_selection, random_name=task.random_name,
            seed=self.config.seed,
        )

    @staticmethod
    def select(records: Sequence, budget: Budget, **kwargs) -> SelectionResult:
        """Signal fusion and budgeted selection without a learner (forwards to select_pool)."""
        return select_pool(records, budget, **kwargs)
