"""Headroom precheck: record how far the full pool and the best reference beat random on V_rank.

The headroom is the weighted mean of the bounded surrogate of the full-pool fit minus that of
the random subset, both on the ranking split, and the same quantity for the best
reference-eligible candidate. r1 = sqrt(log(1 / delta) / (2 n_eff)) with n_eff = 1 / sum(w^2).
The threshold is multiplier * r1 (kind "radius") or threshold * |mean(random)| (kind
"relative"). With ``decide`` false the result is recorded and the election is unchanged. With
``decide`` true the controller elects random when headroom < threshold. Both fits are part of
every benchmark run, so the precheck adds no training.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional

from omniselect.core.gates.surrogate import UnitScores


@dataclass
class PrecheckResult:
    """Headroom statistics and whether adjudication was skipped."""

    enabled: bool
    decide: bool
    skipped: bool
    headroom: float
    best_reference_headroom: Optional[float]
    r1: float
    threshold: float
    threshold_kind: str
    n_eff: float
    delta: float
    multiplier: float
    relative_threshold: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return asdict(self)


def radius_r1(n_eff: float, delta: float) -> float:
    """One-sided Hoeffding radius for a [0, 1] mean over n_eff units."""
    return math.sqrt(math.log(1.0 / float(delta)) / (2.0 * float(n_eff)))


def headroom_precheck(
    full: UnitScores,
    random: UnitScores,
    *,
    delta: float,
    multiplier: float,
    enabled: bool = True,
    decide: bool = False,
    threshold_kind: str = "radius",
    relative_threshold: float = 0.03,
    best_reference: Optional[UnitScores] = None,
) -> PrecheckResult:
    """Headroom of full and of the best reference over random on the same units, and the skip verdict."""
    if threshold_kind not in ("radius", "relative"):
        raise ValueError(f"threshold_kind must be radius or relative, got {threshold_kind!r}")
    n_eff = min(full.n_eff(), random.n_eff())
    headroom = full.mean() - random.mean()
    best_ref_headroom = None if best_reference is None else float(best_reference.mean() - random.mean())
    r1 = radius_r1(n_eff, delta)
    if threshold_kind == "radius":
        threshold = float(multiplier) * r1
    else:
        threshold = float(relative_threshold) * abs(float(random.mean()))
    return PrecheckResult(
        enabled=bool(enabled),
        decide=bool(decide),
        skipped=bool(enabled and decide and headroom < threshold),
        headroom=float(headroom),
        best_reference_headroom=best_ref_headroom,
        r1=float(r1),
        threshold=float(threshold),
        threshold_kind=threshold_kind,
        n_eff=float(n_eff),
        delta=float(delta),
        multiplier=float(multiplier),
        relative_threshold=float(relative_threshold),
    )
