"""Learner protocol: fit(subset, stage) -> model and score(model, split) -> per-unit arrays.

``StagePlan`` maps each fit stage (con, rank, conf, member, report, screen) to a schedule and a
seed. Canonical scoring fits use the scoring schedule and stage seeds of the 2026-07 runners. With
``fidelity.scoring_equals_reported`` every stage uses the reported schedule and one seed, so
the train-once cache can share fits. The ``screen`` stage is the short schedule of low-fidelity
screening. ``classification_units`` builds the per-unit dict of a fitted classifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from omniselect.config.config import OmniSelectConfig

STAGES = ("con", "rank", "conf", "member", "report")
SCREEN = "screen"


@dataclass
class StagePlan:
    """Schedule and seed of each stage, plus the reported schedule used for cost factors."""

    stages: dict[str, dict[str, Any]]
    reported: dict[str, Any]
    schedule_key: str = ""

    def fidelity(self, stage: str) -> dict[str, Any]:
        """Schedule and seed of ``stage``."""
        return dict(self.stages[stage])

    def cost_factor(self, stage: str) -> float:
        """Schedule length of ``stage`` over the reported schedule length (1.0 without a key)."""
        if not self.schedule_key:
            return 1.0
        return float(self.stages[stage][self.schedule_key]) / float(self.reported[self.schedule_key])


def stage_plan(
    ocfg: OmniSelectConfig,
    *,
    scoring: Mapping[str, Any],
    reported: Mapping[str, Any],
    seeds: Mapping[str, Any],
    schedule_key: str = "",
    low: Optional[Mapping[str, Any]] = None,
) -> StagePlan:
    """Build the StagePlan of a learner.

    scoring and reported are schedule dicts (for example {"max_iter": 150}). Seeds maps each of
    con, rank, report to the seed the canonical runner used (conf and member reuse rank and con).
    ``low`` is the short schedule of the screen stage (low-fidelity screening), capped at the
    construction schedule. Without it the screen stage keeps the construction schedule. The screen
    stage records the subsample fraction ``screening.low_fidelity_fraction`` that the driver applies
    to each selection.
    """
    reported = dict(reported)
    report = {**reported, "seed": seeds["report"]}
    if ocfg.fidelity.scoring_equals_reported:
        stages = {stage: dict(report) for stage in STAGES}
    else:
        con = {**dict(scoring), "seed": seeds["con"]}
        rank = {**dict(scoring), "seed": seeds["rank"]}
        stages = {"con": con, "rank": rank, "conf": dict(rank), "member": dict(con), "report": report}
    low = dict(low or {})
    if schedule_key and schedule_key in low and schedule_key in stages["con"]:
        low[schedule_key] = min(low[schedule_key], stages["con"][schedule_key])   # never longer than con
    stages[SCREEN] = {**stages["con"], **low, "subsample": float(ocfg.screening.low_fidelity_fraction)}
    return StagePlan(stages=stages, reported=reported, schedule_key=schedule_key)


@dataclass
class Learner:
    """Base learner. Subclasses implement ``fit`` and ``score``."""

    plan: StagePlan
    keep_models: bool = True
    name: str = ""
    info: dict[str, Any] = field(default_factory=dict)

    def fidelity(self, stage: str) -> dict[str, Any]:
        """Schedule and seed of a fit at ``stage``."""
        return self.plan.fidelity(stage)

    def cost_factor(self, stage: str) -> float:
        """Fraction of one reported-schedule training that a fit at ``stage`` costs."""
        return self.plan.cost_factor(stage)

    def fit(self, subset: np.ndarray, stage: str) -> Any:
        """Train on the pool records ``subset``."""
        raise NotImplementedError

    def score(self, model: Any, split: str) -> dict[str, np.ndarray]:
        """Per-unit outputs on ``split``."""
        raise NotImplementedError


def classification_units(
    model: Any, X: np.ndarray, y: np.ndarray, *, proba: bool = True, predict_from_proba: bool = False,
    n_classes: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """target, prediction, correct and, when available, proba with the model's class order.

    ``prediction`` comes from ``model.predict`` unless ``predict_from_proba`` (TabPFN rows of the
    canonical tabular runner used the argmax of predict_proba).
    """
    out: dict[str, np.ndarray] = {"target": np.asarray(y)}
    probabilities = None
    if proba and hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X))
        out["proba"] = probabilities.astype(np.float32)
        out["classes"] = np.asarray(getattr(model, "classes_", np.arange(probabilities.shape[1])))
    if predict_from_proba and probabilities is not None:
        prediction = out["classes"][probabilities.argmax(axis=1)] if "classes" in out else probabilities.argmax(1)
    else:
        prediction = np.asarray(model.predict(X))
    out["prediction"] = prediction
    out["correct"] = (prediction == np.asarray(y)).astype(np.float32)
    return out


def resolve_device(device: str) -> str:
    """cuda, mps or cpu. ``auto`` picks the first available in that order."""
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
