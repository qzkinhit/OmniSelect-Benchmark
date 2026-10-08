"""Train-once cache: one learner fit per (subset sha, fidelity key), scored on any split.

``FitCache.evaluate(subset, stage, splits)`` fits the learner once and scores the requested
splits. With ``enabled`` the entry is kept under the key (sel_sha12 of the sorted subset,
JSON of ``learner.fidelity(stage)``), so screening, ranking, confirmation and the reported fit
reuse it whenever their fidelity keys agree. Disabled, every call fits again (canonical cost).
With ``sort_subsets`` an enabled cache trains on the sorted ids (order-sensitive tracks turn it off).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from omniselect.utils.hashing import sel_sha12


class Learner(Protocol):
    """Downstream model protocol implemented by every track (tracks/common/downstream.py)."""

    keep_models: bool

    def fidelity(self, stage: str) -> dict[str, Any]:
        """Schedule and seed used by a fit at ``stage`` (con, rank, conf, report)."""

    def fit(self, subset: np.ndarray, stage: str) -> Any:
        """Train on the pool records ``subset`` and return a model."""

    def score(self, model: Any, split: str) -> dict[str, np.ndarray]:
        """Per-unit outputs of ``model`` on ``split`` (con, rank, conf, test)."""


@dataclass
class FitEntry:
    """One fitted model and its per-unit outputs by split."""

    subset_sha: str
    fidelity: dict[str, Any]
    n_records: int
    fit_secs: float
    model: Any = None
    per_unit: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    score_secs: dict[str, float] = field(default_factory=dict)


@dataclass
class FitEvent:
    """Accounting row for one evaluate() call."""

    stage: str
    subset_sha: str
    n_records: int
    cache_hit: bool
    fitted: bool
    fit_secs: float
    score_secs: float
    splits: tuple[str, ...]


class FitCache:
    """Memoizes learner fits by subset hash and fidelity key when ``enabled``."""

    def __init__(self, learner: Learner, enabled: bool, eager_splits: Sequence[str] = (),
                 sort_subsets: bool = True) -> None:
        self.learner = learner
        self.enabled = bool(enabled)
        self.sort_subsets = bool(sort_subsets)
        self.eager_splits = tuple(eager_splits)
        self._store: dict[tuple[str, str], FitEntry] = {}
        self.events: list[FitEvent] = []

    @staticmethod
    def fidelity_key(fidelity: dict[str, Any]) -> str:
        """Canonical JSON of a fidelity dict."""
        return json.dumps(fidelity, sort_keys=True, default=str)

    def evaluate(self, subset: Sequence[int], stage: str, splits: Sequence[str], eager: bool = True) -> FitEntry:
        """Return an entry whose ``per_unit`` holds every split in ``splits``.

        ``eager=False`` scores only ``splits`` (fits that no later stage reuses, such as screening).
        """
        ids = np.asarray(list(subset), dtype=np.int64)
        sha = sel_sha12(ids)
        fidelity = dict(self.learner.fidelity(stage))
        key = (sha, self.fidelity_key(fidelity))
        extra = self.eager_splits if (self.enabled and eager) else ()
        wanted = tuple(dict.fromkeys(list(splits) + list(extra)))
        entry = self._store.get(key) if self.enabled else None
        hit = entry is not None
        fitted = False
        fit_secs = 0.0
        if entry is None:
            fitted = True
            train_ids = np.sort(ids) if (self.enabled and self.sort_subsets) else ids
            start = time.perf_counter()
            model = self.learner.fit(train_ids, stage)
            fit_secs = time.perf_counter() - start
            entry = FitEntry(sha, fidelity, int(len(ids)), fit_secs, model)
            if self.enabled:
                self._store[key] = entry
        missing = [s for s in wanted if s not in entry.per_unit]
        score_secs = 0.0
        if missing and entry.model is None:
            train_ids = np.sort(ids) if (self.enabled and self.sort_subsets) else ids
            start = time.perf_counter()
            entry.model = self.learner.fit(train_ids, stage)
            refit = time.perf_counter() - start
            fit_secs += refit
            entry.fit_secs += refit
            hit = False
            fitted = True
        for split in missing:
            start = time.perf_counter()
            entry.per_unit[split] = self.learner.score(entry.model, split)
            elapsed = time.perf_counter() - start
            entry.score_secs[split] = elapsed
            score_secs += elapsed
        if self.enabled and not getattr(self.learner, "keep_models", True):
            entry.model = None
        self.events.append(
            FitEvent(stage, sha, int(len(ids)), hit and not fitted, fitted, fit_secs, score_secs, tuple(wanted))
        )
        return entry

    def stats(self) -> dict[str, Any]:
        """Totals by stage: calls, fits, cache hits, fit and score seconds, records trained."""
        out: dict[str, Any] = {"enabled": self.enabled, "by_stage": {}}
        for event in self.events:
            row = out["by_stage"].setdefault(
                event.stage,
                {"calls": 0, "fits": 0, "cache_hits": 0, "fit_secs": 0.0, "score_secs": 0.0, "records_trained": 0},
            )
            row["calls"] += 1
            row["cache_hits"] += int(event.cache_hit)
            if event.fitted:
                row["fits"] += 1
                row["records_trained"] += event.n_records
            row["fit_secs"] += event.fit_secs
            row["score_secs"] += event.score_secs
        out["fits"] = sum(r["fits"] for r in out["by_stage"].values())
        out["cache_hits"] = sum(r["cache_hits"] for r in out["by_stage"].values())
        out["distinct_subsets"] = len({e.subset_sha for e in self.events})
        return out
