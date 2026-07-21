"""End-to-end selection: standardized pool -> signals -> Multi-Actor fusion ->
budget-aware importance x diversity selection -> result (ready for a manifest).

This is the system's public method surface. It contains no CLI / dataset / eval I/O;
that lives in the application layer (``run_mmdataselect/run_*.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..budget import Budget
from ..datatypes import UnifiedRecord
from ..fusion.console import MultiActorConsole
from ..selectors.budget_select import BudgetSelector
from ..signals.influence import InfluenceSignal
from ..signals.redundancy import RedundancySignal, hashed_features, set_redundancy


@dataclass
class SelectionResult:
    selected_idx: List[int]
    selected_ids: List[str]
    importance: np.ndarray
    weights: Dict[str, float]
    diagnostics: Dict[str, float] = field(default_factory=dict)


def build_console(
    model_name: Optional[str] = None,
    weights: Optional[Sequence[float]] = None,
    lr: float = 0.5,
) -> MultiActorConsole:
    """Default console: a redundancy actor (model-free) + an influence actor (downstream model)."""
    actors = [
        ("redundancy", RedundancySignal()),
        ("influence", InfluenceSignal(model_name=model_name)),
    ]
    return MultiActorConsole(actors, weights=weights, lr=lr)


def select_pool(
    records: Sequence[UnifiedRecord],
    budget: Budget,
    *,
    model_name: Optional[str] = None,
    lam: float = 0.5,
    method: str = "greedy",
    seed: int = 0,
    weights: Optional[Sequence[float]] = None,
) -> SelectionResult:
    n = len(records)
    if n == 0:
        return SelectionResult([], [], np.zeros(0), {}, {"n_total": 0, "n_selected": 0})

    console = build_console(model_name=model_name, weights=weights)
    actor_scores = console.actor_scores(records)
    importance = console.importance(records, scores=actor_scores)

    token_counts = [len((r.text or "").split()) for r in records]
    k = budget.resolve(n, token_counts=token_counts)

    feats = hashed_features(records)
    selector = BudgetSelector(lam=lam, method=method, seed=seed)
    idx = selector.select(records, importance, k, features=feats)

    selected = [records[i] for i in idx]
    weights_map = dict(zip(console.names, [float(w) for w in console.weights()]))
    diagnostics = {
        "n_total": n,
        "n_selected": len(idx),
        "keep_ratio": round(len(idx) / n, 4),
        "set_redundancy_pool": round(set_redundancy(records), 4),
        "set_redundancy_selected": round(set_redundancy(selected), 4),
        "mean_importance_pool": round(float(np.mean(importance)), 4),
        "mean_importance_selected": round(float(np.mean(importance[idx])) if idx else 0.0, 4),
    }
    return SelectionResult(
        selected_idx=idx,
        selected_ids=[records[i].id for i in idx],
        importance=importance,
        weights=weights_map,
        diagnostics=diagnostics,
    )
