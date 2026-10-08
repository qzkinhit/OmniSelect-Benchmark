"""select_pool: learner-free selection of a text pool under a budget.

Scores the records with a redundancy and an influence actor, blends them with MultiActorConsole,
resolves the budget, and runs BudgetSelector(lam) on hashed n-gram features. Returns the selected
indices and ids, the importance vector, the actor weights and set-redundancy diagnostics. The
benchmark tracks do not use this path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from omniselect.core.selection.budget import Budget
from omniselect.core.datatypes import UnifiedRecord
from omniselect.core.selection.console import MultiActorConsole
from omniselect.core.selection.budget_select import BudgetSelector
from omniselect.core.signals.influence import InfluenceSignal
from omniselect.core.signals.redundancy import RedundancySignal, hashed_features, set_redundancy


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
