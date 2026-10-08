"""DMF base variant on text: dynamic two-signal fusion followed by top-k with an optional coverage term.

The redundancy and influence signals are blended by MultiActorConsole with a single softmax weight
vector and every other console option at its default (off). With ``holdout_idx`` one weight update
is applied from each signal's proxy gain on the probe. The selection keeps the top k by importance,
or greedily adds importance plus lam times marginal coverage on hashed character n-gram features.
The module imports no torch. The influence signal falls back to a CPU proxy when no model loads.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from omniselect.core.datatypes import UnifiedRecord
from omniselect.core.selection.console import MultiActorConsole
from omniselect.core.signals.base import minmax
from omniselect.core.signals.influence import InfluenceSignal
from omniselect.core.signals.redundancy import RedundancySignal, hashed_features


def build_fusion(
    model_name: Optional[str] = None,
    lr: float = 0.5,
) -> MultiActorConsole:
    """Base dynamic multi-signal fusion controller: redundancy + influence signals.

    All advanced fusion parameters (conflict gate, curriculum anneal, group-wise
    weights, EMA and trust region) are left off, so the controller is a plain
    softmax-weighted dynamic blend of the two signals, which is the DMF baseline.
    """
    actors = [
        ("redundancy", RedundancySignal()),
        ("influence", InfluenceSignal(model_name=model_name)),
    ]
    # Only the basic dynamic theta is enabled; every upgrade flag stays default/off.
    return MultiActorConsole(actors, lr=lr)


def _holdout_rewards(
    fusion: MultiActorConsole,
    records: Sequence[UnifiedRecord],
    holdout_idx: Sequence[int],
    k: int,
) -> dict:
    """Per-signal proxy gain on a small held-out probe (black-box, no gradients).

    For each signal we rank the *training* part by that signal alone, take its
    Top-``k`` ids, and reward the signal by the mean importance those picks would
    have received on the held-out part. This is a cheap, model-free stand-in for a
    real downstream held-out gain; it only steers the basic dynamic weights and is
    skipped entirely when no hold-out is provided.
    """
    holdout = set(int(i) for i in holdout_idx)
    train_idx = [i for i in range(len(records)) if i not in holdout]
    if not train_idx or not holdout:
        return {}

    train_recs = [records[i] for i in train_idx]
    hold_recs = [records[i] for i in holdout]
    # Per-signal normalized scores on each split (rows = signals).
    S_train = fusion.actor_scores(train_recs)        # (m, |train|)
    S_hold = fusion.actor_scores(hold_recs)          # (m, |hold|)
    kk = max(1, min(int(k), len(train_idx)))

    rewards: dict = {}
    for m, name in enumerate(fusion.names):
        top = np.argsort(-S_train[m], kind="stable")[:kk]
        # Reward = how that signal scores the held-out split (proxy for transfer).
        rewards[name] = float(np.mean(S_hold[m])) if S_hold.shape[1] else 0.0
        # Tie the reward to the signal's own selectivity on the train split too,
        # so a signal that concentrates value is rewarded over a flat one.
        rewards[name] = 0.5 * rewards[name] + 0.5 * float(np.mean(S_train[m, top]))
    return rewards


def _greedy_diverse_topk(
    records: Sequence[UnifiedRecord],
    importance: np.ndarray,
    k: int,
    lam: float,
) -> List[int]:
    """Lightweight facility-location Top-K: importance + lam * marginal coverage.

    The greedy diversity step of the budget selector with one fixed ``lam`` and no
    further terms, a plain coverage-minus-redundancy augmentation of the fused
    importance ranking.
    """
    n = len(records)
    imp = minmax(importance)
    feats = hashed_features(records)
    selected: List[int] = []
    chosen = np.zeros(n, dtype=bool)
    max_sim = np.zeros(n, dtype=float)
    for _ in range(k):
        gain = imp + lam * (1.0 - max_sim)
        gain[chosen] = -np.inf
        j = int(np.argmax(gain))
        selected.append(j)
        chosen[j] = True
        sims = feats @ feats[j]  # cosine: hashed_features rows are L2-normalized
        max_sim = np.maximum(max_sim, sims)
    return selected


def select(
    records: Sequence[UnifiedRecord],
    k: int,
    *,
    model_name: Optional[str] = None,
    lr: float = 0.5,
    diversity: bool = True,
    lam: float = 0.5,
    holdout_idx: Optional[Sequence[int]] = None,
    seed: int = 0,
) -> List[int]:
    """Select ``k`` pool indices by base dynamic multi-signal fusion.

    Parameters
    ----------
    records : standardized ``UnifiedRecord`` pool (all signals read ``.text``).
    k : number of records to keep (already budget-resolved by the runner). When
        ``k == len(records)`` a *full ranking* of all indices is returned (best
        first), so a caller can truncate it under a token budget.
    model_name : optional downstream model for the influence signal; when ``None``
        (or unavailable) the influence signal uses its deterministic CPU proxy, so
        DMF runs anywhere without torch.
    lr : learning rate for the single dynamic-weight update step.
    diversity : if ``True`` (default), augment the fused importance ranking with a
        lightweight greedy coverage term; if ``False``, plain importance Top-K.
    lam : diversity weight for the greedy step (ignored when ``diversity=False``).
    holdout_idx : optional indices reserved as a probe; when given, one dynamic
        weight update is applied from each signal's proxy gain on it. When ``None``
        the update is skipped (pure base fusion).
    seed : kept for interface symmetry / reproducibility (selection is
        deterministic given the inputs).

    Returns
    -------
    list of int
        Selected pool indices; a full ranking when ``k == len(records)``.
    """
    n = len(records)
    if n == 0 or k <= 0:
        return []
    full_rank = k >= n
    k = min(int(k), n)

    fusion = build_fusion(model_name=model_name, lr=lr)
    actor_scores = fusion.actor_scores(records)  # (m, N), reuse for importance

    # Optional one-shot dynamic weight update from a held-out proxy gain.
    if holdout_idx:
        rewards = _holdout_rewards(fusion, records, holdout_idx, k)
        if rewards:
            fusion.update(rewards)

    importance = fusion.importance(records, scores=actor_scores)

    if full_rank:
        # Full ranking (best first) so the runner can truncate under a token budget.
        return [int(i) for i in np.argsort(-importance, kind="stable")]

    if diversity:
        return _greedy_diverse_topk(records, importance, k, lam)
    return [int(i) for i in np.argsort(-importance, kind="stable")[:k]]
