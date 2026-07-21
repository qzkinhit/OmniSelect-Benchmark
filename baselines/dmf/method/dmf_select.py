"""DMF selection logic — dynamic multi-signal fusion (base variant).

DMF (Dynamic Multi-Signal Fusion) is the representative "fuse heterogeneous
utility signals with adaptive, feedback-driven weights" baseline — the multi-signal
dynamic-fusion comparison our system aims to surpass. This is the *base* variant:
only the basic dynamic weighting (a single learnable theta over the signals,
softmax-blended) is active. Every advanced fusion mechanism (sample-level conflict
gating, the curriculum prior, group-wise weights, an authenticity/truthfulness
front-gate) is deliberately left OFF, so DMF reflects only plain dynamic
multi-signal fusion and isolates what those extra mechanisms add.

Pipeline (pure function of ``(records, k, ...)``):

1. fuse two model-free / downstream-aligned signals — a ``redundancy``
   (information-density) signal and an ``influence`` (downstream learnability)
   signal — into one per-record ``importance`` via the dynamic-fusion controller
   with all upgrade parameters at their defaults (i.e. closed);
2. optionally run *one* dynamic-weight update from each signal's proxy gain on a
   small held-out probe (skipped when no hold-out is available), so the fusion
   weights adapt to which signal actually helps before the final scoring;
3. keep the importance Top-K, optionally augmented with a lightweight greedy
   diversity term over hashed char n-gram features (coverage minus redundancy
   against the already-kept set), mirroring the system's facility-location step
   but with a single fixed lambda and no advanced mechanisms.

The core controller class is reused as-is; this module never re-implements fusion.
``torch``/``transformers`` are only ever reached lazily *inside* the influence
signal, which falls back to a deterministic CPU proxy when they are missing, so
importing this baseline never requires torch.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from mmdataselect.datatypes import UnifiedRecord
from mmdataselect.fusion.console import MultiActorConsole
from mmdataselect.signals.base import minmax
from mmdataselect.signals.influence import InfluenceSignal
from mmdataselect.signals.redundancy import RedundancySignal, hashed_features


def build_fusion(
    model_name: Optional[str] = None,
    lr: float = 0.5,
) -> MultiActorConsole:
    """Base dynamic multi-signal fusion controller: redundancy + influence signals.

    All advanced fusion parameters (conflict gate, curriculum anneal, group-wise
    weights, EMA / trust-region) are left at their defaults, i.e. closed — so the
    controller behaves as a plain softmax-weighted dynamic blend of the two
    signals, which is exactly the DMF baseline we want to compare against.
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
        top = np.argsort(-S_train[m])[:kk]
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

    Mirrors the system's greedy diversity step but with a single fixed ``lam`` and
    no advanced mechanisms — a plain coverage-minus-redundancy augmentation of the
    fused importance ranking.
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
