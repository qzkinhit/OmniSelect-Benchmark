"""IF / MATES selection logic — influence-driven Top-K, pure-Python control flow.

This baseline follows the influence-function view of data selection: keep the
examples whose presence most *helps* the downstream model. Two faithful threads:

* **Koh & Liang (ICML 2017)** influence functions estimate the effect of a single
  training point on the model via a first-order (gradient / Hessian-vector) proxy;
  examples with the largest positive influence on the target objective are kept.
* **MATES (Yu et al., NeurIPS 2024; ``cxcscmu/MATES``)** trains a small *data
  influence model* to predict each example's per-sample influence on the reference
  loss, then selects the Top-K scoring examples — model-aligned data selection.

In both cases selection reduces to: compute a per-sample influence score, then take
the Top-K. This module keeps that logic pure and modality-agnostic. The actual
influence computation is delegated to ``mmdataselect.signals.InfluenceSignal`` (a
downstream-model per-sample loss / learnability proxy with a torch path and a
deterministic CPU fallback), or supplied directly via a precomputed ``influence``
array so an experiment can reuse externally aligned gradient/influence scores.

``select`` is a pure function of ``(records, k, ...)`` so the runner stays a thin
I/O shell, mirroring the other baselines.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from mmdataselect.datatypes import UnifiedRecord


def _topk_indices(scores: np.ndarray, k: int) -> List[int]:
    """Return the indices of the ``k`` highest scores, descending (stable ties).

    ``argsort`` is stable, so equal influence scores keep their original pool order,
    keeping selection deterministic for a fixed input.
    """
    order = np.argsort(-scores, kind="stable")
    return [int(i) for i in order[:k]]


def select(
    records: Sequence[UnifiedRecord],
    k: int,
    *,
    influence: Optional[Sequence[float]] = None,
    model_name: Optional[str] = None,
    seed: int = 0,
) -> List[int]:
    """Select ``k`` pool indices by descending per-sample influence (IF / MATES).

    Parameters
    ----------
    records : sequence of UnifiedRecord
        The candidate pool. All scoring uses the modality-agnostic ``text`` field.
    k : int
        Number of records to keep (already budget-resolved by the runner). Clamped
        into ``[0, len(records)]``. When ``k == len(records)`` the full descending
        ranking is returned, so a token-budget caller can truncate it afterwards.
    influence : optional sequence of float
        Precomputed per-sample influence scores (e.g. gradient-aligned / MATES data
        influence model outputs) reused from an experiment. When given, selection is
        a pure Top-K over these values and no model is loaded.
    model_name : optional str
        Downstream model id passed to ``InfluenceSignal`` when ``influence`` is not
        supplied. ``None`` (or an unavailable torch path) transparently falls back to
        the signal's deterministic CPU proxy, so importing/running this baseline
        never requires torch.
    seed : int
        Reserved for interface parity with the other baselines (influence Top-K is
        deterministic; kept so callers can pass a uniform signature).

    Returns
    -------
    list of int
        Selected indices into ``records``, ordered by descending influence. For
        ``k == len(records)`` this is the complete influence ranking.
    """
    n = len(records)
    k = max(0, min(int(k), n))
    if n == 0 or k == 0:
        return []

    if influence is not None:
        scores = np.asarray(influence, dtype=float)
        if scores.shape[0] != n:
            raise ValueError(
                f"influence length {scores.shape[0]} != number of records {n}"
            )
    else:
        # Lazy import: keeps torch/transformers out of this baseline's import path.
        from mmdataselect.signals import InfluenceSignal

        signal = InfluenceSignal(model_name)
        scores = np.asarray(signal.score(records), dtype=float)

    return _topk_indices(scores, k)
