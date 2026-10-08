"""IF and MATES transfer: top-k records by per-record influence score.

The score is a precomputed ``influence`` array when given, otherwise InfluenceSignal (per-record
downstream-model loss with a CPU fallback). This is the selection rule shared by Koh and Liang
(ICML 2017) influence functions and MATES (Yu et al., NeurIPS 2024). Neither the Hessian-vector
estimate nor the MATES data-influence model is computed here.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from omniselect.core.datatypes import UnifiedRecord


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
        from omniselect.core.signals.influence import InfluenceSignal

        signal = InfluenceSignal(model_name)
        scores = np.asarray(signal.score(records), dtype=float)

    return _topk_indices(scores, k)


# ---- strategy adapter (text track) ------------------------------------------------------------
from omniselect.core.portfolio.registry import SelectionContext, register  # noqa: E402


@register("if_mates", method_dir="IFMates", family="external", fidelity="local implementation",
          requires=("influence", "records"), display="IF / MATES")
def if_mates_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """Full descending influence order (the text track cuts it to the token budget)."""
    return [int(i) for i in select(ctx.records, len(ctx.records), influence=ctx.influence)]
