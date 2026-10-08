"""DSIR (Xie et al., NeurIPS 2023): importance resampling toward a clean target distribution.

``dsir_select`` scores hashed n-gram count vectors by the length-normalized log ratio of a target
and a raw bag-of-features model and takes Gumbel top-k with default_rng(seed). The text lane uses
``text_dsir.dsir_select`` with the clean held-out reference as target. Its ordering was aligned
with the official data-selection package (Spearman 0.85 on the text pool).
"""
from __future__ import annotations

import numpy as np

from benchmark.Methods.DSIR.text_dsir import dsir_select as text_dsir_select
from omniselect.core.portfolio.registry import SelectionContext, register


def dsir_select(counts: np.ndarray, target_counts: np.ndarray, k: int, *,
                seed: int = 0, smoothing: float = 1.0) -> list:
    """Gumbel top-k of the per-document log importance weight on count vectors."""
    C = np.asarray(counts, float)
    n, V = C.shape
    k = min(k, n)
    raw_total = C.sum(axis=0) + smoothing
    p_raw = raw_total / raw_total.sum()
    tc = np.asarray(target_counts, float)
    tc = tc.sum(axis=0) if tc.ndim == 2 else tc
    t_total = tc + smoothing
    p_tgt = t_total / t_total.sum()
    logw_feat = np.log(p_tgt) - np.log(p_raw)
    row_len = C.sum(axis=1) + 1e-9
    logw = (C @ logw_feat) / row_len
    rng = np.random.default_rng(seed)
    gumbel = -np.log(-np.log(rng.uniform(size=n) + 1e-12) + 1e-12)
    keys = logw + gumbel
    return [int(i) for i in np.argsort(-keys, kind="stable")[:k]]


@register("dsir", method_dir="DSIR", family="external", fidelity="exact (official ordering aligned)",
          requires=("texts", "dsir_target_texts"), display="DSIR")
def dsir_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """Text DSIR: a full importance order over the pool (the track cuts it to the token budget)."""
    texts = ctx.extras["texts"]
    ids = ctx.extras.get("ids", [str(i) for i in range(len(texts))])
    order, _ = text_dsir_select(texts, ids, len(texts), target_texts=ctx.extras["dsir_target_texts"], seed=ctx.seed)
    return order
