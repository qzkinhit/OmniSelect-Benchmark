"""EL2N (Paul et al., NeurIPS 2021): error L2 norm ||p - onehot(y)||_2, keep the k hardest.

p is the class-probability vector of the track's probe (columns aligned to label ids) and y the
observed label. Defined only where labelled class probabilities exist (classification tracks).
Qualitative protocol transfer: the source trains ResNets from scratch and averages over inits.
The native ResNet-18 track uses its own early-checkpoint scores.
"""
from __future__ import annotations

import numpy as np

from benchmark.Methods._common import softmax
from omniselect.core.portfolio.registry import SelectionContext, register


def el2n(probs_or_logits: np.ndarray, labels: np.ndarray, k: int, *, is_logits: bool = True) -> list:
    """Top k of the error norm."""
    P = softmax(np.asarray(probs_or_logits, float)) if is_logits else np.asarray(probs_or_logits, float)
    y = np.asarray(labels).astype(int)
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    score = np.linalg.norm(P - onehot, axis=1)
    return [int(i) for i in np.argsort(-score, kind="stable")[:min(k, len(score))]]


@register("el2n", method_dir="EL2N", family="external", fidelity="qualitative protocol transfer",
          requires=("proba", "labels"), display="EL2N")
def el2n_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """EL2N from the probe probabilities."""
    return el2n(ctx.proba, ctx.labels, k, is_logits=False)
