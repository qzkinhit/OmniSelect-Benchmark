"""Bounded per-unit surrogates and paired differences used by the statistical gates.

``unit_scores`` maps a per-unit record to values in [0, 1] with weights summing to one, such
that the weighted mean equals the task metric or a bounded stand-in for it: correctness for
accuracy, per-class weighted correctness (balanced accuracy) for macro-F1, the per-positive
U-statistic for binary AUC, 1 - min(1, MAE / s0) per time block for forecasting, and the
negative per-record mean NLL with domain-balanced token weights for text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np


@dataclass
class UnitScores:
    """Per-unit values, their weights (summing to one) and the unit ids they refer to.

    ``strata`` labels each unit with its class (balanced accuracy) or domain (text) for stratified
    reading orders. ``auc_scores`` and ``auc_positive`` hold the positive-class score and the
    positive mask of every evaluated unit, for positive-negative pair reading (AUC).
    """

    values: np.ndarray
    weights: np.ndarray
    unit_ids: np.ndarray
    kind: str
    strata: Optional[np.ndarray] = None
    auc_scores: Optional[np.ndarray] = None
    auc_positive: Optional[np.ndarray] = None

    def mean(self) -> float:
        """Weighted mean of the values."""
        return float(np.dot(self.weights, self.values))

    def n_eff(self) -> float:
        """Effective number of units, 1 / sum(w^2)."""
        return float(1.0 / np.sum(self.weights ** 2))


def _uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def accuracy_units(pu: Mapping[str, np.ndarray]) -> UnitScores:
    """Correctness in {0, 1}, uniform weights."""
    correct = (np.asarray(pu["prediction"]) == np.asarray(pu["target"])).astype(float)
    return UnitScores(correct, _uniform(len(correct)), np.arange(len(correct)), "correctness")


def balanced_units(pu: Mapping[str, np.ndarray]) -> UnitScores:
    """Correctness weighted by 1 / (C * n_c). The weighted mean is balanced accuracy."""
    target = np.asarray(pu["target"])
    correct = (np.asarray(pu["prediction"]) == target).astype(float)
    classes, counts = np.unique(target, return_counts=True)
    per_class = dict(zip(classes.tolist(), counts.tolist()))
    weights = np.array([1.0 / (len(classes) * per_class[c]) for c in target.tolist()])
    return UnitScores(correct, weights, np.arange(len(correct)), "balanced_correctness", strata=target.copy())


def auc_units(pu: Mapping[str, np.ndarray]) -> UnitScores:
    """Binary AUC as a mean over positive units of mean_j 1[s_i > s_j] + 0.5 * 1[s_i = s_j].

    The positive class is ``classes[1]`` (the second probability column). Negatives are all
    other units. The mean over positives equals roc_auc_score(target, proba[:, 1]).
    """
    proba = np.asarray(pu["proba"], dtype=float)
    if proba.shape[1] != 2:
        raise ValueError("the AUC surrogate is defined for two probability columns")
    classes = np.asarray(pu["classes"]) if "classes" in pu else np.array([0, 1])
    target = np.asarray(pu["target"])
    score = proba[:, 1]
    positive = target == classes[1]
    pos_scores = score[positive]
    neg_sorted = np.sort(score[~positive])
    if len(pos_scores) == 0 or len(neg_sorted) == 0:
        raise ValueError("the AUC surrogate needs positive and negative units")
    below = np.searchsorted(neg_sorted, pos_scores, side="left")
    equal = np.searchsorted(neg_sorted, pos_scores, side="right") - below
    values = (below + 0.5 * equal) / len(neg_sorted)
    ids = np.flatnonzero(positive)
    return UnitScores(values, _uniform(len(values)), ids, "auc_positive_u", auc_scores=score.copy(),
                      auc_positive=positive.copy())


def forecast_units(
    pu: Mapping[str, np.ndarray], s0: float, block_steps: int = 0
) -> UnitScores:
    """1 - min(1, MAE_i / s0) per window, averaged within time blocks when ``block_steps`` > 0.

    Blocks group windows whose start positions fall into the same interval of ``block_steps``
    steps. Block values are window means and block weights are uniform.
    """
    pred = np.asarray(pu["prediction"], dtype=float)
    target = np.asarray(pu["target"], dtype=float)
    mae = np.abs(pred - target).reshape(len(pred), -1).mean(axis=1)
    values = 1.0 - np.minimum(1.0, mae / (float(s0) + 1e-12))
    if block_steps and "start" in pu:
        start = np.asarray(pu["start"]).astype(np.int64)
        block = (start - start.min()) // int(block_steps)
        ids = np.unique(block)
        block_values = np.array([values[block == b].mean() for b in ids])
        return UnitScores(block_values, _uniform(len(ids)), ids, "forecast_block")
    return UnitScores(values, _uniform(len(values)), np.arange(len(values)), "forecast_window")


def text_units(pu: Mapping[str, np.ndarray]) -> UnitScores:
    """Negative per-record mean NLL. Weight n_tokens_i / (D * tokens of the record's domain).

    Each domain carries total weight 1 / D and tokens set the weights within a domain, as in
    the canonical text gate. The weighted mean equals -log of the geometric-mean perplexity.
    """
    nll = np.asarray(pu["nll"], dtype=float)
    tokens = np.asarray(pu["n_tokens"], dtype=float)
    domain = np.asarray(pu["domain"]).astype(str)
    names = sorted(set(domain.tolist()))
    weights = np.zeros(len(nll))
    for name in names:
        mask = domain == name
        weights[mask] = tokens[mask] / (len(names) * tokens[mask].sum())
    return UnitScores(-nll, weights, np.arange(len(nll)), "text_neg_nll", strata=domain.copy())


def unit_scores(
    utility: str, pu: Mapping[str, np.ndarray], *, s0: float = 1.0, block_steps: int = 0
) -> UnitScores:
    """Dispatch on the task utility name (accuracy, macro_f1, auc, neg_mase, neg_gmean_ppl)."""
    if utility in ("accuracy",):
        return accuracy_units(pu)
    if utility in ("macro_f1", "balanced_accuracy"):
        return balanced_units(pu)
    if utility == "auc":
        return auc_units(pu)
    if utility == "neg_mase":
        return forecast_units(pu, s0=s0, block_steps=block_steps)
    if utility == "neg_gmean_ppl":
        return text_units(pu)
    raise KeyError(f"no bounded surrogate for utility {utility!r}")


def paired_differences(
    reference: UnitScores, challenger: UnitScores, clip: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit differences challenger - reference in [-1, 1] and their weights.

    Both records must refer to the same units in the same order. With ``clip`` the raw
    difference is clipped to [-clip, clip] and divided by ``clip`` (text surrogate).
    """
    if not np.array_equal(reference.unit_ids, challenger.unit_ids):
        raise ValueError("paired differences need the same units in the same order")
    if not np.allclose(reference.weights, challenger.weights):
        raise ValueError("paired differences need identical unit weights")
    diff = challenger.values - reference.values
    if clip is not None:
        diff = np.clip(diff, -clip, clip) / clip
    return np.clip(diff, -1.0, 1.0), reference.weights.copy()
