"""Paired comparison of a challenger with the reference: estimate, reading sequence and bootstrap.

``PairedSample`` holds the per-unit differences d_i in [-1, 1] of the bounded surrogate with their
weights (the weighted mean is the estimate), plus what the stratified reading order needs. The
stratified order reads units as follows: balanced accuracy draws a class uniformly and then the
next unused unit of that class, AUC pairs the next unused positive with the next unused negative
and reads the kernel difference, text draws a domain uniformly and then the next unused record,
and every other surrogate (accuracy, forecasting blocks) reads one shuffle of the units. A draw
of an exhausted stratum ends the sequence. The uniform order is one shuffle for every surrogate.
The stratified bootstrap resamples within strata (positives and negatives separately for AUC).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from omniselect.core.gates.surrogate import UnitScores, paired_differences

READING_ORDERS = ("stratified", "uniform")


@dataclass
class PairedSample:
    """Per-unit paired differences of one challenger against the reference on one split."""

    differences: np.ndarray
    weights: np.ndarray
    kind: str                                   # surrogate kind of the reference UnitScores
    strata: Optional[np.ndarray] = None
    auc_ref: Optional[np.ndarray] = None        # positive-class scores of the reference, all units
    auc_ch: Optional[np.ndarray] = None
    auc_positive: Optional[np.ndarray] = None

    @property
    def n_units(self) -> int:
        """Number of units with a difference."""
        return int(len(self.differences))

    def mean(self) -> float:
        """Weighted mean difference (the estimate of the metric gain)."""
        return float(np.dot(self.weights, self.differences))

    def design(self, order: str) -> str:
        """uniform, class, domain or auc_pairs: the sampling design used under ``order``."""
        if order not in READING_ORDERS:
            raise ValueError(f"reading order must be one of {READING_ORDERS}")
        if order == "uniform":
            return "uniform"
        if self.auc_positive is not None:
            return "auc_pairs"
        if self.strata is not None:
            return "domain" if self.kind == "text_neg_nll" else "class"
        return "uniform"

    def sequence(self, order: str, seed: int) -> np.ndarray:
        """Differences in the reading order drawn from default_rng(seed), all in [-1, 1]."""
        rng = np.random.default_rng(seed)
        design = self.design(order)
        if design == "uniform":
            return self.differences[rng.permutation(self.n_units)]
        if design == "auc_pairs":
            pos = np.flatnonzero(self.auc_positive)
            neg = np.flatnonzero(~self.auc_positive)
            pos, neg = rng.permutation(pos), rng.permutation(neg)
            m = min(len(pos), len(neg))
            return _kernel(self.auc_ch, pos[:m], neg[:m]) - _kernel(self.auc_ref, pos[:m], neg[:m])
        labels = np.asarray(self.strata)
        names = sorted(set(labels.tolist()), key=str)
        queues = {name: list(rng.permutation(np.flatnonzero(labels == name))) for name in names}
        out = []
        while True:
            name = names[int(rng.integers(len(names)))]
            if not queues[name]:
                break
            out.append(self.differences[queues[name].pop()])
        return np.asarray(out, dtype=float)

    def bootstrap_means(self, order: str, n_boot: int, seed: int) -> np.ndarray:
        """``n_boot`` bootstrap replicates of the estimate (stratified resampling under ``order``)."""
        from omniselect.core.gates.bootstrap import bootstrap_means

        design = self.design(order)
        if design == "uniform":
            return bootstrap_means(self.differences, self.weights, n_boot, seed)
        rng = np.random.default_rng(seed)
        if design == "auc_pairs":
            pos = np.flatnonzero(self.auc_positive)
            neg = np.flatnonzero(~self.auc_positive)
            out = np.empty(int(n_boot))
            for b in range(int(n_boot)):
                p = pos[rng.integers(0, len(pos), len(pos))]
                q = neg[rng.integers(0, len(neg), len(neg))]
                out[b] = _auc(self.auc_ch[p], self.auc_ch[q]) - _auc(self.auc_ref[p], self.auc_ref[q])
            return out
        labels = np.asarray(self.strata)
        names = sorted(set(labels.tolist()), key=str)
        members = [np.flatnonzero(labels == name) for name in names]
        mass = np.array([self.weights[idx].sum() for idx in members])
        out = np.zeros(int(n_boot))
        for idx, stratum_mass in zip(members, mass):
            draw = idx[rng.integers(0, len(idx), size=(int(n_boot), len(idx)))]
            w = self.weights[draw]
            out += stratum_mass * (w * self.differences[draw]).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-300)
        return out


def _kernel(score: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """1[s_pos > s_neg] + 0.5 * 1[s_pos = s_neg] for aligned positive and negative indices."""
    a, b = score[pos], score[neg]
    return (a > b).astype(float) + 0.5 * (a == b)


def _auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Mann-Whitney AUC of positive against negative scores."""
    neg_sorted = np.sort(neg_scores)
    below = np.searchsorted(neg_sorted, pos_scores, side="left")
    equal = np.searchsorted(neg_sorted, pos_scores, side="right") - below
    return float(np.mean((below + 0.5 * equal) / len(neg_sorted)))


def paired_sample(reference: UnitScores, challenger: UnitScores, clip: Optional[float] = None) -> PairedSample:
    """PairedSample of ``challenger`` against ``reference`` (same units, same weights)."""
    d, w = paired_differences(reference, challenger, clip=clip)
    return PairedSample(d, w, reference.kind, strata=reference.strata, auc_ref=reference.auc_scores,
                        auc_ch=challenger.auc_scores, auc_positive=reference.auc_positive)
