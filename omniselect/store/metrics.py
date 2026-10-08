"""Metrics computed from per-unit arrays, shared by the driver and by recompute.py.

A per-unit record is a dict of arrays for one candidate on one split. Classification keeps
``target``, ``prediction`` and optionally ``proba`` with ``classes``. Forecasting keeps
``target`` and ``prediction`` of shape (units, H) and ``last_value``. Text keeps per-record
``nll`` (mean over tokens), ``n_tokens`` and ``domain``. Every metric here reads only these keys.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

PerUnit = Mapping[str, np.ndarray]


def accuracy(pu: PerUnit) -> float:
    """Fraction of units with prediction equal to target."""
    return float((np.asarray(pu["prediction"]) == np.asarray(pu["target"])).mean())


def macro_f1(pu: PerUnit) -> float:
    """sklearn f1_score(target, prediction, average='macro'), the canonical TEP metric."""
    from sklearn.metrics import f1_score

    return float(f1_score(pu["target"], pu["prediction"], average="macro"))


def balanced_accuracy(pu: PerUnit) -> float:
    """Mean over classes present in target of the per-class recall."""
    target = np.asarray(pu["target"])
    pred = np.asarray(pu["prediction"])
    classes = np.unique(target)
    return float(np.mean([(pred[target == c] == c).mean() for c in classes]))


def per_class_recall(pu: PerUnit) -> dict[str, float]:
    """Recall of each class present in target, keyed by the class id as a string."""
    target = np.asarray(pu["target"])
    pred = np.asarray(pu["prediction"])
    return {str(int(c)): float((pred[target == c] == c).mean()) for c in np.unique(target)}


def auc(pu: PerUnit) -> float:
    """ROC AUC as the canonical tabular runner computes it.

    Two probability columns: roc_auc_score(target, proba[:, 1]). Otherwise the mean over the
    classes present in ``target`` of the binary AUC of (target == c) against the probability
    column of c, with a zero column for a class the model was not trained on. When the model
    covers every target class this equals roc_auc_score(target, proba, multi_class="ovr").
    """
    from sklearn.metrics import roc_auc_score

    proba = np.asarray(pu["proba"], dtype=float)
    target = np.asarray(pu["target"])
    if proba.shape[1] == 2:
        return float(roc_auc_score(target, proba[:, 1]))
    classes = list(np.asarray(pu["classes"])) if "classes" in pu else list(range(proba.shape[1]))
    present = np.unique(target)
    if len(present) < 2:
        raise ValueError("AUC needs at least two classes in target")
    column = {c: j for j, c in enumerate(classes)}
    scores = [roc_auc_score(target == c, proba[:, column[c]] if c in column else np.zeros(len(target)))
              for c in present]
    return float(np.mean(scores))


def ece(pu: PerUnit, bins: int = 15) -> float:
    """Expected calibration error of the top-class probability over ``bins`` equal-width bins."""
    proba = np.asarray(pu["proba"], dtype=float)
    classes = np.asarray(pu["classes"]) if "classes" in pu else np.arange(proba.shape[1])
    conf = proba.max(axis=1)
    pred = classes[proba.argmax(axis=1)]
    correct = (pred == np.asarray(pu["target"])).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def mase(pu: PerUnit) -> float:
    """mean|prediction - target| / (mean|target - last_value| + 1e-9), the canonical runner's MASE."""
    pred = np.asarray(pu["prediction"], dtype=float)
    target = np.asarray(pu["target"], dtype=float)
    last = np.asarray(pu["last_value"], dtype=float).reshape(-1, 1)
    mae = np.abs(pred - target).mean()
    naive = np.abs(target - last).mean()
    return float(mae / (naive + 1e-9))


def neg_mase(pu: PerUnit) -> float:
    """Negative MASE, the utility maximized by the forecasting tracks."""
    return -mase(pu)


def mase_pool_scale(pu: PerUnit, s0: float) -> float:
    """Mean absolute error divided by a fixed scale ``s0`` measured on the pool segment."""
    pred = np.asarray(pu["prediction"], dtype=float)
    target = np.asarray(pu["target"], dtype=float)
    return float(np.abs(pred - target).mean() / (float(s0) + 1e-9))


def per_domain_ppl(pu: PerUnit) -> dict[str, float]:
    """Token-weighted perplexity per domain: exp(sum nll * n_tokens / sum n_tokens)."""
    nll = np.asarray(pu["nll"], dtype=float)
    tokens = np.asarray(pu["n_tokens"], dtype=float)
    domain = np.asarray(pu["domain"]).astype(str)
    out = {}
    for name in sorted(set(domain.tolist())):
        mask = domain == name
        out[name] = float(np.exp((nll[mask] * tokens[mask]).sum() / max(tokens[mask].sum(), 1.0)))
    return out


def gmean_ppl(pu: PerUnit) -> float:
    """Geometric mean over domains of the per-domain perplexity."""
    values = list(per_domain_ppl(pu).values())
    return float(np.exp(np.mean(np.log(values))))


def neg_gmean_ppl(pu: PerUnit) -> float:
    """Negative geometric-mean perplexity, the utility maximized by the text track."""
    return -gmean_ppl(pu)


def neg_mean_log_ppl(pu: PerUnit) -> float:
    """Negative mean over domains of the token-weighted mean NLL (= -log gmean_ppl)."""
    return -float(np.log(gmean_ppl(pu)))


METRICS: dict[str, Callable[..., Any]] = {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "balanced_accuracy": balanced_accuracy,
    "per_class_recall": per_class_recall,
    "auc": auc,
    "ece": ece,
    "mase": mase,
    "neg_mase": neg_mase,
    "per_domain_ppl": per_domain_ppl,
    "gmean_ppl": gmean_ppl,
    "neg_gmean_ppl": neg_gmean_ppl,
    "neg_mean_log_ppl": neg_mean_log_ppl,
}

# Utilities are maximized. Each maps to (metric name, sign applied to the metric).
UTILITIES: dict[str, tuple[str, float]] = {
    "accuracy": ("accuracy", 1.0),
    "macro_f1": ("macro_f1", 1.0),
    "balanced_accuracy": ("balanced_accuracy", 1.0),
    "auc": ("auc", 1.0),
    "neg_mase": ("mase", -1.0),
    "neg_gmean_ppl": ("gmean_ppl", -1.0),
}

# Scalar metrics written to scores.json for each family, so recompute.py can check them.
FAMILY_METRICS: dict[str, tuple[str, ...]] = {
    "classification": ("accuracy", "macro_f1", "balanced_accuracy"),
    "classification_proba": ("accuracy", "macro_f1", "balanced_accuracy", "auc", "ece"),
    "forecasting": ("mase",),
    "text": ("gmean_ppl",),
}


def compute(name: str, pu: PerUnit, **kwargs: Any) -> Any:
    """Compute metric ``name`` on one per-unit record."""
    if name not in METRICS:
        raise KeyError(f"unknown metric {name!r}; known: {sorted(METRICS)}")
    return METRICS[name](pu, **kwargs)


def utility(name: str, pu: PerUnit) -> float:
    """Signed utility (higher is better) of one per-unit record."""
    metric, sign = UTILITIES[name]
    return sign * float(compute(metric, pu))


def family_of(pu: PerUnit) -> str:
    """Infer the per-unit family from the stored keys."""
    if "nll" in pu:
        return "text"
    if "last_value" in pu:
        return "forecasting"
    if "proba" in pu:
        return "classification_proba"
    return "classification"


def standard_metrics(pu: PerUnit) -> dict[str, float]:
    """The scalar metrics of the record's family, skipping any that cannot be computed."""
    out: dict[str, float] = {}
    for name in FAMILY_METRICS[family_of(pu)]:
        try:
            out[name] = float(compute(name, pu))
        except (ValueError, IndexError):
            continue
    return out
