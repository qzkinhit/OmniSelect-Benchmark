"""Learned cleanliness: a positive-unlabelled classifier of V_con records (clean) against pool records.

Per-record features are computed the same way for pool and V_con records, always against the pool
(a pool record excludes itself) by ``cleanliness_features`` per track family. ``learned_cleanliness``
fits a gradient-boosted classifier of V_con (1) against the pool (0) with class-balanced weights and
returns cross-fitted pool scores in [0, 1] and the cross-validated AUC of V_con against the pool.
It reads the pool and V_con only.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from omniselect.config.config import CleanlinessConfig
from omniselect.core.selection.cooperative import NeighbourGraph
from omniselect.tools.pairing import stable_seed

EPS = 1e-6


def _distance_features(graph: NeighbourGraph) -> np.ndarray:
    """log distance to the nearest and to the last (10th) neighbour."""
    return np.c_[np.log(graph.distance[:, 0] + EPS), np.log(graph.distance[:, -1] + EPS)]


def _folds(n: int, k: int, seed: int, labels: Optional[np.ndarray] = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shuffled k folds (stratified by ``labels`` when every class has at least k records)."""
    from sklearn.model_selection import KFold, StratifiedKFold

    k = max(2, min(int(k), n))
    if labels is not None and np.unique(labels, return_counts=True)[1].min() >= k:
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(n), labels))
    return list(KFold(k, shuffle=True, random_state=seed).split(np.zeros(n)))


def classification_features(graph: NeighbourGraph, con_graph: NeighbourGraph, pool_X: np.ndarray,
                            pool_y: np.ndarray, con_X: np.ndarray, con_y: np.ndarray, seed: int,
                            cfg: CleanlinessConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """kNN label agreement, cross-fitted p(observed label) and margin, log nearest and 10th distances.

    p(observed label) and the margin (p(observed) minus the largest other probability) come from
    LogisticRegression(C=1) fitted on ``cfg.folds`` pool folds: out of fold for pool records, the
    mean of the fold models for V_con records. A label absent from a fold model has probability 0.
    """
    from sklearn.linear_model import LogisticRegression

    pool_y, con_y = np.asarray(pool_y), np.asarray(con_y)
    classes = np.unique(np.r_[pool_y, con_y])
    yp, yc = np.searchsorted(classes, pool_y), np.searchsorted(classes, con_y)
    Xp = np.asarray(pool_X, dtype=np.float64).reshape(len(pool_y), -1)
    Xc = np.asarray(con_X, dtype=np.float64).reshape(len(con_y), -1)
    Pp = np.zeros((len(yp), len(classes)))
    Pc = np.zeros((len(yc), len(classes)))
    folds = _folds(len(yp), cfg.folds, stable_seed(seed, "pu"), yp)
    for tr, te in folds:
        model = LogisticRegression(C=1.0, max_iter=cfg.logreg_iter).fit(Xp[tr], yp[tr])
        Pp[np.ix_(te, model.classes_)] = model.predict_proba(Xp[te])
        Pc[:, model.classes_] += model.predict_proba(Xc) / len(folds)

    def likelihood(P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """p(observed label) and its margin over the largest other class probability."""
        rows = np.arange(len(y))
        observed = P[rows, y]
        other = P.copy()
        other[rows, y] = -np.inf
        return observed, observed - other.max(axis=1)

    agree_p = (yp[graph.index] == yp[:, None]).mean(axis=1)
    agree_c = (yp[con_graph.index] == yc[:, None]).mean(axis=1)
    Fp = np.c_[agree_p, np.c_[likelihood(Pp, yp)], _distance_features(graph)]
    Fc = np.c_[agree_c, np.c_[likelihood(Pc, yc)], _distance_features(con_graph)]
    names = ["knn_label_agreement", "p_observed_label", "label_margin", "log_nn1_distance", "log_nn10_distance"]
    return Fp, Fc, names


def _lag1(X: np.ndarray) -> np.ndarray:
    """Lag-1 autocorrelation of each row (0 for a constant row)."""
    Z = X - X.mean(axis=1, keepdims=True)
    den = (Z[:, :-1] ** 2).sum(axis=1)
    num = (Z[:, :-1] * Z[:, 1:]).sum(axis=1)
    return np.where(den > 0, num / (den + 1e-9), 0.0)


def forecasting_features(graph: NeighbourGraph, con_graph: NeighbourGraph, Xp: np.ndarray, Yp: np.ndarray,
                         Xc: np.ndarray, Yc: np.ndarray, seed: int, cfg: CleanlinessConfig
                         ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Cross-fitted ridge residual, log nearest and 10th distances, window standard deviation, lag-1 autocorrelation.

    The residual is the mean absolute error of Ridge(alpha) from the input window to the target,
    out of fold on the pool and averaged over the fold models on V_con, divided by the pool median.
    """
    from sklearn.linear_model import Ridge

    Xp, Yp = np.asarray(Xp, dtype=np.float64), np.asarray(Yp, dtype=np.float64)
    Xc, Yc = np.asarray(Xc, dtype=np.float64), np.asarray(Yc, dtype=np.float64)
    pred_p = np.zeros_like(Yp)
    pred_c = np.zeros_like(Yc)
    folds = _folds(len(Xp), cfg.folds, stable_seed(seed, "pu"))
    for tr, te in folds:
        model = Ridge(alpha=cfg.ridge_alpha).fit(Xp[tr], Yp[tr])
        pred_p[te] = model.predict(Xp[te])
        pred_c += model.predict(Xc) / len(folds)
    res_p = np.abs(Yp - pred_p).mean(axis=1)
    res_c = np.abs(Yc - pred_c).mean(axis=1)
    scale = float(np.median(res_p)) + 1e-12
    Fp = np.c_[res_p / scale, _distance_features(graph), Xp.std(axis=1), _lag1(Xp)]
    Fc = np.c_[res_c / scale, _distance_features(con_graph), Xc.std(axis=1), _lag1(Xc)]
    return Fp, Fc, ["ridge_residual", "log_nn1_distance", "log_nn10_distance", "window_std", "lag1_autocorr"]


def text_features(graph: NeighbourGraph, con_graph: NeighbourGraph, pool: dict[str, np.ndarray],
                  con: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Text authenticity, compression ratio, length in tokens, log nearest and 10th distances.

    ``pool`` and ``con`` map authenticity, compression_ratio and length_tokens to per-record arrays.
    """
    keys = ("authenticity", "compression_ratio", "length_tokens")
    Fp = np.c_[np.column_stack([np.asarray(pool[k], dtype=float) for k in keys]), _distance_features(graph)]
    Fc = np.c_[np.column_stack([np.asarray(con[k], dtype=float) for k in keys]), _distance_features(con_graph)]
    return Fp, Fc, [*keys, "log_nn1_distance", "log_nn10_distance"]


def _classifier(seed: int, cfg: CleanlinessConfig):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=cfg.max_iter, learning_rate=cfg.learning_rate,
                                          max_leaf_nodes=cfg.max_leaf_nodes, random_state=stable_seed(seed, "pu"))


def _balanced_fit(Fpool: np.ndarray, Fcon: np.ndarray, seed: int, cfg: CleanlinessConfig):
    """Classifier of V_con (1) against pool (0), each class carrying half of the total weight."""
    X = np.r_[Fpool, Fcon]
    s = np.r_[np.zeros(len(Fpool)), np.ones(len(Fcon))]
    w = np.r_[np.full(len(Fpool), 0.5 / len(Fpool)), np.full(len(Fcon), 0.5 / len(Fcon))] * len(s)
    return _classifier(seed, cfg).fit(X, s, sample_weight=w)


def learned_cleanliness(Fp: np.ndarray, Fc: np.ndarray, seed: int, cfg: CleanlinessConfig
                        ) -> tuple[np.ndarray, Optional[float]]:
    """Cross-fitted pool scores P(V_con | features) in [0, 1] and the cross-validated AUC of V_con against the pool.

    Pool scores: for each of ``cfg.folds`` pool folds the classifier is trained on the other pool
    folds and every V_con record. AUC: V_con and the pool are both split into folds, each fold is
    scored by a classifier trained on the other folds, and the AUC is taken over all held-out scores
    (None with fewer V_con records than folds).
    """
    from sklearn.metrics import roc_auc_score

    Fp, Fc = np.asarray(Fp, dtype=np.float64), np.asarray(Fc, dtype=np.float64)
    rs = stable_seed(seed, "pu")
    score = np.zeros(len(Fp))
    for tr, te in _folds(len(Fp), cfg.folds, rs):
        score[te] = _balanced_fit(Fp[tr], Fc, seed, cfg).predict_proba(Fp[te])[:, 1]
    auc = None
    if len(Fc) >= cfg.folds:
        pool_folds = _folds(len(Fp), cfg.folds, rs)
        con_folds = _folds(len(Fc), cfg.folds, rs)
        held_p, held_c = np.zeros(len(Fp)), np.zeros(len(Fc))
        for (ptr, pte), (ctr, cte) in zip(pool_folds, con_folds):
            clf = _balanced_fit(Fp[ptr], Fc[ctr], seed, cfg)
            held_p[pte] = clf.predict_proba(Fp[pte])[:, 1]
            held_c[cte] = clf.predict_proba(Fc[cte])[:, 1]
        auc = float(roc_auc_score(np.r_[np.zeros(len(Fp)), np.ones(len(Fc))], np.r_[held_p, held_c]))
    return score, auc


def cleanliness_features(inputs: dict, graph: NeighbourGraph, con_graph: NeighbourGraph, seed: int,
                         cfg: CleanlinessConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Features of the track family named by ``inputs["kind"]`` (classification, forecasting or text)."""
    kind = inputs["kind"]
    if kind == "classification":
        return classification_features(graph, con_graph, inputs["pool_X"], inputs["pool_y"], inputs["con_X"],
                                       inputs["con_y"], seed, cfg)
    if kind == "forecasting":
        return forecasting_features(graph, con_graph, inputs["Xp"], inputs["Yp"], inputs["Xc"], inputs["Yc"], seed,
                                    cfg)
    if kind == "text":
        return text_features(graph, con_graph, inputs["pool"], inputs["con"])
    raise ValueError(f"unknown cleanliness feature kind {kind!r}")
