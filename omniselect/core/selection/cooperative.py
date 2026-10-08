"""Cooperative candidates of protocol v2.2: a cleanliness gate, two geometric vetoes, a selector inside.

``gate`` keeps the ceil(rho k) records with the highest per-record cleanliness score (ties by pool
index, all records when rho is None). ``neighbour_graph`` finds nearest pool neighbours in the
selector's feature space. ``outlier_flags`` and ``duplicate_groups`` turn it into the two vetoes,
``apply_vetoes`` and ``refill`` build the admissible set, and herding or k-center selects inside it.
``CooperativeFamily`` binds one cell's scores and geometry and measures each selection.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

SELECTORS = ("herding", "kcenter")
METRICS = ("cosine", "euclidean")
# Gate scores. Authenticity reads the pool only, cleanliness (core/signals/cleanliness.py) also reads V_con.
GATE_SCORES = ("authenticity", "cleanliness")
POOL_ONLY_SCORES = ("authenticity",)


def parse_rho(value: Any) -> Optional[float]:
    """Keep ratio of a grid entry: None for 'none' (no gate), otherwise a float of at least 1."""
    if value is None or (isinstance(value, str) and value.strip().lower() == "none"):
        return None
    rho = float(value)
    if not math.isfinite(rho) or rho < 1.0:
        raise ValueError(f"keep ratio must be 'none' or a number >= 1, got {value!r}")
    return rho


def cell_name(score: Optional[str], rho: Optional[float], selector: str) -> str:
    """Candidate name of one grid cell: 'coop gate=<score> rho=<rho> within=<selector>', or
    'coop rho=none within=<selector>' for the ungated cells (vetoes only)."""
    if rho is None:
        return f"coop rho=none within={selector}"
    return f"coop gate={score} rho={float(rho)} within={selector}"


def gate(score: np.ndarray, k: int, rho: Optional[float]) -> np.ndarray:
    """Pool indices of the ceil(rho k) highest scores, highest first, ties by pool index.

    ``score`` is any per-record cleanliness score (larger is cleaner). rho None keeps every record.
    The kept count is capped at the pool size.
    """
    s = np.asarray(score, dtype=float)
    order = np.argsort(-s, kind="stable")
    if rho is None:
        return order
    size = min(len(s), int(math.ceil(round(float(rho) * int(k), 9))))
    return order[:size]


@dataclass
class NeighbourGraph:
    """The ``knn`` nearest pool records of every query row, nearest first (ties by pool index)."""

    index: np.ndarray        # (rows, knn) int64 pool indices
    distance: np.ndarray     # (rows, knn) float64
    metric: str
    seconds: float = 0.0


def neighbour_graph(features: np.ndarray, metric: str, knn: int = 10, query: Optional[np.ndarray] = None,
                    block: int = 2 ** 24) -> NeighbourGraph:
    """Nearest pool neighbours under cosine distance of L2-normalized rows or Euclidean distance.

    Without ``query`` the rows are the pool records and a record's own entry is excluded. With
    ``query`` (rows in the same representation) every query row gets its nearest pool records.
    Distances are computed in row chunks of at most ``block`` entries, so a pool of 50,000 records
    needs a few hundred megabytes. Among records at the same distance the lower pool index comes first.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    t0 = time.perf_counter()

    def prepare(A: np.ndarray) -> np.ndarray:
        """Float64 rows, L2-normalized for the cosine metric."""
        A = np.asarray(A, dtype=np.float64)
        A = A.reshape(A.shape[0], -1)
        return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12) if metric == "cosine" else A

    P = prepare(features)
    Q = P if query is None else prepare(query)
    n = P.shape[0]
    if n < 2:
        raise ValueError("the neighbour graph needs at least two pool records")
    m = max(1, min(int(knn), n - 1 if query is None else n))
    sq_p = np.einsum("ij,ij->i", P, P)
    sq_q = sq_p if query is None else np.einsum("ij,ij->i", Q, Q)
    rows = max(1, min(Q.shape[0], block // n))
    index = np.empty((Q.shape[0], m), dtype=np.int64)
    distance = np.empty((Q.shape[0], m), dtype=np.float64)
    for start in range(0, Q.shape[0], rows):
        gram = Q[start:start + rows] @ P.T
        if metric == "cosine":
            D = np.clip(1.0 - gram, 0.0, None)
        else:
            D = np.sqrt(np.clip(sq_q[start:start + rows, None] + sq_p[None, :] - 2.0 * gram, 0.0, None))
        local = np.arange(D.shape[0])
        if query is None:
            D[local, start + local] = np.inf
        kth = np.partition(D, m - 1, axis=1)[:, m - 1]
        for r in local:
            cand = np.flatnonzero(D[r] <= kth[r])
            cand = cand[np.argsort(D[r, cand], kind="stable")][:m]
            index[start + r] = cand
            distance[start + r] = D[r, cand]
    return NeighbourGraph(index, distance, metric, time.perf_counter() - t0)


def outlier_flags(graph: NeighbourGraph, z_max: float = 3.0) -> tuple[np.ndarray, dict[str, float]]:
    """Records whose robust z-score of the distance to the last (10th) neighbour exceeds ``z_max``.

    z = (d - median d) / (1.4826 MAD) over the pool. A zero MAD flags no record.
    """
    d = graph.distance[:, -1]
    med = float(np.median(d))
    scale = 1.4826 * float(np.median(np.abs(d - med)))
    if scale <= 0.0:
        return np.zeros(len(d), dtype=bool), {"median": med, "scale": scale}
    return (d - med) / scale > float(z_max), {"median": med, "scale": scale}


def duplicate_groups(graph: NeighbourGraph, ratio: float = 0.1) -> tuple[np.ndarray, dict[str, float]]:
    """Connected-component label of every record under the near-duplicate relation.

    Two records are linked when one is among the other's neighbours at a distance below ``ratio``
    times the median nearest-neighbour distance of the pool. Records without a link form singleton
    components.
    """
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    n, m = graph.index.shape
    threshold = float(ratio) * float(np.median(graph.distance[:, 0]))
    linked = graph.distance < threshold
    rows = np.repeat(np.arange(n), m)[linked.ravel()]
    cols = graph.index[linked]
    adjacency = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(adjacency, directed=False)
    return labels.astype(np.int64), {"threshold": threshold}


def apply_vetoes(admissible: np.ndarray, outliers: Optional[np.ndarray], groups: Optional[np.ndarray]
                 ) -> tuple[list[int], list[int], list[int]]:
    """Walk ``admissible`` in its order (cleanest first): drop outliers, then every member of a
    near-duplicate group after its first surviving member. Returns (kept, outlier drops, duplicate
    drops), each in admissible order. None disables a veto."""
    sizes = np.bincount(groups) if groups is not None else None
    seen: set[int] = set()
    kept: list[int] = []
    out_drop: list[int] = []
    dup_drop: list[int] = []
    for raw in admissible:
        i = int(raw)
        if outliers is not None and outliers[i]:
            out_drop.append(i)
            continue
        if groups is not None and sizes[groups[i]] > 1:
            if int(groups[i]) in seen:
                dup_drop.append(i)
                continue
            seen.add(int(groups[i]))
        kept.append(i)
    return kept, out_drop, dup_drop


def refill(kept: list[int], dropped: Sequence[int], k: int) -> tuple[list[int], int]:
    """``kept`` extended by the first ``dropped`` records (cleanest first) until it holds k records."""
    need = max(0, int(k) - len(kept))
    extra = [int(i) for i in list(dropped)[:need]]
    return list(kept) + extra, len(extra)


def herding_order(features: np.ndarray, k: int) -> list[int]:
    """Herding with the rule of benchmark/Methods/Herding: step t adds the unchosen row minimizing
    ||(running sum + x_i) / t - mean||, computed as ||x_i||^2 - 2 x_i (t mean - running sum)."""
    X = np.asarray(features, dtype=np.float64)
    X = X.reshape(X.shape[0], -1)
    n = X.shape[0]
    k = min(int(k), n)
    target = X.mean(axis=0)
    sq = np.einsum("ij,ij->i", X, X)
    running = np.zeros_like(target)
    chosen = np.zeros(n, dtype=bool)
    out: list[int] = []
    for t in range(k):
        d = sq - 2.0 * (X @ ((t + 1) * target - running))
        d[chosen] = np.inf
        i = int(np.argmin(d))
        out.append(i)
        chosen[i] = True
        running += X[i]
    return out


def kcenter_order(features: np.ndarray, k: int, seed: int) -> list[int]:
    """Farthest-first traversal with the rule of benchmark/Methods/KCenter: the first row drawn by
    default_rng(seed), then the row farthest (Euclidean) from the chosen rows, first index on ties."""
    X = np.asarray(features, dtype=np.float64)
    X = X.reshape(X.shape[0], -1)
    n = X.shape[0]
    k = min(int(k), n)
    if k == 0:
        return []
    sq = np.einsum("ij,ij->i", X, X)
    start = int(np.random.default_rng(seed).integers(n))
    out = [start]
    available = np.ones(n, dtype=bool)
    available[start] = False
    dist = np.clip(sq + sq[start] - 2.0 * (X @ X[start]), 0.0, None)
    for _ in range(1, k):
        i = int(np.argmax(np.where(available, dist, -np.inf)))
        out.append(i)
        available[i] = False
        dist = np.minimum(dist, np.clip(sq + sq[i] - 2.0 * (X @ X[i]), 0.0, None))
    return out


def select_within(admissible: Sequence[int], k: int, selector: str, features: np.ndarray, seed: int) -> list[int]:
    """k pool indices chosen by herding or k-center (start drawn by default_rng(seed)) on the rows of
    ``features`` of the admissible records."""
    adm = np.asarray(admissible, dtype=np.int64)
    X = np.asarray(features)[adm]
    if selector == "herding":
        local = herding_order(X, k)
    elif selector == "kcenter":
        local = kcenter_order(X, k, seed)
    else:
        raise ValueError(f"selector must be one of {SELECTORS}, got {selector!r}")
    return [int(adm[j]) for j in local]


@dataclass
class CooperativeFamily:
    """One cell's inputs of the cooperative candidates.

    scores: gate scores by name (``GATE_SCORES``). graph: the pool neighbour graph of the veto
    geometry. selector_features: herding and k-center. outlier_z or duplicate_ratio None disables a
    veto. Token-budgeted tracks give ``budget_gate(order, rho)`` (the records of ``order`` inside rho
    times the budget), ``budget_cut(order)`` (the budgeted selection of a full order) and
    ``token_orders`` (selector name to a function ordering a subset of pool indices). Their
    selections are full orders.
    """

    scores: dict[str, np.ndarray]
    graph: NeighbourGraph
    selector_features: np.ndarray
    seed: int
    reference_rho: float = 1.3
    outlier_z: Optional[float] = 3.0
    duplicate_ratio: Optional[float] = 0.1
    budget_gate: Optional[Callable[[list[int], float], list[int]]] = None
    budget_cut: Optional[Callable[[list[int]], list[int]]] = None
    token_orders: dict[str, Callable[[np.ndarray], list[int]]] = field(default_factory=dict)
    admissible_sets: dict[str, np.ndarray] = field(default_factory=dict)
    _vetoes: Optional[tuple] = field(default=None, repr=False)

    def vetoes(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict[str, Any]]:
        """Outlier flags, duplicate group labels and their pool-level statistics (computed once)."""
        if self._vetoes is None:
            stats: dict[str, Any] = {"metric": self.graph.metric, "knn": int(self.graph.index.shape[1])}
            flags = labels = None
            if self.outlier_z is not None:
                flags, info = outlier_flags(self.graph, self.outlier_z)
                stats.update({"outlier_z": float(self.outlier_z), "outlier_median": info["median"],
                              "outlier_scale": info["scale"], "pool_outliers": int(flags.sum())})
            if self.duplicate_ratio is not None:
                labels, info = duplicate_groups(self.graph, self.duplicate_ratio)
                sizes = np.bincount(labels)
                stats.update({"duplicate_ratio": float(self.duplicate_ratio),
                              "duplicate_threshold": info["threshold"],
                              "pool_duplicate_groups": int((sizes > 1).sum()),
                              "pool_duplicate_records": int(sizes[sizes > 1].sum())})
            self._vetoes = (flags, labels, stats)
        return self._vetoes

    def select(self, score_name: Optional[str], rho: Optional[float], selector: str, k: int, name: str
               ) -> tuple[list[int], dict[str, Any]]:
        """Selection of one cell and its measurements (``selection_info`` of the run record).

        ``score_name`` picks the gate score (unused when rho is None). Count budgets return k pool
        indices. Token budgets return a full order: the selector's order of the admissible set, then
        the vetoed records and the records outside the gate, each in score order, so that the budget
        cut refills from the vetoed records first.
        """
        flags, labels, stats = self.vetoes()
        t0 = time.perf_counter()
        score = np.asarray(self.scores[score_name if rho is not None else "authenticity"], dtype=float)
        n = len(score)
        tokens = self.budget_gate is not None
        order = np.argsort(-score, kind="stable")
        if tokens:
            inside = None if rho is None else set(int(i) for i in self.budget_gate([int(i) for i in order], rho))
            admissible = order if inside is None else np.asarray([i for i in order if int(i) in inside], dtype=np.int64)
        else:
            admissible = gate(score, k, rho)
        kept, out_drop, dup_drop = apply_vetoes(admissible, flags, labels)
        vetoed = set(out_drop) | set(dup_drop)
        dropped = [int(i) for i in admissible if int(i) in vetoed]
        if tokens:
            within = [int(i) for i in self.token_orders[selector](np.asarray(kept, dtype=np.int64))]
            placed = set(kept) | vetoed
            selection = within + dropped + [int(i) for i in order if int(i) not in placed]
            kept_set = set(kept)
            refilled = sum(1 for i in self.budget_cut(selection) if int(i) not in kept_set)
        else:
            kept, refilled = refill(kept, dropped, k)
            selection = select_within(kept, k, selector, self.selector_features, self.seed)
        self.admissible_sets[name] = np.asarray(kept, dtype=np.int64)
        info = {
            "rule": "cooperative", "gate_score": score_name if rho is not None else None, "rho": rho,
            "selector": selector, "pool": int(n), "admissible_gate": int(len(admissible)),
            "outlier_drops": len(out_drop), "duplicate_drops": len(dup_drop), "refill": int(refilled),
            "admissible_size": len(kept), **stats, "seconds": time.perf_counter() - t0,
            "neighbour_secs": self.graph.seconds,
        }
        return selection, info
