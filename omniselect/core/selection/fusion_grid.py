"""Fusion grid cells: weighted channel sum, authenticity prefilter, budget selector.

A cell (w, q, lam) scores record i by sum_c w_c S_c[i] over min-max normalized channels,
removes the records below the q-quantile of the prefilter channel, and selects k records with
BudgetSelector(lam) (plain top-k when lam = 0). ``build_cells`` enumerates the grid in the
canonical order (weights, then q, then lam) and can merge cells with identical subsets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from omniselect.core.selection.budget_select import BudgetSelector
from omniselect.tools.pairing import validate_selection
from omniselect.utils.hashing import sel_sha12

# Channel-weight simplex for three channels (authenticity, influence, redundancy).
W3: list[tuple[float, ...]] = [
    (1, 0, 0), (0, 1, 0), (0, 0, 1), (.5, .5, 0), (.5, 0, .5), (0, .5, .5), (.34, .33, .33)
]

# Parses three- and four-weight cells and the "<cell> solver=infomax" challengers of v2.1.
_CELL_RE = re.compile(r"fuse w=\(([^)]+)\) q=([\d.]+) lam=([\d.]+)")


def default_grid(n_channels: int) -> list[tuple[float, ...]]:
    """W3 for three channels, otherwise the unit vectors plus the uniform weight."""
    if n_channels == 3:
        return list(W3)
    grid = [tuple(1.0 if j == i else 0.0 for j in range(n_channels)) for i in range(n_channels)]
    grid.append(tuple(1.0 / n_channels for _ in range(n_channels)))
    return grid


@dataclass
class FusionGrid:
    """Grid specification. ``weights=None`` means ``default_grid(n_channels)``."""

    weights: list[tuple[float, ...]] | None = None
    q_grid: tuple[float, ...] = (0.0, 0.25)
    lam_grid: tuple[float, ...] = (0.0, 0.25, 0.6)
    prefilter_channel: int = 0
    aliases: dict[str, list[str]] = field(default_factory=dict)


def cell_name(w: Sequence[float], q: float, lam: float) -> str:
    """Name used in leaderboards, identical to the canonical runners' strings."""
    return f"fuse w={tuple(round(x, 2) for x in w)} q={q} lam={lam}"


def parse_cell_name(name: str) -> tuple[np.ndarray, float, float] | None:
    """Inverse of ``cell_name`` for grid cells. None for other names."""
    match = _CELL_RE.match(name)
    if not match:
        return None
    w = np.array([float(x) for x in match.group(1).split(",")], dtype=float)
    return w, float(match.group(2)), float(match.group(3))


def fusion_select(
    S: np.ndarray,
    prefilter: np.ndarray,
    records: Sequence,
    features: np.ndarray,
    w: Sequence[float],
    q: float,
    lam: float,
    k: int,
    similarity=None,
    method: str = "greedy",
) -> list[int]:
    """Exactly k distinct indices chosen by one fusion cell (canonical controller rule).

    ``similarity`` replaces the feature dot product of the diversity term (coverage.space) and
    ``method="infomax"`` solves the relaxed objective instead of the greedy selector.
    """
    n = S.shape[1]
    imp = np.zeros(n)
    for c in range(S.shape[0]):
        imp += float(w[c]) * S[c]
    ungated_imp = imp.copy()
    if q > 0:
        imp = imp.copy()
        imp[prefilter < float(np.quantile(prefilter, q))] = -np.inf
    if lam <= 0:
        selected = [int(i) for i in np.argsort(-imp, kind="stable")[:k]]
        return validate_selection(selected, pool_size=n, expected_size=k, method="controller fusion")
    finite = np.isfinite(imp)
    if finite.all():
        selected = BudgetSelector(lam=lam, method=method).select(records, imp, k, features=features,
                                                                 similarity=similarity)
        return validate_selection(selected, pool_size=n, expected_size=k, method="controller fusion")
    # BudgetSelector min-max normalizes its input, so -inf gated scores would become NaN. The
    # selector runs on the gate-passing records and the indices are mapped back.
    keep = np.where(finite)[0]
    if len(keep) < k:
        remainder = np.where(~finite)[0]
        fill_order = np.argsort(-ungated_imp[remainder], kind="stable")
        selected = [int(i) for i in keep]
        selected.extend(int(i) for i in remainder[fill_order[: k - len(selected)]])
        return validate_selection(selected, pool_size=n, expected_size=k, method="controller fusion")
    if len(keep) == k:
        return validate_selection(keep, pool_size=n, expected_size=k, method="controller fusion")
    sub_recs = [records[int(i)] for i in keep]
    sub_feats = features[keep] if features is not None else None
    sub_sim = similarity.subset(keep) if similarity is not None else None
    sel = BudgetSelector(lam=lam, method=method).select(sub_recs, imp[keep], k, features=sub_feats,
                                                        similarity=sub_sim)
    selected = [int(keep[int(j)]) for j in sel]
    return validate_selection(selected, pool_size=n, expected_size=k, method="controller fusion")


def extend_grid(weights: Sequence[Sequence[float]]) -> list[tuple[float, ...]]:
    """Four-channel grid of protocol v2.1 from a three-channel grid.

    (a, f, c) becomes (a, f, c, 0), then (0, 0, 0, 1), then (a/2, f/2, c/2, 1/2) for each vector,
    then (1/4, 1/4, 1/4, 1/4). Exact duplicates are dropped.
    """
    base = [tuple(float(x) for x in w) for w in weights]
    out = [w + (0.0,) for w in base] + [(0.0, 0.0, 0.0, 1.0)]
    out += [tuple(x / 2 for x in w) + (0.5,) for w in base] + [(0.25, 0.25, 0.25, 0.25)]
    return list(dict.fromkeys(out))


def build_cells(
    S: np.ndarray,
    records: Sequence,
    features: np.ndarray,
    k: int,
    grid: FusionGrid,
    dedupe: bool = False,
    similarity=None,
) -> list[tuple[str, list[int]]]:
    """All grid cells as (name, selection) in canonical order.

    With ``dedupe`` a cell whose sorted subset equals an earlier cell's is dropped and its name is
    recorded in ``grid.aliases[first_name]``.
    """
    weights = grid.weights if grid.weights is not None else default_grid(S.shape[0])
    prefilter = S[grid.prefilter_channel]
    cells: list[tuple[str, list[int]]] = []
    first_by_sha: dict[str, str] = {}
    grid.aliases = {}
    for w in weights:
        for q in grid.q_grid:
            for lam in grid.lam_grid:
                name = cell_name(w, q, lam)
                sel = fusion_select(S, prefilter, records, features, w, q, lam, k, similarity=similarity)
                if dedupe:
                    sha = sel_sha12(sel)
                    if sha in first_by_sha:
                        grid.aliases.setdefault(first_by_sha[sha], []).append(name)
                        continue
                    first_by_sha[sha] = name
                cells.append((name, sel))
    return cells
