"""Conservative text transfers for representation-based baselines.

These names deliberately end in ``_text``. They preserve each baseline's
selection rule while replacing image/classification embeddings with frozen
autoregressive-LM representations. Classification-error and gradient methods
(EL2N, GraNd, and CCS) are intentionally not transferred here.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


TEXT_TRANSFER_PROTOCOLS = {
    "coverage_text": "farthest-first coverage on frozen LM mean-pooled hidden states",
    "herding_text": "moment-matching herding on frozen LM mean-pooled hidden states",
    "density_text": "inverse-density sampling on frozen LM mean-pooled hidden states",
}


def complete_order(prefix: Sequence[int], n: int) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for raw in prefix:
        index = int(raw)
        if not 0 <= index < n:
            raise ValueError(f"selection index out of range: {index}")
        if index not in seen:
            seen.add(index)
            out.append(index)
    out.extend(index for index in range(n) if index not in seen)
    if len(out) != n or len(set(out)) != n:
        raise RuntimeError("failed to construct a complete unique ordering")
    return out


def _domain_indices(domains: Sequence[str]) -> list[tuple[str, np.ndarray]]:
    values = np.asarray(domains)
    return [(str(domain), np.where(values == domain)[0]) for domain in sorted(set(values))]


def coverage_token_order(
    features: np.ndarray,
    domains: Sequence[str],
    token_counts: Sequence[int],
    domain_budgets: Mapping[str, int],
) -> list[int]:
    """Farthest-first coverage prefix per domain under the token budget."""
    X = np.asarray(features, dtype=np.float32)
    tokens = np.asarray(token_counts, dtype=np.int64)
    prefix: list[int] = []
    for domain, indices in _domain_indices(domains):
        local = X[indices]
        chosen = np.zeros(len(indices), dtype=bool)
        centroid = local.mean(axis=0)
        first = int(np.argmax(local @ centroid))
        chosen[first] = True
        global_first = int(indices[first])
        prefix.append(global_first)
        used = int(tokens[global_first])
        max_similarity = local @ local[first]
        while used < int(domain_budgets[domain]) and not chosen.all():
            farthest = max_similarity.copy()
            farthest[chosen] = np.inf
            local_index = int(np.argmin(farthest))
            chosen[local_index] = True
            global_index = int(indices[local_index])
            prefix.append(global_index)
            used += int(tokens[global_index])
            max_similarity = np.maximum(max_similarity, local @ local[local_index])
    return complete_order(prefix, len(X))


def herding_token_order(
    features: np.ndarray,
    domains: Sequence[str],
    token_counts: Sequence[int],
    domain_budgets: Mapping[str, int],
) -> list[int]:
    """Exact moment-matching herding prefix per domain under the token budget."""
    X = np.asarray(features, dtype=np.float32)
    tokens = np.asarray(token_counts, dtype=np.int64)
    prefix: list[int] = []
    for domain, indices in _domain_indices(domains):
        local = X[indices]
        target = local.mean(axis=0)
        squared_norm = np.einsum("ij,ij->i", local, local)
        running = np.zeros_like(target)
        chosen = np.zeros(len(indices), dtype=bool)
        used = 0
        step = 0
        while used < int(domain_budgets[domain]) and not chosen.all():
            desired = float(step + 1) * target - running
            # The omitted ||desired||² term and common denominator do not affect
            # the argmin, avoiding one (n,d) temporary allocation per step.
            distance = squared_norm - 2.0 * (local @ desired)
            distance[chosen] = np.inf
            local_index = int(np.argmin(distance))
            chosen[local_index] = True
            running += local[local_index]
            global_index = int(indices[local_index])
            prefix.append(global_index)
            used += int(tokens[global_index])
            step += 1
    return complete_order(prefix, len(X))
