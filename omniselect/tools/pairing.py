"""Paired-run helpers shared by all tracks: selection fingerprints, exact k-means medoids, RNG resets.

``sel_sha12`` hashes the sorted ids of a selection and ``order_sha12`` the ids in training order.
``reset_rng(*parts)`` seeds Python, numpy and torch with crc32 of the parts, called before each
strategy with (seed, 'select', name) and before fits with (seed, stage). ``validate_selection``
rejects duplicates, out-of-range ids and wrong sizes. ``exact_kmeans_representatives`` returns k medoids.
"""
import hashlib
import operator
import random
import zlib

import numpy as np


def exact_kmeans_representatives(features, labels, centers, expected_size):
    """Return an exact-size deterministic medoid set from fitted k-means state.

    The first pass preserves the usual coreset rule: choose the sample nearest
    each non-empty cluster center. Mini-batch k-means can leave centers empty,
    however, so that pass may return fewer rows than the requested budget. In
    that case the remaining rows are ordered by distance to their assigned
    center (then by row ID for stable tie-breaking) and used to fill the
    deficit. The result is unique and always has ``expected_size`` entries.
    """
    matrix = np.asarray(features)
    assignments = np.asarray(labels)
    centroids = np.asarray(centers)
    expected_size = operator.index(expected_size)

    if matrix.ndim != 2:
        raise ValueError(f"features must be a 2-D matrix, got shape {matrix.shape}")
    if assignments.shape != (len(matrix),):
        raise ValueError(
            "labels must contain one cluster assignment per feature row, "
            f"got shape {assignments.shape} for {len(matrix)} rows"
        )
    if centroids.ndim != 2 or centroids.shape[1:] != matrix.shape[1:]:
        raise ValueError(
            "centers must have the same feature width as features, "
            f"got {centroids.shape} and {matrix.shape}"
        )
    if not 0 <= expected_size <= len(matrix):
        raise ValueError(
            f"expected_size must be in [0, {len(matrix)}], got {expected_size}"
        )
    if expected_size == 0:
        return []
    if len(centroids) != expected_size:
        raise ValueError(
            "the fitted center count must equal the requested budget, "
            f"got {len(centroids)} centers for expected_size={expected_size}"
        )
    if not np.isfinite(matrix).all() or not np.isfinite(centroids).all():
        raise ValueError("features and centers must contain only finite values")
    if not np.issubdtype(assignments.dtype, np.integer):
        raise TypeError("cluster labels must be integers")
    if np.any(assignments < 0) or np.any(assignments >= len(centroids)):
        raise ValueError("cluster labels contain an out-of-range center ID")

    row_ids = np.arange(len(matrix), dtype=np.int64)
    assigned_distance = np.linalg.norm(
        matrix - centroids[assignments], axis=1
    )
    selected: list[int] = []
    selected_set: set[int] = set()

    # Stable row-ID tie breaking makes the selection invariant to NumPy's
    # implementation-specific ordering of equal distances.
    for cluster_id in range(len(centroids)):
        members = row_ids[assignments == cluster_id]
        if not len(members):
            continue
        member_order = np.lexsort((members, assigned_distance[members]))
        representative = int(members[member_order[0]])
        selected.append(representative)
        selected_set.add(representative)

    if len(selected) < expected_size:
        remaining = row_ids[
            np.fromiter(
                (int(row_id) not in selected_set for row_id in row_ids),
                dtype=bool,
                count=len(row_ids),
            )
        ]
        fill_order = np.lexsort((remaining, assigned_distance[remaining]))
        selected.extend(
            int(row_id)
            for row_id in remaining[fill_order[: expected_size - len(selected)]]
        )

    if len(selected) != expected_size or len(set(selected)) != expected_size:
        raise RuntimeError("k-means representative completion failed its exact budget")
    return selected


def validate_selection(sel, *, pool_size, expected_size, method):
    """Validate and normalize a selector output before downstream training.

    Every budget-matched method must return exactly ``expected_size`` distinct,
    in-range integer row IDs.  Failing here prevents an invalid selector from
    receiving a different data budget or silently relying on NumPy's negative
    indexing semantics.
    """
    pool_size = operator.index(pool_size)
    expected_size = operator.index(expected_size)
    if pool_size < 0:
        raise ValueError(f"{method}: pool_size must be non-negative, got {pool_size}")
    if not 0 <= expected_size <= pool_size:
        raise ValueError(
            f"{method}: expected_size must be in [0, {pool_size}], got {expected_size}"
        )

    try:
        values = list(sel)
    except TypeError as exc:
        raise TypeError(f"{method}: selector output must be an iterable of integer IDs") from exc

    normalized = []
    seen = set()
    for position, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{method}: selection ID at position {position} is boolean")
        try:
            index = operator.index(value)
        except TypeError as exc:
            raise TypeError(
                f"{method}: selection ID at position {position} is not an integer: {value!r}"
            ) from exc
        if not 0 <= index < pool_size:
            raise ValueError(
                f"{method}: selection ID {index} at position {position} is outside "
                f"[0, {pool_size})"
            )
        if index in seen:
            raise ValueError(f"{method}: duplicate selection ID {index}")
        seen.add(index)
        normalized.append(index)

    if len(normalized) != expected_size:
        raise ValueError(
            f"{method}: selected {len(normalized)} rows, expected exactly {expected_size}"
        )
    return normalized


def sel_sha12(sel):
    return hashlib.sha256(str(sorted(int(i) for i in sel)).encode()).hexdigest()[:12]


def order_sha12(sel):
    return hashlib.sha256(str([int(i) for i in sel]).encode()).hexdigest()[:12]


def arrays_sha256(*arrays):
    """Stable fingerprint for the exact numeric inputs used by a paired trial."""
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(np.asarray(value))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def stable_seed(*parts):
    s = zlib.crc32("|".join(str(p) for p in parts).encode()) % (2 ** 31 - 1)
    return s


def reset_rng(*parts):
    s = stable_seed(*parts)
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
    except Exception:
        pass
    return s
