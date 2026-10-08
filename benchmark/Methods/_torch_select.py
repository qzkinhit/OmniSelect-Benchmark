"""Torch versions of herding, k-center greedy and k-means coverage for large pools (``selection_device``).

``herding_torch`` and ``kcenter_torch`` apply the rules of benchmark/Methods/Herding and KCenter
step for step: the same candidate distances (direct norms of the difference vectors, computed in
row chunks), the same start record, and ties resolved by the first index as numpy's argmin and
argmax do. ``kmeans_torch`` runs Lloyd's k-means with the k-means++ initialization of
scikit-learn (the same RandomState draws, ``2 + log(k)`` local trials, shared across ``n_init``
runs), scikit-learn's tolerance rule and inertia choice, and relocates empty clusters to the
records farthest from their centres. Computation runs in ``dtype`` (float32 by default) on
``device`` with TF32 disabled (sums of the k-means++ potential in float64, float32 on MPS), and the
results return to the CPU as Python lists or numpy arrays.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np

CHUNK_ROWS = 32768


@contextmanager
def no_tf32() -> Iterator[None]:
    """Disable TF32 in cuBLAS and cuDNN and request full float32 matmul precision, restored on exit."""
    import torch

    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[:2]
        torch.set_float32_matmul_precision(saved[2])


def _tensor(features, device: str, dtype):
    import torch

    X = np.asarray(features, dtype=np.float64)
    X = X.reshape(X.shape[0], -1)
    return torch.as_tensor(X, dtype=dtype, device=device)


def _row_norms(X, point, chunk: int = CHUNK_ROWS):
    """||X_i - point|| for every row, computed in row chunks."""
    import torch

    out = torch.empty(X.shape[0], dtype=X.dtype, device=X.device)
    for s in range(0, X.shape[0], chunk):
        out[s:s + chunk] = torch.linalg.vector_norm(X[s:s + chunk] - point, dim=1)
    return out


def herding_torch(features, k: int, device: str = "cpu", dtype=None) -> list[int]:
    """Herding with the rule of benchmark/Methods/Herding/method.py (argmin of ||(running + x_i) / t - mean||)."""
    import torch

    dtype = dtype or torch.float32
    with no_tf32(), torch.no_grad():
        X = _tensor(features, device, dtype)
        n = X.shape[0]
        k = min(int(k), n)
        target = X.mean(dim=0)
        running = torch.zeros_like(target)
        chosen = torch.empty(k, dtype=torch.long, device=device)
        inf = torch.tensor(float("inf"), dtype=dtype, device=device)
        mask = torch.ones(n, dtype=torch.bool, device=device)
        d = torch.empty(n, dtype=dtype, device=device)
        for t in range(k):                   # indices stay on the device, no host sync per step
            for s in range(0, n, CHUNK_ROWS):
                cand = (running + X[s:s + CHUNK_ROWS]) / (t + 1)
                d[s:s + CHUNK_ROWS] = torch.linalg.vector_norm(cand - target, dim=1)
            i = torch.argmin(torch.where(mask, d, inf)).view(1)
            chosen[t:t + 1] = i
            running = running + X.index_select(0, i)[0]
            mask.index_fill_(0, i, False)
    return [int(v) for v in chosen.cpu().tolist()]


def kcenter_torch(features, k: int, seed: int = 0, device: str = "cpu", dtype=None) -> list[int]:
    """k-center greedy with the rule of benchmark/Methods/KCenter/method.py (start from default_rng(seed))."""
    import torch

    dtype = dtype or torch.float32
    with no_tf32(), torch.no_grad():
        X = _tensor(features, device, dtype)
        n = X.shape[0]
        k = min(int(k), n)
        start = int(np.random.default_rng(seed).integers(n))
        chosen = torch.empty(k, dtype=torch.long, device=device)
        chosen[0] = start
        available = torch.ones(n, dtype=torch.bool, device=device)
        available[start] = False
        neg = torch.tensor(float("-inf"), dtype=dtype, device=device)
        dist = _row_norms(X, X[start])
        for t in range(1, k):                # indices stay on the device, no host sync per step
            i = torch.argmax(torch.where(available, dist, neg)).view(1)
            chosen[t:t + 1] = i
            available.index_fill_(0, i, False)
            dist = torch.minimum(dist, _row_norms(X, X.index_select(0, i)[0]))
    return [int(v) for v in chosen.cpu().tolist()]


def _assign(X, C, xx, cc, max_elems: int = 1 << 27):
    """Nearest centre (first index on ties) and its squared distance ||x||^2 - 2 x.c + ||c||^2 (clipped
    at 0) for every row of X, in row chunks of at most ``max_elems`` distance entries."""
    import torch

    n, k = X.shape[0], C.shape[0]
    rows = max(1, min(CHUNK_ROWS, max_elems // max(k, 1)))
    labels = torch.empty(n, dtype=torch.long, device=X.device)
    best = torch.empty(n, dtype=X.dtype, device=X.device)
    for s in range(0, n, rows):
        block = torch.clamp(xx[s:s + rows, None] - 2.0 * (X[s:s + rows] @ C.T) + cc[None, :], min=0.0)
        values, idx = torch.min(block, dim=1)
        labels[s:s + rows], best[s:s + rows] = idx, values
    return labels, best


def _acc_dtype(tensor):
    """float64 for accumulations, float32 on MPS (which has no float64)."""
    import torch

    return tensor.dtype if tensor.device.type == "mps" else torch.float64


def kmeans_pp_torch(X, n_clusters: int, random_state: np.random.RandomState, xx):
    """scikit-learn's greedy k-means++ (``_kmeans_plusplus`` with unit weights) on a torch matrix.

    Uses the same RandomState calls in the same order: one choice for the first centre, then
    ``2 + int(log(k))`` uniforms per centre, candidates by searchsorted on the cumulative potential.
    Returns the centre indices (numpy int array).
    """
    import torch

    n = X.shape[0]
    trials = 2 + int(np.log(n_clusters))
    indices = np.full(n_clusters, -1, dtype=int)
    first = int(random_state.choice(n, p=np.full(n, 1.0 / n)))
    indices[0] = first
    closest = torch.clamp(xx - 2.0 * (X @ X[first]) + xx[first], min=0.0)
    pot = float(closest.sum())
    for c in range(1, n_clusters):
        rand_vals = random_state.uniform(size=trials) * pot
        acc = _acc_dtype(X)
        cum = torch.cumsum(closest.to(acc), dim=0)
        cand = torch.searchsorted(cum, torch.as_tensor(rand_vals, dtype=acc, device=X.device))
        cand = torch.clamp(cand, max=n - 1)
        d = torch.clamp(xx[None, :] - 2.0 * (X[cand] @ X.T) + xx[cand][:, None], min=0.0)
        d = torch.minimum(closest[None, :], d)
        pots = d.to(acc).sum(dim=1)
        best = int(torch.argmin(pots))
        pot = float(pots[best])
        closest = d[best]
        indices[c] = int(cand[best])
    return indices


def kmeans_torch(features, k: int, *, n_init: int = 3, seed: int = 0, device: str = "cpu", dtype=None,
                 max_iter: int = 300, tol: float = 1e-4) -> tuple[np.ndarray, np.ndarray, dict]:
    """Lloyd's k-means with scikit-learn's k-means++ initialization. Returns (labels, centres, info).

    The ``n_init`` runs share one RandomState(seed) as scikit-learn's KMeans does. A run stops when
    the labels no longer change or when the squared centre shift is at most ``tol`` times the mean
    feature variance, and the run with the lowest inertia is kept (first on ties). An empty cluster
    takes the record farthest from its current centre (largest distance first, ties by index), which
    follows scikit-learn's rule except for the order among the farthest records.
    """
    import torch

    dtype = dtype or torch.float32
    with no_tf32(), torch.no_grad():
        X = _tensor(features, device, dtype)
        n = X.shape[0]
        k = min(int(k), n)
        xx = (X * X).sum(dim=1)
        rs = np.random.RandomState(seed)
        tol_scaled = float(X.var(dim=0, unbiased=False).mean()) * tol
        best = None
        runs = []
        for _ in range(int(n_init)):
            idx = kmeans_pp_torch(X, k, rs, xx)
            centers = X[torch.as_tensor(idx, device=device)].clone()
            labels = None
            strict, it = False, 0
            for it in range(1, int(max_iter) + 1):
                cc = (centers * centers).sum(dim=1)
                new_labels, dist_to_own = _assign(X, centers, xx, cc)
                counts = torch.bincount(new_labels, minlength=k)
                sums = torch.zeros_like(centers).index_add_(0, new_labels, X)
                new_centers = centers.clone()
                filled = counts > 0
                new_centers[filled] = sums[filled] / counts[filled, None].to(dtype)
                empty = torch.nonzero(~filled).flatten()
                if len(empty):
                    order = torch.argsort(-dist_to_own, stable=True)[: len(empty)]
                    new_centers[empty] = X[order]
                shift = float(((new_centers - centers) ** 2).sum())
                centers = new_centers
                if labels is not None and torch.equal(new_labels, labels):
                    strict = True
                    labels = new_labels
                    break
                labels = new_labels
                if shift <= tol_scaled:
                    break
            cc = (centers * centers).sum(dim=1)
            labels, dist_to_own = _assign(X, centers, xx, cc)
            inertia = float(dist_to_own.to(_acc_dtype(dist_to_own)).sum())
            runs.append({"iterations": it, "strict_convergence": strict, "inertia": inertia})
            if best is None or inertia < best[0]:
                best = (inertia, labels.cpu().numpy(), centers.cpu().numpy())
    return best[1], best[2], {"runs": runs, "k": k, "n_init": int(n_init)}


def torch_device(selection_device: str) -> str | None:
    """Torch device of a ``selection_device`` value, or None for the numpy and scikit-learn path ("cpu")."""
    if selection_device in ("", "cpu", None):
        return None
    return "cpu" if selection_device == "torch_cpu" else selection_device


def record(ctx, name: str, device: str | None) -> None:
    """Note the selection device of strategy ``name`` in ctx.extras['strategy_info'] (scores.json selection_info)."""
    info = ctx.extras.get("strategy_info")
    if isinstance(info, dict):
        info.setdefault(name, {})["selection_device"] = device or "cpu (numpy and scikit-learn)"


def kmeans_coverage_torch(features, k: int, seed: int = 0, device: str = "cpu", n_init: int = 3) -> list[int]:
    """Exact-size k-means medoids from ``kmeans_torch`` and the nearest-to-centroid rule of the numpy path."""
    from omniselect.tools.pairing import exact_kmeans_representatives

    labels, centers, _ = kmeans_torch(features, k, n_init=n_init, seed=seed, device=device)
    X = np.asarray(features, dtype=np.float64).reshape(len(features), -1)
    return exact_kmeans_representatives(X.astype(np.float32), labels, centers.astype(np.float32), k)
