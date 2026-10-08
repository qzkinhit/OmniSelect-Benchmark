"""Clustered coverage: k-means with k clusters on the track representation, one medoid each.

The record nearest each centroid is kept. Empty clusters are filled by distance to the
assigned centre (omniselect.tools.pairing.exact_kmeans_representatives), so exactly k distinct
records are returned. The text track uses domain-wise farthest-first traversal instead
(omniselect.core.selection.text_transfers.coverage_token_order).
"""
from __future__ import annotations

from omniselect.core.portfolio.registry import SelectionContext, register
from omniselect.tools.pairing import exact_kmeans_representatives


def kmeans_coverage(features, k: int, seed: int = 0) -> list[int]:
    """Exact-size k-means medoids (KMeans n_init=3, random_state=seed)."""
    from sklearn.cluster import KMeans

    n = features.shape[0]
    kk = min(k, n)
    km = KMeans(n_clusters=kk, n_init=3, random_state=seed).fit(features)
    return exact_kmeans_representatives(features, km.labels_, km.cluster_centers_, k)


@register("coreset", method_dir="Coverage", family="builtin", fidelity="exact", display="Coverage")
def coreset(ctx: SelectionContext, k: int) -> list[int]:
    """k-means coverage on ``extras['coverage_features']`` (the track representation otherwise).

    The vision track passes its float32 CLIP embeddings there, as the canonical runner clustered them.
    On a non-cpu ``extras['selection_device']`` the torch k-means of benchmark/Methods/_torch_select.py
    runs instead.
    """
    from benchmark.Methods._torch_select import kmeans_coverage_torch, record, torch_device

    device = torch_device(ctx.extras.get("selection_device", "cpu"))
    record(ctx, "coreset", device)
    features = ctx.extras.get("coverage_features", ctx.features)
    if device is None:
        return kmeans_coverage(features, k, seed=ctx.seed)
    return kmeans_coverage_torch(features, k, seed=ctx.seed, device=device)
