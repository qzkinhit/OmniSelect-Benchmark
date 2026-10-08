"""Torch herding, k-center and k-means coverage against the numpy and scikit-learn paths (selection_device)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmark.Methods.Coverage.method import kmeans_coverage  # noqa: E402
from benchmark.Methods.Herding.method import herding  # noqa: E402
from benchmark.Methods.KCenter.method import kcenter_greedy  # noqa: E402
from benchmark.Methods._torch_select import (  # noqa: E402
    herding_torch,
    kcenter_torch,
    kmeans_coverage_torch,
    kmeans_pp_torch,
    kmeans_torch,
    torch_device,
)

POOL = np.random.default_rng(0).normal(size=(2000, 64))


def test_herding_and_kcenter_select_the_numpy_ids_on_cpu():
    k = 200
    assert herding_torch(POOL, k, "cpu") == herding(POOL, k)
    for seed in (0, 5):
        assert kcenter_torch(POOL, k, seed=seed, device="cpu") == kcenter_greedy(POOL, k, seed=seed)


def test_kmeans_reproduces_scikit_learn_kmeans():
    from sklearn.cluster import KMeans, kmeans_plusplus

    X = POOL.astype(np.float32)
    Xt = torch.as_tensor(POOL, dtype=torch.float64)
    _, sk_idx = kmeans_plusplus(POOL, 50, random_state=np.random.RandomState(3))
    assert np.array_equal(kmeans_pp_torch(Xt, 50, np.random.RandomState(3), (Xt * Xt).sum(1)), sk_idx)
    for k in (20, 200):
        ref = KMeans(n_clusters=k, n_init=3, random_state=3).fit(X)
        labels, centers, info = kmeans_torch(X, k, n_init=3, seed=3, device="cpu")
        assert np.array_equal(labels, ref.labels_) and len(info["runs"]) == 3
        assert np.array_equal(np.bincount(labels, minlength=k), np.bincount(ref.labels_, minlength=k))
        assert np.abs(centers - ref.cluster_centers_).max() < 1e-5
    # medoid ids: identical at k = 20. At k = 200 two of 200 medoids differ, because the float32
    # centre sums round differently from scikit-learn's and move a nearest-point near-tie
    assert kmeans_coverage_torch(X, 20, seed=3) == kmeans_coverage(X, 20, seed=3)
    overlap = set(kmeans_coverage_torch(X, 200, seed=3)) & set(kmeans_coverage(X, 200, seed=3))
    assert len(overlap) >= 196


def test_selection_device_names():
    assert torch_device("cpu") is None and torch_device("torch_cpu") == "cpu" and torch_device("cuda") == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_paths_match_the_cpu_torch_paths():
    k = 200
    assert kcenter_torch(POOL, k, seed=0, device="cuda") == kcenter_torch(POOL, k, seed=0, device="cpu")
    assert len(set(herding_torch(POOL, k, "cuda")) & set(herding(POOL, k))) >= 0.98 * k
    labels, _, _ = kmeans_torch(POOL.astype(np.float32), 20, n_init=3, seed=3, device="cuda")
    ref, _, _ = kmeans_torch(POOL.astype(np.float32), 20, n_init=3, seed=3, device="cpu")
    assert (labels == ref).mean() > 0.99


def test_driver_records_the_selection_device(tmp_path):
    import json

    from tracks.common.experiment import load_track, resolve_configs, run_cell

    track = load_track("process")
    methods = ("herding", "kcenter", "coreset", "random")
    rows = {}
    for device in ("cpu", "torch_cpu"):
        tcfg, ocfg = resolve_configs(track, "tep21", learner="mlp", seed=0, protocol="v2", smoke=True,
                                     track_overrides={"pool_n": 200, "val_n": 200, "test_n": 150, "mlp_max_iter": 20,
                                                      "methods": methods, "selection_device": device},
                                     omni_overrides={})
        run_cell(track, tcfg, ocfg, out_root=tmp_path, batch=f"sel_{device}", cli=["test"], standalone=True)
        cell = tmp_path / f"sel_{device}" / "process" / "tep21" / "mlp" / "seed_0"
        rows[device] = json.loads((cell / "metrics.json").read_text())["rows"]
        info = json.loads((cell / "candidates" / "herding" / "scores.json").read_text())["selection_info"]
        assert info["selection_device"] == ("cpu" if device == "torch_cpu" else "cpu (numpy and scikit-learn)")
        assert info["selection_secs"] >= 0
    for name in ("herding", "kcenter"):
        assert rows["cpu"][name]["sel_sha12"] == rows["torch_cpu"][name]["sel_sha12"], name
