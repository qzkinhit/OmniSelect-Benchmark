"""OpenML Electricity (id 151, version 1) with the numeric-array contract of the canonical runs.

``load_table`` fetches the table through sklearn, keeps numeric columns, encodes the target,
checks the OpenML metadata and the sha256 of the numeric arrays against the recorded contract,
draws pool, validation and test indices with default_rng(seed), and standardizes with the pool
statistics. Other OpenML names are accepted without a contract check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from omniselect.utils.hashing import arrays_sha256

ELECTRICITY_CONTRACT = {
    "id": "151", "name": "electricity", "version": "1", "file_id": "2419",
    "md5_checksum": "8ca97867d960ae029ae3a9ac2c923d34",
    "numeric_arrays_sha256": "c22ea751baf5d634ebab2945379785d16839416c3bf62e8a4d5f92cace62269f",
}


@dataclass
class TableArrays:
    """Standardized arrays and index lists of one draw."""

    Xp: np.ndarray
    yp: np.ndarray
    Xval: np.ndarray
    yval: np.ndarray
    Xt: np.ndarray
    yt: np.ndarray
    pool_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    n_classes: int
    manifest: dict = field(default_factory=dict)


def load_table(name: str, pool_n: int, val_n: int, test_n: int, seed: int) -> TableArrays:
    """Canonical draw of an OpenML table (version 1)."""
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler

    d = fetch_openml(name, version=1, as_frame=True)
    X = np.nan_to_num(d.data.select_dtypes(include="number").to_numpy(dtype=float))
    y_raw = d.target.to_numpy()
    classes = {c: i for i, c in enumerate(sorted(set(y_raw)))}
    y = np.array([classes[c] for c in y_raw])
    details = dict(getattr(d, "details", {}) or {})
    digest = arrays_sha256(X, y)
    manifest = {"openml_id": details.get("id"), "openml_name": details.get("name", name),
                "openml_version": details.get("version", "1"), "openml_file_id": details.get("file_id"),
                "openml_md5_checksum": details.get("md5_checksum"), "numeric_arrays_sha256": digest,
                "rows": int(len(X)), "numeric_features": int(X.shape[1])}
    if name == "electricity":
        for key in ("id", "name", "version", "file_id", "md5_checksum"):
            if details.get(key) != ELECTRICITY_CONTRACT[key]:
                raise ValueError(f"OpenML Electricity {key} drifted: {details.get(key)!r}")
        if digest != ELECTRICITY_CONTRACT["numeric_arrays_sha256"]:
            raise ValueError(f"OpenML Electricity numeric-array sha256 mismatch: {digest}")
    if pool_n + val_n + test_n > len(X):
        raise ValueError("requested split sizes exceed the table")
    perm = np.random.default_rng(seed).permutation(len(X))
    pi, vi, ti = perm[:pool_n], perm[pool_n:pool_n + val_n], perm[pool_n + val_n:pool_n + val_n + test_n]
    sc = StandardScaler().fit(X[pi])
    return TableArrays(sc.transform(X[pi]), y[pi], sc.transform(X[vi]), y[vi], sc.transform(X[ti]), y[ti],
                       pi, vi, ti, len(classes), manifest)
