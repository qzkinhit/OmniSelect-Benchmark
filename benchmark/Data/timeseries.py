"""Forecasting series: ETT (OT column) and DaISy process-identification streams, sha256-checked.

``load_series`` returns the raw univariate series and the source path. ETTh1, ETTh2 and ETTm1
read data/processed/<name>.csv (column OT). Daisy_cstr and daisy_steamgen read data/daisy/<name>.dat
(column 2 concentration, column 5 drum pressure), decompressing the .gz copy when needed. Every
file is compared with the sha256 the canonical runs used.
"""
from __future__ import annotations

import gzip
import os
import shutil

import numpy as np

from omniselect.utils.hashing import file_sha256

SERIES_SHA256 = {
    "ETTh1": "5c155a1b14dcafcdc64f76b86c30637b80c5db42f2454d70da341cd7a8305575",
    "ETTh2": "14964a31bcfab7cdb8e5499962525fc58c719dc90c41f9a39ddf80f3def72f52",
    "ETTm1": "093cc4efd56a6bf68fb20cc93a2a79a4fbb06f02c8f4e7e5efa5520cc68afce6",
    "daisy_cstr": "0ffdda8a1b962d377dc34371be105bd9dcaef7fcca40554e666841efeec6b84d",
    "daisy_steamgen": "7f1e66031197c9502c7c7583b313b6349b7da678644410f4902d18b743eabc23",
}
DAISY_COLUMN = {"cstr": 2, "steamgen": 5}


def load_series(name: str, data_root: str) -> tuple[np.ndarray, str]:
    """(series as float array, source path) after the sha256 check."""
    if name not in SERIES_SHA256:
        raise KeyError(f"unknown series {name!r}; known: {sorted(SERIES_SHA256)}")
    if name.startswith("daisy_"):
        short = name.split("_", 1)[1]
        path = os.path.join(data_root, "daisy", f"{short}.dat")
        if not os.path.exists(path) and os.path.exists(path + ".gz"):
            with gzip.open(path + ".gz", "rb") as f_in, open(path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        _check(path, name)
        return np.loadtxt(path)[:, DAISY_COLUMN[short]].astype(float), path
    import pandas as pd

    path = os.path.join(data_root, "processed", f"{name.lower()}.csv")
    _check(path, name)
    return pd.read_csv(path)["OT"].to_numpy(dtype=float), path


def _check(path: str, name: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing {path}; run python data/fetch_data.py --only ett daisy")
    observed = file_sha256(path)
    if observed != SERIES_SHA256[name]:
        raise ValueError(f"{name} sha256 mismatch: expected {SERIES_SHA256[name]}, found {observed}")


def normalize(series: np.ndarray) -> np.ndarray:
    """z-score with mean and std of the first 70% of the series (the canonical normalization)."""
    head = series[: int(0.7 * len(series))]
    return (series - head.mean()) / (head.std() + 1e-9)


def windows(series: np.ndarray, starts: np.ndarray, L: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Input windows of length L and targets of length H at the given start positions."""
    X = np.stack([series[i:i + L] for i in starts])
    Y = np.stack([series[i + L:i + L + H] for i in starts])
    return X, Y
