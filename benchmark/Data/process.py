"""Tennessee Eastman Process (Braatz files): d00..d21 training and test files, sha256-checked.

``load_tep`` stacks d00 plus the first ``n_faults`` fault files (test fault files keep rows
[160:], after the fault onset), standardizes with the training statistics, and draws pool,
validation, test and calibration indices with default_rng(seed) as the canonical runner did.
The checksums are read from data/tep/SHA256SUMS.txt.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from omniselect.utils.hashing import file_sha256


@dataclass
class TEPArrays:
    """Standardized arrays and index lists of one TEP draw."""

    Xp: np.ndarray
    yp: np.ndarray
    Xval: np.ndarray
    yval: np.ndarray
    Xt: np.ndarray
    yt: np.ndarray
    Xcal: np.ndarray
    ycal: np.ndarray
    pool_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    cal_idx: np.ndarray
    n_classes: int
    files: list


def _read(tep_dir: str, fname: str) -> np.ndarray:
    a = np.loadtxt(os.path.join(tep_dir, fname))
    return a.T if a.shape[0] == 52 else a


def verify(tep_dir: str, manifest: str, n_faults: int) -> list[str]:
    """Check every required file against the manifest. Return the file names."""
    expected = {}
    with open(manifest, encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and not line.startswith("#"):
                digest, name = line.split(None, 1)
                expected[name.strip()] = digest
    required = ["d00.dat", "d00_te.dat"]
    for fault in range(1, n_faults + 1):
        required.extend([f"d{fault:02d}.dat", f"d{fault:02d}_te.dat"])
    for name in required:
        path = os.path.join(tep_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing {path}; run python data/fetch_data.py --only tep")
        if file_sha256(path) != expected.get(name):
            raise ValueError(f"TEP file {name} sha256 mismatch")
    return required


def load_tep(data_root: str, manifest: str, n_faults: int, pool_n: int, val_n: int, test_n: int,
             seed: int) -> TEPArrays:
    """Canonical TEP draw (normal plus faults 1..n_faults, 22 classes for n_faults = 21)."""
    from sklearn.preprocessing import StandardScaler

    tep_dir = os.path.join(data_root, "tep")
    files = verify(tep_dir, manifest, n_faults)
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte = [_read(tep_dir, "d00.dat")], [], [_read(tep_dir, "d00_te.dat")], []
    ytr.append(np.zeros(len(Xtr[-1]), int))
    yte.append(np.zeros(len(Xte[-1]), int))
    for k in range(1, n_faults + 1):
        tr = _read(tep_dir, f"d{k:02d}.dat")
        Xtr.append(tr)
        ytr.append(np.full(len(tr), k, int))
        te = _read(tep_dir, f"d{k:02d}_te.dat")[160:]
        Xte.append(te)
        yte.append(np.full(len(te), k, int))
    Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
    Xte, yte = np.vstack(Xte), np.concatenate(yte)
    if pool_n > len(Xtr) or val_n + test_n > len(Xte):
        raise ValueError("requested TEP split sizes exceed the source arrays")
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    pi = rng.permutation(len(Xtr))[:pool_n]
    ti = rng.permutation(len(Xte))
    vi, tei, ci = ti[:val_n], ti[val_n:val_n + test_n], ti[val_n + test_n:]
    return TEPArrays(Xtr[pi], ytr[pi], Xte[vi], yte[vi], Xte[tei], yte[tei], Xte[ci], yte[ci],
                     pi, vi, tei, ci, n_faults + 1, files)
