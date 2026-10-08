"""Content fingerprints for selections, id lists, arrays and files.

``sel_sha12`` hashes the sorted integer ids of a selection and is the cache key of the
train-once cache and the ``sel_sha12`` field of every run record. ``order_sha12`` keeps the
training order. ``ids_sha256`` hashes an id list as written to splits.json. The functions use
hashlib only and return lowercase hex strings.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence

import numpy as np


def sel_sha12(selection: Iterable[int]) -> str:
    """First 12 hex digits of sha256 over ``str(sorted(ids))``. Equals tools.pairing.sel_sha12."""
    return hashlib.sha256(str(sorted(int(i) for i in selection)).encode()).hexdigest()[:12]


def order_sha12(selection: Iterable[int]) -> str:
    """First 12 hex digits of sha256 over ``str(list(ids))`` in the given order."""
    return hashlib.sha256(str([int(i) for i in selection]).encode()).hexdigest()[:12]


def ids_sha256(ids: Sequence[Any]) -> str:
    """sha256 over the JSON list of ids (strings or integers) in the given order."""
    payload = json.dumps([i if isinstance(i, str) else int(i) for i in ids], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def arrays_sha256(*arrays: Any) -> str:
    """sha256 over dtype, shape and bytes of each array. Equals tools.pairing.arrays_sha256."""
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def file_sha256(path: str) -> str:
    """sha256 of a file read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
