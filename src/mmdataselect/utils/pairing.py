"""Pairing instrumentation for the unified same-code-state batch
(CODEX_CANONICAL_PARITY_GATE item 1, option B).

Three primitives shared by all four runners:
  sel_sha12(sel)      content fingerprint of a selection (sorted ids), recorded per method
                      row so the parity gate can compare controller-picked vs standalone.
  order_sha12(sel)    content fingerprint that preserves the actual training order.
  reset_rng(*parts)   deterministic RNG reset (python/numpy/torch) from crc32 of the
                      parts. Called (a) before every select() with (seed,'select',method)
                      so selection is path-independent, and (b) before every final fit
                      with (seed,'fit',stage), never with the method or selected ids.
                      Therefore every method in a paper seed uses the same model
                      initialization. Selections are sorted before fitting so the data
                      order is canonical. Identical selections still yield identical
                      metrics regardless of the code path that produced them. Activated
                      by PAIRED_RNG=1 to keep older canonical replays reproducible.
"""
import hashlib
import random
import zlib

import numpy as np


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
