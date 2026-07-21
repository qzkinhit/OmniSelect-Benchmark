"""Shared pytest fixtures + import path wiring.

All tests run on pure CPU and must import without torch. We insert both the repo
root and ``src`` onto ``sys.path`` so ``import mmdataselect.*`` resolves regardless
of how pytest is invoked.
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
for _p in (_SRC, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mmdataselect.datatypes import (  # noqa: E402
    DOMAIN_CODE,
    DOMAIN_GENERAL,
    DOMAIN_MATH,
    Modality,
    UnifiedRecord,
)


@pytest.fixture
def small_pool():
    """A small, deterministic pool with a deliberate redundancy structure.

    The first three records share near-identical text (a redundant cluster); the
    remaining records are lexically distinct. This lets diversity-aware selection
    and set-redundancy diagnostics be asserted meaningfully.
    """
    dup = "the quick brown fox jumps over the lazy dog"
    records = [
        UnifiedRecord(id="r0", modality=Modality.TEXT, domain=DOMAIN_GENERAL, text=dup),
        UnifiedRecord(id="r1", modality=Modality.TEXT, domain=DOMAIN_GENERAL, text=dup),
        UnifiedRecord(id="r2", modality=Modality.TEXT, domain=DOMAIN_GENERAL, text=dup),
        UnifiedRecord(
            id="r3",
            modality=Modality.IMAGE_TEXT,
            domain=DOMAIN_MATH,
            text="integrate sin over zero to pi yields exactly two units",
            meta={"src": "math"},
        ),
        UnifiedRecord(
            id="r4",
            modality=Modality.TEXT,
            domain=DOMAIN_CODE,
            text="def fibonacci(n): return n if n < 2 else fib(n-1) + fib(n-2)",
            meta={"lang": "python"},
        ),
        UnifiedRecord(
            id="r5",
            modality=Modality.AUDIO_TEXT,
            domain=DOMAIN_GENERAL,
            text="transcribed speech about volcanic island geology and tectonics",
        ),
    ]
    return records
