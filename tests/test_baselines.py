"""Smoke tests for the faithful baselines (random / full_data / dsir / zip / if_mates / dmf).

Each baseline exposes a pure ``select`` returning kept pool indices, returns a full
ranking when k == n, and never needs torch to import (influence falls back to a CPU
proxy). Kept CPU-only and fast.
"""
import numpy as np
import pytest

from mmdataselect.datatypes import DOMAIN_CODE, DOMAIN_MATH, Modality, UnifiedRecord


@pytest.fixture
def pool():
    recs = []
    for i in range(24):
        dom = DOMAIN_MATH if i % 2 else DOMAIN_CODE
        txt = f"def f{i}(x): return x + {i}  # sample {i} solve x^2 = {i} step by step"
        recs.append(UnifiedRecord(id=str(i), modality=Modality.TEXT, domain=dom, text=txt))
    return recs


def _check(idx, k, n):
    assert len(idx) == k
    assert len(set(idx)) == k
    assert all(0 <= i < n for i in idx)


def test_zip(pool):
    from baselines.zip.method import select

    n = len(pool)
    _check(select(pool, 12), 12, n)
    assert len(select(pool, n)) == n  # k == n -> full ranking


def test_if_mates_precomputed(pool):
    from baselines.if_mates.method import select

    n = len(pool)
    infl = np.linspace(0, 1, n)
    idx = select(pool, 10, influence=infl)
    _check(idx, 10, n)
    # Top-K by influence -> must include the highest-influence record
    assert (n - 1) in idx
    assert len(select(pool, n, influence=infl)) == n


def test_if_mates_cpu_fallback(pool):
    from baselines.if_mates.method import select

    _check(select(pool, 8), 8, len(pool))  # no model -> CPU proxy, must still run


def test_dmf(pool):
    from baselines.dmf.method import select

    n = len(pool)
    _check(select(pool, 12), 12, n)
    assert len(select(pool, n)) == n


def test_dsir(pool):
    from baselines.dsir.method.dsir_select import dsir_select

    idx, weights = dsir_select([r.text for r in pool], [r.id for r in pool], 10, domains=[r.domain for r in pool])
    _check(idx, 10, len(pool))


def test_random_and_full(pool):
    from baselines.full_data.method.full_select import select as full_select
    from baselines.random.method.random_select import select as rand_select

    n = len(pool)
    ids = [r.id for r in pool]
    assert len(full_select(ids)) == n
    rsel = rand_select(ids, 10, seed=0)
    assert len(rsel) == 10 and len(set(rsel)) == 10
    assert rand_select(ids, 10, seed=0) == rand_select(ids, 10, seed=0)  # deterministic
