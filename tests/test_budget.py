"""Budget.from_cfg + resolve across fraction / records / tokens, incl. edge cases."""
from __future__ import annotations

from mmdataselect.budget import Budget


def test_from_cfg_defaults_when_none():
    b = Budget.from_cfg(None)
    assert b.kind == "fraction"
    assert b.value == 0.5


def test_from_cfg_reads_nested_budget_block():
    b = Budget.from_cfg({"budget": {"kind": "records", "value": 7}})
    assert b.kind == "records"
    assert b.value == 7.0  # value is coerced to float


def test_from_cfg_reads_flat_block():
    # A flat dict (no "budget" key) is treated as the budget spec itself.
    b = Budget.from_cfg({"kind": "tokens", "value": 1000})
    assert b.kind == "tokens"
    assert b.value == 1000.0


def test_resolve_fraction_rounds_and_clamps():
    b = Budget(kind="fraction", value=0.5)
    assert b.resolve(10) == 5
    # round-half-to-even style via int(round(...)): 0.25 * 10 = 2.5 -> 2
    assert Budget(kind="fraction", value=0.25).resolve(10) == 2
    # always keep at least 1 even for tiny fractions on a non-empty pool.
    assert Budget(kind="fraction", value=0.0).resolve(10) == 1
    # fraction >= 1 is clamped to the pool size.
    assert Budget(kind="fraction", value=2.0).resolve(10) == 10


def test_resolve_records_exact_with_clamp():
    assert Budget(kind="records", value=3).resolve(10) == 3
    # request more than available -> clamp to n_total.
    assert Budget(kind="records", value=99).resolve(10) == 10
    # request <= 0 -> floor at 1 on a non-empty pool.
    assert Budget(kind="records", value=0).resolve(10) == 1


def test_resolve_tokens_with_counts():
    # avg token count = 10; cap value 25 -> int(25/10) = 2 records fit.
    b = Budget(kind="tokens", value=25)
    counts = [10, 10, 10, 10]
    assert b.resolve(4, token_counts=counts) == 2


def test_resolve_tokens_clamped_to_pool():
    # Large cap relative to avg length cannot exceed n_total.
    b = Budget(kind="tokens", value=10_000)
    assert b.resolve(3, token_counts=[5, 5, 5]) == 3


def test_resolve_tokens_without_counts_falls_back_to_half():
    # No token_counts available -> heuristic half the pool.
    b = Budget(kind="tokens", value=1234)
    assert b.resolve(8) == 4
    assert b.resolve(8, token_counts=[]) == 4


def test_resolve_empty_pool_is_zero():
    for kind in ("fraction", "records", "tokens"):
        assert Budget(kind=kind, value=0.5).resolve(0) == 0
