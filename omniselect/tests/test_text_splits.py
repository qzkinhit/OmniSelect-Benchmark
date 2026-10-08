"""Text held-out splits: v2 totals from splits.text_sizes, canonical halves and quarters unchanged."""
from __future__ import annotations

import os

import pytest

from omniselect.core.datatypes import Modality, UnifiedRecord
from tracks.text.pool import split_heldout

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def _held(per_domain: int) -> list[UnifiedRecord]:
    return [UnifiedRecord(id=f"{d}{i:04d}", modality=Modality.TEXT, domain=d, text="x")
            for d in ("code", "general", "image", "math", "table") for i in range(per_domain)]


def test_v2_sizes_and_disjointness():
    held = _held(400)
    v2 = split_heldout(held, 0, "three_way")
    assert (len(v2.con), len(v2.rank), len(v2.conf), len(v2.report)) == (400, 600, 500, 500)
    ids = [r.id for part in (v2.con, v2.rank, v2.conf, v2.report) for r in part]
    assert len(set(ids)) == 2000 and [r.id for r in v2.reference] == [r.id for r in v2.con]
    # the report split is the tail of each domain in file order
    assert {r.id for r in v2.report} == {f"{d}{i:04d}" for d in ("code", "general", "image", "math", "table")
                                         for i in range(300, 400)}
    custom = split_heldout(held, 0, "three_way", sizes=(200, 800, 500, 500))
    assert (len(custom.con), len(custom.rank)) == (200, 800)
    canonical = split_heldout(held, 0, "two_way")
    assert (len(canonical.reference), len(canonical.con), len(canonical.rank), len(canonical.report)) == \
        (1000, 250, 250, 500)


@pytest.mark.skipif(not os.path.isfile(os.path.join(DATA, "processed", "qpool_heldout.jsonl")),
                    reason="registered text pool not present")
def test_registered_heldout_counts():
    from benchmark.Data.text import load_pool

    _, held, _ = load_pool(DATA)
    v2 = split_heldout(held, 0, "three_way")
    assert (len(v2.con), len(v2.rank), len(v2.conf), len(v2.report)) == (400, 600, 500, 500)
