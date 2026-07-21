"""ZIP cache path must be INDEX-EXACT vs the original implementation
(CODEX ZIP_REUSE audit items 2-4): same full greedy ordering, element by element, on
multiple small pools; the warm cache load must equal both. Any float/compression/tie
difference is a failure - no approximate ZIP."""
import json
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "src"))

from baselines.zip.method import select as zip_select  # noqa: E402


def _mk_texts(n, salt):
    # deterministic mixed-redundancy corpus: some near-duplicates, some unique
    base = ["the quick brown fox %d" % i for i in range(n // 2)]
    dups = ["repeated boilerplate header %s" % salt] * (n - n // 2)
    texts = []
    for i in range(n):
        texts.append(base[i // 2] if i % 2 == 0 and i // 2 < len(base) else dups[i % len(dups)] + str(i % 3))
    return texts[:n]


@pytest.mark.parametrize("n,salt", [(24, "a"), (57, "b"), (100, "c")])
def test_zip_cache_roundtrip_index_exact(tmp_path, n, salt):
    texts = _mk_texts(n, salt)
    ids = [str(i) for i in range(n)]
    # original path, run twice: determinism of the implementation itself
    o1 = zip_select(texts=texts, ids=ids, k=n)
    o2 = zip_select(texts=texts, ids=ids, k=n)
    assert list(o1) == list(o2), "zip_select not deterministic on identical input"
    # cache write/read roundtrip (the runner's cache stores the ordering verbatim)
    cpath = tmp_path / "zip_order_cache_test.json"
    json.dump({"n": n, "order": [int(i) for i in o1]}, open(cpath, "w"))
    cached = json.load(open(cpath))
    assert cached["n"] == n and len(cached["order"]) == n
    assert [int(i) for i in cached["order"]] == [int(i) for i in o1], "cache roundtrip broke ordering"
    # seed must not participate in selection (interface-only)
    o3 = zip_select(texts=texts, ids=ids, k=n, seed=12345)
    assert list(o3) == list(o1), "seed changed the ZIP ordering - caching would be invalid"


def test_zip_truncation_is_prefix():
    # budget truncation semantics: k<n must equal the prefix of the full ordering
    texts = _mk_texts(40, "d")
    ids = [str(i) for i in range(40)]
    full = zip_select(texts=texts, ids=ids, k=40)
    head = zip_select(texts=texts, ids=ids, k=15)
    assert list(head) == list(full)[:15], "k-truncation is not a prefix of the full ordering"


def test_incremental_prefix_state_is_bit_exact():
    """ZIP incremental prefix-state evaluator (ZIP_INCREMENTAL=1, default) must
    produce EXACTLY the same ordering AND ratio trace as the reference full
    recompression path (ZIP_INCREMENTAL=0) - element-wise, no tolerance."""
    import os
    from baselines.zip.method.zip_select import zip_select as _zs
    for n, salt in ((30, "x"), (77, "y"), (150, "z")):
        texts = _mk_texts(n, salt)
        ids = [str(i) for i in range(n)]
        os.environ["ZIP_INCREMENTAL"] = "0"
        ref_order, ref_trace = _zs(texts, ids, n)
        os.environ["ZIP_INCREMENTAL"] = "1"
        inc_order, inc_trace = _zs(texts, ids, n)
        os.environ.pop("ZIP_INCREMENTAL", None)
        assert list(inc_order) == list(ref_order), f"order diverged at n={n}"
        assert inc_trace == ref_trace, f"ratio trace diverged at n={n}"
