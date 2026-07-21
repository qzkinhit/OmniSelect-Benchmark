"""Build a small *real* standardized pool + held-out set from HF datasets.

Streams a few records per domain (general / math / code), maps each to a
UnifiedRecord, and caches:

    data/processed/real_pool.jsonl      (selection pool)
    data/processed/real_heldout.jsonl   (held-out eval set, same mixture)

Domains use confirmed-reachable corpora; the code source tries a couple of
fallbacks. Re-runs are cached (delete the files to rebuild).
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from mmdataselect.datatypes import DOMAIN_CODE, DOMAIN_GENERAL, DOMAIN_MATH, Modality, UnifiedRecord  # noqa: E402
from mmdataselect.utils.io import write_jsonl  # noqa: E402
from mmdataselect.utils.logger import get_logger  # noqa: E402

log = get_logger("build_real_pool")

N_PER_DOMAIN = int(os.environ.get("POOL_N", "60"))
N_HELDOUT = int(os.environ.get("HELDOUT_N", "45"))
MIN_CHARS, MAX_CHARS = 200, 2000

# (path, config, split, [text field candidates])
SOURCES = {
    DOMAIN_GENERAL: [("HuggingFaceFW/fineweb-edu", None, "train", ["text"])],
    DOMAIN_MATH: [("HuggingFaceTB/finemath", "finemath-4plus", "train", ["text"])],
    DOMAIN_CODE: [
        ("code_search_net", "python", "train", ["whole_func_string", "func_code_string"]),
        ("bigcode/the-stack-smol", None, "train", ["content"]),
    ],
}


def _stream(path, config, split, fields, take):
    from datasets import load_dataset

    ds = load_dataset(path, name=config, split=split, streaming=True)
    out = []
    for ex in ds:
        key = next((f for f in fields if ex.get(f)), None)
        if not key:
            continue
        t = " ".join(str(ex[key]).split())
        if MIN_CHARS <= len(t) <= MAX_CHARS:
            out.append(t[:MAX_CHARS])
        if len(out) >= take:
            break
    return out


def _collect(domain, take):
    for path, config, split, fields in SOURCES[domain]:
        try:
            rows = _stream(path, config, split, fields, take)
            if rows:
                log.info("%s <- %s [%s]: %d records", domain, path, config or "-", len(rows))
                return rows, path
        except Exception as e:  # noqa: BLE001
            log.warning("%s source %s failed: %s", domain, path, str(e)[:100])
    log.warning("%s: no source available; skipping", domain)
    return [], None


def main() -> int:
    pool_path = os.path.join(_REPO, "data/processed/real_pool.jsonl")
    held_path = os.path.join(_REPO, "data/processed/real_heldout.jsonl")
    if os.path.exists(pool_path) and os.path.exists(held_path):
        log.info("cached pool/heldout already exist; delete to rebuild")
        print("BUILD CACHED")
        return 0

    take = N_PER_DOMAIN + (N_HELDOUT // 3 + 1)
    pool, held = [], []
    for domain in (DOMAIN_GENERAL, DOMAIN_MATH, DOMAIN_CODE):
        texts, src = _collect(domain, take)
        for i, t in enumerate(texts):
            rec = UnifiedRecord(
                id=f"{domain[:2]}{i:04d}",
                modality=Modality.TEXT,
                domain=domain,
                text=t,
                meta={"source": src},
            )
            (held if i < N_HELDOUT // 3 else pool).append(rec)

    if not pool:
        log.error("empty pool — no datasets reachable")
        return 3
    write_jsonl((r.to_dict() for r in pool), pool_path)
    write_jsonl((r.to_dict() for r in held), held_path)
    from collections import Counter

    dist = Counter(r.domain for r in pool)
    print(f"BUILD OK | pool={len(pool)} {dict(dist)} -> {pool_path}")
    print(f"          heldout={len(held)} -> {held_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
