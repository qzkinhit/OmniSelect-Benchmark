"""Five-domain text pool: qpool_train.jsonl and qpool_heldout.jsonl as UnifiedRecords.

``load_pool`` reads both files from data/processed (records with id, domain, text and meta),
checks their sha256 against the registered pool (25,000 training and 2,000 held-out records) and
returns them in file order. ``pool_sha256`` hashes the training file, as the canonical text runner
did for its caches. The held-out file is split per domain by the text track.
"""
from __future__ import annotations

import os

from omniselect.core.datatypes import UnifiedRecord
from omniselect.utils.hashing import file_sha256
from omniselect.utils.io import read_jsonl


# sha256 of the registered five-domain pool (data/processed/pool_manifest.json). Any other file is refused.
REGISTERED_SHA256 = {
    "qpool_train.jsonl": "84e174dbb097288c6b4473af2af8d6cb46a0b00a000b6534545725e53f9939c5",
    "qpool_heldout.jsonl": "1e4a45c9c959995a3c10c840dc2a8b84ce33bd82ba17682c8f50c3b4b3a1e785",
}


def load_pool(data_root: str, *, manifest_path: str | None = None
              ) -> tuple[list[UnifiedRecord], list[UnifiedRecord], str]:
    """Load registered 25k data, or the strict 100k contract via an explicit manifest.

    The text runner can opt in with OMNISELECT_TEXT_MANIFEST. No manifest is
    autodiscovered, and an invalid explicit manifest never falls back to 25k.
    """
    if manifest_path is None:
        manifest_path = os.environ.get("OMNISELECT_TEXT_MANIFEST")
    if manifest_path is not None:
        from benchmark.Data.text_manifest import load_manifest_pool
        train, held, path = load_manifest_pool(data_root, manifest_path)
        return ([UnifiedRecord.from_dict(row) for row in train],
                [UnifiedRecord.from_dict(row) for row in held], path)
    pool_path = os.path.join(data_root, "processed", "qpool_train.jsonl")
    held_path = os.path.join(data_root, "processed", "qpool_heldout.jsonl")
    for path in (pool_path, held_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing {path}; see data/README.md")
        expected = REGISTERED_SHA256[os.path.basename(path)]
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(f"{path} sha256 {observed[:12]} is not the registered pool {expected[:12]} "
                             "(25,000 training and 2,000 held-out records, see data/README.md)")
    pool = [UnifiedRecord.from_dict(d) for d in read_jsonl(pool_path)]
    held = [UnifiedRecord.from_dict(d) for d in read_jsonl(held_path)]
    return pool, held, pool_path


def pool_sha256(path: str) -> str:
    """sha256 of the pool file."""
    return file_sha256(path)
