"""Explicit 100k text-pool contract; never relaxes the registered 25k loader.

The manifest records source provenance. The offline validator additionally verifies
every source shard against its exact revision in the Hugging Face hub cache.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath

from omniselect.utils.hashing import file_sha256
from omniselect.utils.io import read_jsonl

DOMAINS = ["general", "math", "code", "image", "table"]
KINDS = ("truncation", "template", "crossdomain", "lowtier")
PINNED_REVISIONS = {
    "HuggingFaceFW/fineweb-edu": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    "HuggingFaceTB/finemath": "e92b25a616738fe95dc186b64dfb19f9c8525594",
    "codeparrot/codeparrot-clean-valid": "4db92d2ec0c1b4c41eeb439cfae16854511d9dcd",
    "yerevann/coco-karpathy": "448fdb1bc7b2d09e46881c4541a14d796a3d41e8",
    "mstz/adult": "f90a6fc5f6efb865382c05b75ab9037b62104172",
}
SOURCE_PREFIXES = {
    "HuggingFaceFW/fineweb-edu": ("sample/10BT/",),
    "HuggingFaceTB/finemath": ("finemath-4plus/", "finemath-3plus/"),
    "codeparrot/codeparrot-clean-valid": ("",),
    "yerevann/coco-karpathy": ("",),
    "mstz/adult": ("income/",),
}
R6_CONFIG = {
    "POOL_HI": 12000, "HELD_PER": 400, "MODALITIES": DOMAINS,
    "LOW_FRAC": 0.40, "MINC": 150, "MAXC": 2000,
}


def _require(ok, message):
    if not ok:
        raise ValueError(f"invalid explicit text manifest: {message}")


def validate_metadata(manifest):
    _require(isinstance(manifest, dict), "manifest must be an object")
    _require(manifest.get("schema") is None, "legacy manifest must not declare a newer schema")
    config = manifest.get("config", {})
    _require(isinstance(config, dict), "config must be an object")
    for key, value in R6_CONFIG.items():
        _require(config.get(key) == value, f"config.{key} must equal {value!r}")
    _require(manifest.get("counts") == {"train": 100000, "heldout": 2000}, "counts must be 100000/2000")
    _require(manifest.get("revisions") == PINNED_REVISIONS, "source revisions differ from pinned sources")
    for key in ("train_sha256", "heldout_sha256"):
        _require(re.fullmatch(r"[0-9a-f]{64}", str(manifest.get(key, ""))), f"missing or invalid {key}")
    shards = manifest.get("shards")
    _require(isinstance(shards, list) and shards, "source shards must be nonempty")
    seen, coverage = set(), set()
    for shard in shards:
        _require(isinstance(shard, dict), "source shard must be an object")
        repo, name = shard.get("repo"), shard.get("file")
        _require(repo in PINNED_REVISIONS, "unknown source repository")
        _require(shard.get("rev") == PINNED_REVISIONS[repo], f"unpinned shard revision for {repo}")
        _require(isinstance(name, str) and name and not PurePosixPath(name).is_absolute()
                 and ".." not in PurePosixPath(name).parts, "invalid source shard path")
        matched = [p for p in SOURCE_PREFIXES[repo] if name.startswith(p)]
        _require(matched, f"unexpected source subset for {repo}")
        _require((repo, name) not in seen, f"duplicate source shard {repo}/{name}")
        seen.add((repo, name))
        coverage.update((repo, p) for p in matched)
        _require(type(shard.get("bytes")) is int and shard["bytes"] > 0, "invalid source byte count")
        _require(re.fullmatch(r"[0-9a-f]{64}", str(shard.get("sha256", ""))), "invalid source sha256")
    expected = {(repo, prefix) for repo, prefixes in SOURCE_PREFIXES.items() for prefix in prefixes}
    _require(coverage == expected, "missing pinned source subset, including low-tier finemath-3plus")


def validate_record_splits(train, held, *, pool_hi, held_per):
    """Check exact counts and all rows, including low-quality train/held text overlap."""
    _require(pool_hi > 0 and pool_hi % 6 == 0 and held_per > 0, "invalid record-count contract")
    per_kind = pool_hi // 6
    _require(len(train) == len(DOMAINS) * (pool_hi + 4 * per_kind), "training count mismatch")
    _require(len(held) == len(DOMAINS) * held_per, "held-out count mismatch")
    ids, texts = set(), {}
    for split, rows in (("train", train), ("heldout", held)):
        counts, hashes = Counter(), set()
        for row in rows:
            _require(isinstance(row, dict), f"{split} row must be an object")
            _require(isinstance(row.get("id"), str) and row["id"], f"{split} id missing")
            _require(row["id"] not in ids, "duplicate ID within or across train/heldout")
            ids.add(row["id"])
            _require(row.get("domain") in DOMAINS and row.get("modality") == "text", "invalid domain/modality")
            _require(isinstance(row.get("text"), str) and row["text"].strip(), "empty or invalid text")
            meta = row.get("meta")
            _require(isinstance(meta, dict), "meta must be an object")
            quality, noise = meta.get("quality"), meta.get("noise", "")
            allowed = ((quality == "high" and noise == "heldout") if split == "heldout" else
                       ((quality == "high" and noise == "") or (quality == "low" and noise in KINDS)))
            _require(allowed, f"invalid {split} quality/noise")
            counts[(row["domain"], quality, noise)] += 1
            hashes.add(hashlib.sha256(row["text"].encode("utf-8")).digest())
        for domain in DOMAINS:
            if split == "heldout":
                _require(counts[(domain, "high", "heldout")] == held_per, f"heldout count for {domain}")
            else:
                _require(counts[(domain, "high", "")] == pool_hi, f"high count for {domain}")
                for kind in KINDS:
                    _require(counts[(domain, "low", kind)] == per_kind, f"{kind} count for {domain}")
        texts[split] = hashes
    _require(not texts["train"] & texts["heldout"], "train/heldout text overlap")


def load_manifest_pool(data_root, manifest_path):
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if isinstance(manifest, dict) and manifest.get("schema") == "r6_text100k_adult_v2":
        from benchmark.Data.text_manifest_adult_v2 import load_pool
        return load_pool(data_root, manifest)
    validate_metadata(manifest)
    processed = Path(data_root) / "processed"
    paths = [processed / "qpool_train.jsonl", processed / "qpool_heldout.jsonl"]
    for path, key in zip(paths, ("train_sha256", "heldout_sha256")):
        _require(file_sha256(str(path)) == manifest[key], f"{path.name} sha256 mismatch")
    train, held = [list(read_jsonl(str(path))) for path in paths]
    validate_record_splits(train, held, pool_hi=12000, held_per=400)
    return train, held, str(paths[0])


def verify_source_shards(manifest, hub_root):
    """Require every shard at its pinned revision, with exact size and sha256."""
    if isinstance(manifest, dict) and manifest.get("schema") == "r6_text100k_adult_v2":
        from benchmark.Data.text_manifest_adult_v2 import validate_metadata as validate_v2
        validate_v2(manifest)
    else:
        validate_metadata(manifest)
    for shard in manifest["shards"]:
        path = (Path(hub_root) / ("datasets--" + shard["repo"].replace("/", "--")) /
                "snapshots" / shard["rev"] / shard["file"])
        _require(path.is_file(), f"missing pinned source shard {path}")
        _require(path.stat().st_size == shard["bytes"], f"source size mismatch: {path}")
        _require(file_sha256(str(path)) == shard["sha256"], f"source sha256 mismatch: {path}")
