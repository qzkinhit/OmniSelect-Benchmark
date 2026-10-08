"""Small offline fixtures for the explicit R6 contract; no real pool is built."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from benchmark.Data import text as text_data
from benchmark.Data import text_manifest as contract
from omniselect.utils.hashing import file_sha256


def make_manifest():
    shards = []
    for repo, prefixes in contract.SOURCE_PREFIXES.items():
        for prefix in prefixes:
            shards.append({"repo": repo, "rev": contract.PINNED_REVISIONS[repo],
                           "file": prefix + "fixture.parquet", "bytes": 1, "sha256": "a" * 64})
    return {"config": copy.deepcopy(contract.R6_CONFIG), "counts": {"train": 100000, "heldout": 2000},
            "revisions": dict(contract.PINNED_REVISIONS), "shards": shards,
            "train_sha256": "b" * 64, "heldout_sha256": "c" * 64}


def make_rows(pool_hi=6, held_per=1):
    train, held = [], []
    for domain in contract.DOMAINS:
        for quality, noise, count in [("high", "", pool_hi)] + [
                ("low", kind, pool_hi // 6) for kind in contract.KINDS]:
            for i in range(count):
                identity = f"{domain}-{quality}-{noise}-{i}"
                train.append({"id": identity, "modality": "text", "domain": domain, "text": identity,
                              "meta": {"quality": quality, "noise": noise}})
        for i in range(held_per):
            identity = f"{domain}-held-{i}"
            held.append({"id": identity, "modality": "text", "domain": domain, "text": identity,
                         "meta": {"quality": "high", "noise": "heldout"}})
    return train, held


def test_default_does_not_autodiscover_manifest_or_allow_unregistered_pool(tmp_path, monkeypatch):
    monkeypatch.delenv("OMNISELECT_TEXT_MANIFEST", raising=False)
    processed = tmp_path / "processed"
    processed.mkdir()
    for name in text_data.REGISTERED_SHA256:
        (processed / name).write_text("{}\n")
    (processed / "pool_manifest.json").write_text(json.dumps(make_manifest()))
    with pytest.raises(ValueError, match="not the registered pool"):
        text_data.load_pool(str(tmp_path))


@pytest.mark.parametrize("field", ["counts", "revisions", "shards", "heldout_sha256", "config"])
def test_bad_manifest_rejected(field):
    manifest = make_manifest()
    manifest[field] = {} if field != "heldout_sha256" else "not-a-sha"
    with pytest.raises(ValueError):
        contract.validate_metadata(manifest)


@pytest.mark.parametrize("failure", ["duplicate_id", "held_id_overlap", "low_text_overlap", "count", "kind"])
def test_all_rows_and_low_text_isolation_checked(failure):
    train, held = make_rows()
    if failure == "duplicate_id":
        train[1]["id"] = train[0]["id"]
    elif failure == "held_id_overlap":
        held[0]["id"] = train[0]["id"]
    elif failure == "low_text_overlap":
        train[6]["text"] = held[0]["text"]
    elif failure == "count":
        train.pop()
    else:
        train[6]["meta"]["noise"] = "unknown"
    with pytest.raises(ValueError):
        contract.validate_record_splits(train, held, pool_hi=6, held_per=1)


def test_explicit_loader_full_counts_in_memory(tmp_path, monkeypatch):
    # Real file hashing uses tiny placeholders; JSONL decoding is replaced with
    # 102,000 in-memory synthetic rows so no large pool file or network is needed.
    train, held = make_rows(12000, 400)
    processed = tmp_path / "processed"
    processed.mkdir()
    manifest = make_manifest()
    for name, key in (("qpool_train.jsonl", "train_sha256"), ("qpool_heldout.jsonl", "heldout_sha256")):
        path = processed / name
        path.write_text("synthetic decoding fixture\n")
        manifest[key] = file_sha256(str(path))
    mp = processed / "pool_manifest.json"
    mp.write_text(json.dumps(manifest))
    monkeypatch.setattr(contract, "read_jsonl", lambda p: iter(held if "heldout" in p else train))
    monkeypatch.setenv("OMNISELECT_TEXT_MANIFEST", str(mp))
    pool_records, held_records, path = text_data.load_pool(str(tmp_path))
    assert (len(pool_records), len(held_records)) == (100000, 2000)
    assert path == str(processed / "qpool_train.jsonl")
    assert pool_records[0].id == train[0]["id"]
    (processed / "qpool_heldout.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        text_data.load_pool(str(tmp_path))


def test_source_verification_requires_all_shards_at_exact_revision(tmp_path):
    manifest = make_manifest()
    paths = []
    for shard in manifest["shards"]:
        path = (tmp_path / ("datasets--" + shard["repo"].replace("/", "--")) /
                "snapshots" / shard["rev"] / shard["file"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        shard["bytes"] = path.stat().st_size
        shard["sha256"] = file_sha256(str(path))
        paths.append(path)
    contract.verify_source_shards(manifest, tmp_path)
    paths[-1].unlink()
    with pytest.raises(ValueError, match="missing pinned source shard"):
        contract.verify_source_shards(manifest, tmp_path)


def test_validator_failure_removes_stale_success_marker(tmp_path):
    path = Path(__file__).resolve().parents[2] / "data" / "validate_text_pool.py"
    spec = importlib.util.spec_from_file_location("text_pool_validator_test", path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    marker = tmp_path / "VALIDATED"
    marker.write_text("stale")
    manifest = tmp_path / "bad.json"
    manifest.write_text("{}")
    assert validator.main(["--data-root", str(tmp_path), "--manifest", str(manifest),
                           "--marker", str(marker)]) == 1
    assert not marker.exists()


def test_builder_writes_only_the_explicit_data_root(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "data" / "build_text_pool.py"
    spec = importlib.util.spec_from_file_location("text_pool_builder_test", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    for key, value in {"POOL_HI": 6, "HELD_PER": 1, "N_LOW": 4, "PER": 1}.items():
        monkeypatch.setattr(builder, key, value)
    monkeypatch.setattr(builder, "_texts", lambda repo, cfg, extract, need, skip=0:
                        [f"{repo}/{cfg}/{skip}/{i} " + "x" * 150 for i in range(need)])
    root = tmp_path / "independent-data"
    assert builder.main(["--data-root", str(root)]) == 0
    processed = root / "processed"
    manifest = json.loads((processed / "pool_manifest.json").read_text())
    assert manifest["counts"] == {"train": 50, "heldout": 5}
    assert manifest["train_sha256"] == file_sha256(str(processed / "qpool_train.jsonl"))
    assert manifest["heldout_sha256"] == file_sha256(str(processed / "qpool_heldout.jsonl"))
