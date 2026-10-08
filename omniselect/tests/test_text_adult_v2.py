"""Independent fixtures for source mapping, isolation and versioned contract failures."""
from __future__ import annotations

import copy
import json

import pytest

from benchmark.Data import text_manifest as legacy
from benchmark.Data import text_manifest_adult_v2 as v2
from data import build_text_pool_adult_v2 as builder
from omniselect.tests.test_text_manifest import make_manifest


def uci_row(age=30):
    return [str(age), "Private", "123456", "Bachelors", "13", "Never-married", "Sales",
            "Not-in-family", "White", "Male", "0", "0", "40", "United-States", "<=50K"]


def adult_key(age=30):
    return v2.uci_adult(uci_row(age))


def text(domain, i):
    return (f"{domain} record {i}: " + "A complete source record with informative distinct content. " * 3).strip()


def fixture_rows():
    high = {d: [text(d, i) for i in range(6)] for d in legacy.DOMAINS}
    high["table"] = [v2.adult_text(adult_key(20 + i)) for i in range(6)]
    held = [dict(v2.record(d, 0, v2.adult_text(adult_key(99)) if d == "table" else text(d, "held")),
                 id=d + "held", meta={"quality": "high", "noise": "heldout"}) for d in legacy.DOMAINS]
    return high, held


def test_uci_and_mstz_semantic_mapping_is_exact():
    key = adult_key()
    mstz = dict(zip(v2.ADULT_KEYS, key))
    mstz.update(age="030", capital_gain="0.000", is_male="true")
    assert v2.canonical_adult(mstz) == key
    assert v2.adult_text_key(v2.adult_text(key)) == key
    assert dict(zip(v2.ADULT_KEYS, key))["education"] == 13
    assert dict(zip(v2.ADULT_KEYS, key))["final_weight"] == 123456


def test_test_partition_labels_are_not_accepted():
    row = uci_row()
    row[-1] = "<=50K."
    with pytest.raises(ValueError, match="training income"):
        v2.uci_adult(row)


def test_select_excludes_canonical_held_and_duplicates_in_first_seen_order():
    held = adult_key(99)
    key = adult_key(30)
    candidates = [(v2.adult_text(held), {"row": 0}, held),
                  (v2.adult_text(key), {"row": 1}, key),
                  (v2.adult_text(key).replace("0.0", "0.000"), {"row": 2}, key),
                  (v2.adult_text(adult_key(31)), {"row": 3}, adult_key(31))]
    selected, audit = builder.select(iter(candidates), 2, set(), {held})
    assert selected == [v2.adult_text(key), v2.adult_text(adult_key(31))]
    assert [row["row"] for row in audit["rows"]] == [1, 3]
    assert audit["counts"]["excluded_adult_key"] == 2
    with pytest.raises(ValueError, match="source shortfall"):
        builder.select(iter(candidates), 3, set(), {held})


def test_complete_small_recipe_and_semantic_held_isolation():
    high, held = fixture_rows()
    train = v2.inject(high, [text("math-low", 0)], per=1)
    v2.validate_rows(train, held, pool_hi=6, held_per=1)
    # Same semantic Adult row, distinct byte serialization, bypasses exact-text overlap.
    high["table"][0] = held[-1]["text"].replace("99", "099", 1)
    train = v2.inject(high, [text("math-low", 0)], per=1)
    with pytest.raises(ValueError, match="canonical Adult train/held overlap"):
        v2.validate_rows(train, held, pool_hi=6, held_per=1)


@pytest.mark.parametrize("failure", ["duplicate_high", "injection", "normalized_held", "duplicate_adult"])
def test_new_contract_rejects_bad_rows(failure):
    high, held = fixture_rows()
    if failure == "duplicate_high":
        high["code"][1] = high["code"][0]
    if failure == "duplicate_adult":
        high["table"][1] = high["table"][0].replace("0.0", "0.000")
    train = v2.inject(high, [text("math-low", 0)], per=1)
    if failure == "injection":
        train[6]["text"] = "arbitrary contamination"
    if failure == "normalized_held":
        train[6]["text"] = "  " + held[0]["text"] + "  "
    with pytest.raises(ValueError):
        v2.validate_rows(train, held, pool_hi=6, held_per=1)


def v2_manifest():
    manifest = make_manifest()
    for shard in manifest["shards"]:
        if shard["repo"] == v2.ADULT_REPO:
            shard.update(file="income/train.csv", sha256=v2.ADULT_SHA, bytes=1203621)
    manifest.update(schema=v2.SCHEMA, heldout_sha256=v2.HELD_SHA,
                    external_sources=copy.deepcopy(v2.EXTERNAL_SOURCES),
                    adult_mapping=copy.deepcopy(v2.ADULT_MAPPING), policy=copy.deepcopy(v2.POLICY),
                    selection_sha256="e" * 64)
    return manifest


@pytest.mark.parametrize("failure", ["source", "test_file", "held", "mapping", "policy", "primary", "schema"])
def test_source_and_version_contract_fail_closed(failure):
    manifest = v2_manifest()
    v2.validate_metadata(manifest)
    if failure == "source":
        manifest["external_sources"][1]["sha256"] = "a" * 64
    elif failure == "test_file":
        manifest["external_sources"][1]["file"] = "adult.test"
    elif failure == "held":
        manifest["heldout_sha256"] = "a" * 64
    elif failure == "mapping":
        manifest["adult_mapping"]["education"] = "education"
    elif failure == "policy":
        manifest["policy"]["adult"] = "accept duplicate records"
    elif failure == "primary":
        manifest["shards"][-1]["file"] = "income/test.csv"
    else:
        manifest["schema"] = "unknown"
    with pytest.raises(ValueError):
        v2.validate_metadata(manifest)


def test_new_schema_never_uses_legacy_validator():
    with pytest.raises(ValueError, match="legacy manifest"):
        legacy.validate_metadata(v2_manifest())


def test_external_sources_must_exist_and_match(tmp_path):
    with pytest.raises(ValueError, match="missing UCI source"):
        v2.verify_external_sources(v2_manifest(), tmp_path)
    (tmp_path / "adult.zip").write_bytes(b"wrong zip")
    with pytest.raises(ValueError, match="UCI source size mismatch"):
        v2.verify_external_sources(v2_manifest(), tmp_path)


def test_offline_source_replay_uses_manifest_shards_in_stable_order(tmp_path, monkeypatch):
    manifest = v2_manifest()
    general = manifest["shards"][0]
    path = tmp_path / "datasets--HuggingFaceFW--fineweb-edu" / "snapshots" / general["rev"] / general["file"]
    path.parent.mkdir(parents=True)
    path.write_text("fixture")
    monkeypatch.setattr(builder, "read_shard", lambda p, n: iter([{"text": text("general", 0)}]))
    reader = builder.SourceReader(tmp_path, replay=manifest)
    result = list(reader.rows(general["repo"]))
    assert result[0][1] == {"repo": general["repo"], "file": general["file"], "row": 0}
    assert reader.shards[0]["bytes"] == 7


def test_new_loader_checks_selection_evidence_hash(tmp_path, monkeypatch):
    manifest = v2_manifest()
    processed = tmp_path / "processed"
    processed.mkdir()
    for filename in ("qpool_train.jsonl", "qpool_heldout.jsonl", "selection_evidence.json"):
        (processed / filename).write_text(json.dumps({}))
    monkeypatch.setattr(v2, "file_sha256", lambda p: (manifest["train_sha256"] if "train" in p
                       else manifest["heldout_sha256"] if "heldout" in p else "0" * 64))
    with pytest.raises(ValueError, match="selection evidence sha256"):
        v2.load_pool(tmp_path, manifest)


def test_builder_supplements_real_mapped_rows_and_preserves_recipe(tmp_path):
    high, held = fixture_rows()

    class FakeReader:
        def rows(self, repo, config=None):
            if repo == v2.ADULT_REPO:
                keys = [adult_key(99), adult_key(20), adult_key(20), adult_key(21), adult_key(22)]
                rows = [dict(zip(v2.ADULT_KEYS, key)) for key in keys]
            elif config == "finemath-3plus":
                rows = [{"text": text("lowtier", i)} for i in range(410)]
            else:
                domain = next(d for d, spec in builder.sources.HI_SRC.items() if spec[0] == repo)
                values = [next(r["text"] for r in held if r["domain"] == domain)] + high[domain]
                values.insert(2, high[domain][0])
                field = "content" if domain == "code" else "sentences" if domain == "image" else "text"
                rows = [{field: [value] if domain == "image" else value} for value in values]
            for i, row in enumerate(rows):
                yield row, {"repo": repo, "file": str(config), "row": i}

    # One held row and one prior-source duplicate precede the three genuine additions.
    (tmp_path / "adult.data").write_text("\n".join(", ".join(uci_row(age)) for age in [99, 20, 23, 24, 25]) + "\n")
    train, evidence = builder.build(FakeReader(), held, tmp_path, pool_hi=6)
    v2.validate_rows(train, held, pool_hi=6, held_per=1)
    assert len(train) == 50
    assert [row["repo"] for row in evidence["table"]["rows"]] == [v2.ADULT_REPO] * 3 + ["UCI/Adult"] * 3
    assert evidence["math_lowtier"]["rows"][0]["row"] == 406
    table_high = [row["text"] for row in train if row["domain"] == "table" and row["meta"]["quality"] == "high"]
    assert table_high == high["table"]
