"""Versioned R6 source repair. Original 25k and legacy 100k contracts stay closed."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from benchmark.Data import text_manifest as legacy
from omniselect.utils.hashing import file_sha256
from omniselect.utils.io import read_jsonl

SCHEMA = "r6_text100k_adult_v2"
HELD_SHA = "1e4a45c9c959995a3c10c840dc2a8b84ce33bd82ba17682c8f50c3b4b3a1e785"
ADULT_REPO = "mstz/adult"
ADULT_SHA = "a26420446d4a38fc432c4470b37b7eb19937a7a666078aa7604f4b155f796d3b"
UCI_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
EXTERNAL_SOURCES = [
    {"file": "adult.zip", "url": UCI_URL, "bytes": 620237,
     "sha256": "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb"},
    {"file": "adult.data", "archive": "adult.zip", "bytes": 3974305,
     "sha256": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d"},
]
ADULT_KEYS = (
    "age", "capital_gain", "capital_loss", "education", "final_weight",
    "hours_worked_per_week", "marital_status", "native_country", "occupation",
    "race", "relationship", "is_male", "workclass", "over_threshold",
)
ADULT_MAPPING = dict(zip(ADULT_KEYS, (
    "age", "float(capital-gain)", "float(capital-loss)", "education-num", "fnlwgt",
    "hours-per-week", "marital-status", "native-country", "occupation", "race",
    "relationship", "sex == Male", "workclass", "income == >50K",
)))
POLICY = {
    "domains": legacy.DOMAINS,
    "shards": "lexicographic first four per source subset; native row order; record consumed shards",
    "held": "copy registered 2000-row heldout bytes unchanged; exclude globally normalized text",
    "high": "first 12000 unique usable records per domain; globally unique normalized high text",
    "adult": "mstz income/train.csv first; then UCI adult.data in file order; never adult.test",
    "adult_dedup": "typed 14-field canonical tuple; exclude all held Adult keys and earlier accepted keys",
    "serialization": "fixed ADULT_KEYS order; Python scalar strings joined by '. '; whitespace normalized",
    "low": "legacy four-kind recipe, 2000 each; math lowtier skip first 12400 length-valid rows",
    "math_lowtier": "after skip, unique normalized text excluding all high and held text",
    "length": "normalize whitespace; accept high/source math lowtier iff 150 <= characters <= 2000",
}


def norm(text):
    return " ".join(text.split())


def canonical_adult(row):
    legacy._require(set(row) == set(ADULT_KEYS), "Adult fields differ from frozen mapping")
    value = dict(row)
    for name in ("age", "education", "final_weight", "hours_worked_per_week", "over_threshold"):
        value[name] = int(value[name])
    for name in ("capital_gain", "capital_loss"):
        value[name] = float(value[name])
        legacy._require(math.isfinite(value[name]), "nonfinite Adult numeric field")
    male = str(value["is_male"]).lower()
    legacy._require(male in ("true", "false"), "invalid Adult sex indicator")
    value["is_male"] = male == "true"
    legacy._require(value["over_threshold"] in (0, 1), "invalid Adult income indicator")
    return tuple(value[name] for name in ADULT_KEYS)


def uci_adult(row):
    legacy._require(len(row) == 15, "UCI Adult row must have 15 fields")
    legacy._require(row[9] in ("Male", "Female"), "invalid UCI Adult sex")
    legacy._require(row[14] in ("<=50K", ">50K"), "unexpected UCI training income label")
    return canonical_adult(dict(zip(ADULT_KEYS, (
        row[0], row[10], row[11], row[4], row[2], row[12], row[5], row[13],
        row[6], row[8], row[7], row[9] == "Male", row[1], int(row[14] == ">50K"),
    ))))


def adult_text(key):
    return norm(". ".join(f"{name} is {value}" for name, value in zip(ADULT_KEYS, key)))


def adult_text_key(text):
    pairs = [part.split(" is ", 1) for part in text.split(". ")]
    legacy._require(all(len(pair) == 2 for pair in pairs), "invalid serialized Adult text")
    legacy._require([pair[0] for pair in pairs] == list(ADULT_KEYS), "Adult text field order changed")
    return canonical_adult(dict(pairs))


def record(domain, index, text, quality="high", noise=""):
    return {"id": f"{domain[:2]}{quality[0]}{index:05d}", "modality": "text", "domain": domain,
            "text": text, "meta": {"quality": quality, "noise": noise}}


def inject(high, math_low, *, per=2000):
    """Freeze all low records as deterministic derivatives, except audited math lowtier."""
    train = []
    for di, domain in enumerate(legacy.DOMAINS):
        rows = high[domain]
        train.extend(record(domain, i, text) for i, text in enumerate(rows))
        train.extend(record(domain, 10000 + i, text[:80 + i % 70], "low", "truncation")
                     for i, text in enumerate(rows[:per]))
        for i in range(per):
            si, ri = divmod(i, 12)
            train.append(record(domain, 20000 + i, f"{rows[per + si]} [v{ri}#{si}]", "low", "template"))
        other = high[legacy.DOMAINS[(di + 1) % len(legacy.DOMAINS)]]
        train.extend(record(domain, 30000 + i, other[i], "low", "crossdomain") for i in range(per))
        low = math_low if domain == "math" else [text[:120] for text in rows[per:2 * per]]
        train.extend(record(domain, 40000 + i, text, "low", "lowtier") for i, text in enumerate(low))
    return train


def validate_metadata(manifest):
    legacy._require(manifest.get("schema") == SCHEMA, "unknown Adult repair schema")
    legacy.validate_metadata({key: value for key, value in manifest.items() if key != "schema"})
    legacy._require(manifest.get("heldout_sha256") == HELD_SHA, "registered heldout bytes changed")
    legacy._require(manifest.get("external_sources") == EXTERNAL_SOURCES, "UCI source fingerprint changed")
    legacy._require(manifest.get("adult_mapping") == ADULT_MAPPING, "Adult mapping changed")
    legacy._require(manifest.get("policy") == POLICY, "source or deduplication policy changed")
    adult = [shard for shard in manifest["shards"] if shard["repo"] == ADULT_REPO]
    legacy._require(len(adult) == 1 and adult[0]["file"] == "income/train.csv"
                    and adult[0]["sha256"] == ADULT_SHA and adult[0]["bytes"] == 1203621,
                    "Adult primary training source changed")
    legacy._require(re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("selection_sha256", ""))),
                    "missing row-selection evidence sha256")


def validate_rows(train, held, *, pool_hi=12000, held_per=400):
    legacy.validate_record_splits(train, held, pool_hi=pool_hi, held_per=held_per)
    high = {domain: [row["text"] for row in train if row["domain"] == domain
                     and row["meta"]["quality"] == "high"] for domain in legacy.DOMAINS}
    held_text = {norm(row["text"]) for row in held}
    high_text = [norm(text) for rows in high.values() for text in rows]
    legacy._require(len(set(high_text)) == len(high_text), "duplicate normalized high text")
    legacy._require(all(text == norm(text) and 150 <= len(text) <= 2000
                        for rows in high.values() for text in rows), "high length or whitespace changed")
    legacy._require(not {norm(row["text"]) for row in train} & held_text, "normalized train/held overlap")
    held_keys = {adult_text_key(row["text"]) for row in held if row["domain"] == "table"}
    keys = [adult_text_key(text) for text in high["table"]]
    legacy._require(len(set(keys)) == len(keys), "duplicate canonical Adult high record")
    legacy._require(not set(keys) & held_keys, "canonical Adult train/held overlap")
    math_low = [row["text"] for row in train if row["domain"] == "math"
                and row["meta"].get("noise") == "lowtier"]
    legacy._require(all(text == norm(text) and 150 <= len(text) <= 2000 for text in math_low),
                    "math lowtier length or whitespace changed")
    legacy._require(len(set(math_low)) == len(math_low), "duplicate math lowtier text")
    legacy._require(not set(math_low) & (set(high_text) | held_text), "math lowtier overlaps high/held")
    legacy._require(train == inject(high, math_low, per=pool_hi // 6), "record order or injection recipe changed")


def load_pool(data_root, manifest):
    validate_metadata(manifest)
    processed = Path(data_root) / "processed"
    paths = [processed / "qpool_train.jsonl", processed / "qpool_heldout.jsonl"]
    for path, key in zip(paths, ("train_sha256", "heldout_sha256")):
        legacy._require(file_sha256(str(path)) == manifest[key], f"{path.name} sha256 mismatch")
    evidence = processed / "selection_evidence.json"
    legacy._require(file_sha256(str(evidence)) == manifest["selection_sha256"], "selection evidence sha256 mismatch")
    train, held = [list(read_jsonl(str(path))) for path in paths]
    validate_rows(train, held)
    return train, held, str(paths[0])


def verify_external_sources(manifest, root):
    validate_metadata(manifest)
    for source in EXTERNAL_SOURCES:
        path = Path(root) / source["file"]
        legacy._require(path.is_file(), f"missing UCI source {path}")
        legacy._require(path.stat().st_size == source["bytes"], f"UCI source size mismatch: {path}")
        legacy._require(file_sha256(str(path)) == source["sha256"], f"UCI source sha256 mismatch: {path}")


def rows_sha(rows):
    h = hashlib.sha256()
    for row in rows:
        h.update((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    return h.hexdigest()
