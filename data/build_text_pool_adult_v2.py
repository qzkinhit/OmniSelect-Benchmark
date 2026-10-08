"""Explicit 100k repair builder. Uses only adult.data to supplement the pinned Adult source.

Output must be a new directory. No registered or legacy data file is rewritten.
Offline replay reconstructs every selected row and low-quality injection from verified sources.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from benchmark.Data import text_manifest as legacy  # noqa: E402
from benchmark.Data import text_manifest_adult_v2 as v2  # noqa: E402
from data import text_pool_sources as sources  # noqa: E402
from omniselect.utils.hashing import file_sha256  # noqa: E402
from omniselect.utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402


class SourceReader:
    def __init__(self, hub_root, replay=None):
        self.hub_root = Path(hub_root)
        self.replay = replay
        self.shards = []
        self.revisions = {}

    def rows(self, repo, config=None):
        prefix = (config + "/") if config else legacy.SOURCE_PREFIXES[repo][0]
        rev = legacy.PINNED_REVISIONS[repo]
        if self.replay is not None:
            names = sorted(s["file"] for s in self.replay["shards"]
                           if s["repo"] == repo and s["file"].startswith(prefix))
        else:
            from huggingface_hub import HfApi
            info = HfApi().repo_info(repo, repo_type="dataset", revision=rev)
            legacy._require(info.sha == rev, f"source revision drift for {repo}")
            candidates = [s.rfilename for s in info.siblings if s.rfilename.startswith(prefix)]
            names = (sorted(n for n in candidates if n.endswith(".parquet"))
                     or sorted(n for n in candidates if n.endswith((".jsonl", ".json.gz", ".jsonl.gz")))
                     or sorted(n for n in candidates if n.endswith(".csv")))[:4]
        legacy._require(names, f"no source shards for {repo}/{prefix}")
        for name in names:
            if self.replay is not None:
                path = self.hub_root / ("datasets--" + repo.replace("/", "--")) / "snapshots" / rev / name
            else:
                from huggingface_hub import hf_hub_download
                path = Path(hf_hub_download(repo, name, repo_type="dataset", revision=rev,
                                           cache_dir=str(self.hub_root)))
            shard = {"repo": repo, "file": name, "rev": rev, "bytes": path.stat().st_size,
                     "sha256": file_sha256(str(path))}
            if repo == v2.ADULT_REPO:
                legacy._require(name == "income/train.csv" and shard["sha256"] == v2.ADULT_SHA,
                                "unregistered primary Adult source")
            self.shards.append(shard)
            self.revisions[repo] = rev
            for index, row in enumerate(read_shard(path, name)):
                yield row, {"repo": repo, "file": name, "row": index}


def read_shard(path, name):
    if name.endswith(".csv"):
        import pyarrow.csv as pacsv
        for batch in pacsv.read_csv(path).to_batches(max_chunksize=2048):
            yield from batch.to_pylist()
    elif name.endswith(".parquet"):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            yield from batch.to_pylist()
    else:
        op = gzip.open if name.endswith(".gz") else open
        with op(path, "rt", encoding="utf-8", errors="strict") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def prepare_uci(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "adult.zip"
    if not archive.exists():
        temporary = root / "adult.zip.download"
        with urllib.request.urlopen(v2.UCI_URL, timeout=60) as response, open(temporary, "wb") as output:
            shutil.copyfileobj(response, output)
        legacy._require(file_sha256(str(temporary)) == v2.EXTERNAL_SOURCES[0]["sha256"], "UCI archive sha drift")
        os.replace(temporary, archive)
    legacy._require(file_sha256(str(archive)) == v2.EXTERNAL_SOURCES[0]["sha256"], "UCI archive sha drift")
    data = root / "adult.data"
    if not data.exists():
        with zipfile.ZipFile(archive) as handle:
            raw = handle.read("adult.data")
        legacy._require(hashlib.sha256(raw).hexdigest() == v2.EXTERNAL_SOURCES[1]["sha256"], "UCI data sha drift")
        temporary = root / "adult.data.extract"
        temporary.write_bytes(raw)
        os.replace(temporary, data)
    for expected in v2.EXTERNAL_SOURCES:
        path = root / expected["file"]
        legacy._require(path.stat().st_size == expected["bytes"]
                        and file_sha256(str(path)) == expected["sha256"], "UCI fingerprint mismatch")


def adult_candidates(reader, external_root):
    for row, origin in reader.rows(v2.ADULT_REPO):
        key = v2.adult_text_key(sources._table_row(row))
        yield v2.adult_text(key), origin, key
    with open(Path(external_root) / "adult.data", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.reader(handle, skipinitialspace=True)):
            if not row:
                continue
            key = v2.uci_adult(row)
            yield v2.adult_text(key), {"repo": "UCI/Adult", "file": "adult.data", "row": index}, key


def text_candidates(reader, domain, low=False):
    repo, config, extract = (sources.LOWTIER_SRC if low else sources.HI_SRC)[domain]
    for row, origin in reader.rows(repo, config):
        raw = extract(row)
        if raw:
            yield v2.norm(str(raw)), origin, None


def select(candidates, count, forbidden_text, forbidden_keys, *, skip=0):
    output, evidence = [], []
    stats = {"scanned": 0, "invalid_length": 0, "skipped_prefix": 0,
             "excluded_text": 0, "excluded_adult_key": 0, "selected": 0}
    seen_text, seen_keys = set(forbidden_text), set(forbidden_keys)
    for text, origin, key in candidates:
        stats["scanned"] += 1
        if not 150 <= len(text) <= 2000:
            stats["invalid_length"] += 1
            continue
        if stats["skipped_prefix"] < skip:
            stats["skipped_prefix"] += 1
            continue
        if key is not None and key in seen_keys:
            stats["excluded_adult_key"] += 1
            continue
        if text in seen_text:
            stats["excluded_text"] += 1
            continue
        output.append(text)
        evidence.append({**origin, "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
        seen_text.add(text)
        if key is not None:
            seen_keys.add(key)
        stats["selected"] += 1
        if len(output) == count:
            return output, {"rows": evidence, "counts": stats}
    raise ValueError(f"source shortfall: selected {len(output)}/{count}; counters={stats}")


def build(reader, held, external_root, *, pool_hi=12000):
    forbidden = {v2.norm(row["text"]) for row in held}
    held_keys = {v2.adult_text_key(row["text"]) for row in held if row["domain"] == "table"}
    high, evidence = {}, {}
    for domain in legacy.DOMAINS:
        candidates = (adult_candidates(reader, external_root) if domain == "table"
                      else text_candidates(reader, domain))
        high[domain], evidence[domain] = select(candidates, pool_hi, forbidden, held_keys)
        forbidden.update(high[domain])
        print(f"[high] {domain}: {evidence[domain]['counts']}", flush=True)
    math_low, evidence["math_lowtier"] = select(text_candidates(reader, "math", low=True),
                                               pool_hi // 6, forbidden, (), skip=pool_hi + 400)
    print(f"[math_lowtier] {evidence['math_lowtier']['counts']}", flush=True)
    return v2.inject(high, math_low, per=pool_hi // 6), evidence


def verify_replay(data_root, manifest, hub_root, external_root):
    """Offline source-to-record replay verifies provenance, ordering and all transformations."""
    legacy.verify_source_shards(manifest, hub_root)
    v2.verify_external_sources(manifest, external_root)
    train, held, _ = v2.load_pool(data_root, manifest)
    reader = SourceReader(hub_root, replay=manifest)
    replayed, evidence = build(reader, held, external_root)
    legacy._require(replayed == train and v2.rows_sha(replayed) == manifest["train_sha256"],
                    "source replay differs from frozen train bytes")
    expected = json.loads((Path(data_root) / "processed/selection_evidence.json").read_text())
    legacy._require(evidence == expected, "source replay row evidence differs")
    legacy._require(reader.shards == manifest["shards"], "consumed source shard order differs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--registered-held", required=True)
    parser.add_argument("--hub-root", required=True)
    parser.add_argument("--external-root", required=True)
    args = parser.parse_args(argv)
    held_path = Path(args.registered_held)
    legacy._require(file_sha256(str(held_path)) == v2.HELD_SHA, "registered heldout fingerprint mismatch")
    output = Path(args.data_root) / "processed"
    legacy._require(not output.exists() or not any(output.iterdir()), "output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    prepare_uci(args.external_root)
    reader = SourceReader(args.hub_root)
    held = list(read_jsonl(str(held_path)))
    train, evidence = build(reader, held, args.external_root)
    v2.validate_rows(train, held)
    write_jsonl(train, str(output / "qpool_train.jsonl"))
    shutil.copyfile(held_path, output / "qpool_heldout.jsonl")
    write_json(evidence, str(output / "selection_evidence.json"))
    manifest = {"schema": v2.SCHEMA, "builder": Path(__file__).name,
                "config": legacy.R6_CONFIG, "counts": {"train": len(train), "heldout": len(held)},
                "revisions": reader.revisions, "shards": reader.shards,
                "external_sources": v2.EXTERNAL_SOURCES, "adult_mapping": v2.ADULT_MAPPING,
                "policy": v2.POLICY, "train_sha256": file_sha256(str(output / "qpool_train.jsonl")),
                "heldout_sha256": file_sha256(str(output / "qpool_heldout.jsonl")),
                "selection_sha256": file_sha256(str(output / "selection_evidence.json"))}
    v2.validate_metadata(manifest)
    write_json(manifest, str(output / "pool_manifest.json"))
    print(f"BUILD_DONE {v2.SCHEMA} train={len(train)} held={len(held)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
