"""Server-scale pool builder over DOWNLOADED parquet/json shards.

datasets-5.x streaming dies on this link with httpx 'client has been closed'
(5/5 attempts). This builder replaces ONLY the row source: it lists each HI_SRC
repo, downloads one or two shards via hf_hub_download (chunked + resumable, far
more robust than a long-lived stream), and iterates rows locally. Everything
else - extractors, length filter, noise injection, record layout, file writing -
is the original build_quality_pool.main() via a monkeypatched _stream, so the
construction stays bit-for-bit on the audited code path.

Sources whose repo has no readable shard (script-only datasets such as
mstz/adult under datasets 5.x) fall back to the LOCAL backup pool's records of
that domain (built from the very same sources back when scripts still loaded),
with an explicit log line. Env: POOL_HI, HELD_PER, MODALITIES, MAX_SHARDS.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

MAX_SHARDS = int(os.environ.get("MAX_SHARDS", "2"))
_BACKUP = os.path.join(_REPO, "data/processed/local_scale_backup")

_spec = importlib.util.spec_from_file_location(
    "build_quality_pool", os.path.join(_REPO, "scripts/build_quality_pool.py"))
bqp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bqp)

_PREF = {
    "HuggingFaceFW/fineweb-edu": "sample/10BT/",
    "HuggingFaceTB/finemath": None,        # use config as dir prefix
    "codeparrot/codeparrot-clean-valid": "",
    "yerevann/coco-karpathy": "",
    "mstz/adult": "",
}


def _shard_rows(repo, config):
    from huggingface_hub import HfApi, hf_hub_download
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    pref = _PREF.get(repo)
    if pref is None:
        pref = (config + "/") if config else ""
    cand = [f for f in files if f.startswith(pref)] if pref else list(files)
    pq_files = sorted(f for f in cand if f.endswith(".parquet"))
    js_files = sorted(f for f in cand if f.endswith((".jsonl", ".json.gz", ".jsonl.gz")))
    picks = (pq_files or js_files)[:MAX_SHARDS]
    if not picks:
        raise RuntimeError(f"no readable shards in {repo} (prefix={pref!r})")
    for fname in picks:
        path = hf_hub_download(repo, fname, repo_type="dataset")
        if fname.endswith(".parquet"):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=2048):
                yield from batch.to_pylist()
        else:
            op = gzip.open if fname.endswith(".gz") else open
            with op(path, "rt", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except Exception:
                            continue


def _backup_texts(domain, take):
    out = []
    for name in ("qpool_train.jsonl", "qpool_heldout.jsonl"):
        p = os.path.join(_BACKUP, name)
        if not os.path.exists(p):
            continue
        for line in open(p, errors="ignore"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("domain") == domain and r.get("meta", {}).get("quality") == "high":
                out.append(" ".join(str(r.get("text", "")).split())[: bqp.MAXC])
            if len(out) >= take:
                return out
    return out


def _stream_shards(path, config, extract, take, skip=0):
    try:
        rows = _shard_rows(path, config)
        out, seen = [], 0
        for ex in rows:
            try:
                raw = extract(ex)
            except Exception:
                raw = None
            if not raw:
                continue
            t = " ".join(str(raw).split())
            if bqp.MINC <= len(t) <= bqp.MAXC:
                seen += 1
                if seen <= skip:
                    continue
                out.append(t[: bqp.MAXC])
            if len(out) >= take:
                break
        if out:
            return out
        raise RuntimeError(f"shards of {path} yielded 0 usable texts")
    except Exception as e:
        # script-only or unreadable repo: fall back to the backup pool's records
        dom = None
        for d, (p, c, _x) in list(bqp.HI_SRC.items()) + list(bqp.LOWTIER_SRC.items()):
            if p == path and c == config:
                dom = d
                break
        fb = _backup_texts(dom, take) if dom else []
        print(f"[shards] FALLBACK {path} ({e}); backup texts for domain={dom}: {len(fb)}")
        if not fb:
            raise
        return fb


bqp._stream = _stream_shards

if __name__ == "__main__":
    raise SystemExit(bqp.main())
