"""FAIL-CLOSED server-scale pool builder (audit round, replaces silent-fallback builds).

Contract (validated independently by scripts/validate_qpool.py):
  per domain: train high = POOL_HI, train low = POOL_HI*LOW_FRAC/(1-LOW_FRAC) split
  evenly over four injection kinds; heldout high = HELD_PER. Any source shortfall is a
  HARD ERROR (exit 2) - no backup-pool fallback, no silent under-filling.

Differences from build_quality_pool.py (kept intentionally, all audit findings):
  - IDs: heldout uses its own namespace ({dom}hH{i}) so train/heldout ids can never
    collide; low kinds keep the 10k/20k/30k/40k offsets.
  - general also receives the cross-domain kind (texts from the next domain mislabeled
    as general), so every domain is strictly 60/40.
  - rows come from pinned-revision parquet/json shards via resumable hf_hub_download;
    the manifest records revisions, shard names, sizes and sha256.
  - two-pass build: fetch all domains first (counts checked), inject second.
Env: POOL_HI, HELD_PER, MODALITIES, MAX_SHARDS.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

import numpy as np  # noqa: F401  (parity with sibling builders)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

_spec = importlib.util.spec_from_file_location(
    "build_quality_pool", os.path.join(_REPO, "scripts/build_quality_pool.py"))
bqp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bqp)

from mmdataselect.utils.io import write_jsonl  # noqa: E402

POOL_HI = int(os.environ.get("POOL_HI", "3000"))
HELD_PER = int(os.environ.get("HELD_PER", "400"))
MODALITIES = os.environ.get("MODALITIES", "general,math,code,image,table").split(",")
MAX_SHARDS = int(os.environ.get("MAX_SHARDS", "4"))
LOW_FRAC = 0.40
N_LOW = int(round(POOL_HI * LOW_FRAC / (1 - LOW_FRAC)))
PER = N_LOW // 4
assert PER * 4 == N_LOW, "N_LOW must split evenly over four kinds"

_PREF = {
    "HuggingFaceFW/fineweb-edu": "sample/10BT/",
    "codeparrot/codeparrot-clean-valid": "",
    "yerevann/coco-karpathy": "",
    "mstz/adult": "income/",
}

_MANIFEST = {"revisions": {}, "shards": [], "builder": os.path.basename(__file__),
             "config": {"POOL_HI": POOL_HI, "HELD_PER": HELD_PER, "MODALITIES": MODALITIES,
                        "LOW_FRAC": LOW_FRAC, "MINC": bqp.MINC, "MAXC": bqp.MAXC}}


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _shard_rows(repo, config):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    info = api.repo_info(repo, repo_type="dataset")
    rev = info.sha
    _MANIFEST["revisions"][repo] = rev
    files = [f.rfilename for f in info.siblings]
    pref = _PREF.get(repo)
    if pref is None:
        pref = (config + "/") if config else ""
    cand = [f for f in files if f.startswith(pref)] if pref else list(files)
    pq_files = sorted(f for f in cand if f.endswith(".parquet"))
    js_files = sorted(f for f in cand if f.endswith((".jsonl", ".json.gz", ".jsonl.gz")))
    csv_files = sorted(f for f in cand if f.endswith(".csv"))
    picks = (pq_files or js_files or csv_files)[:MAX_SHARDS]
    if not picks:
        raise RuntimeError(f"FAIL-CLOSED: no readable shards in {repo} (prefix={pref!r})")
    for fname in picks:
        path = hf_hub_download(repo, fname, repo_type="dataset", revision=rev)
        _MANIFEST["shards"].append({"repo": repo, "file": fname, "rev": rev,
                                    "bytes": os.path.getsize(path), "sha256": _sha(path)})
        if fname.endswith(".csv"):
            import pyarrow.csv as pacsv
            table = pacsv.read_csv(path)
            for batch in table.to_batches(max_chunksize=2048):
                yield from batch.to_pylist()
            continue
        if fname.endswith(".parquet"):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=2048):
                yield from batch.to_pylist()
        else:
            import gzip
            op = gzip.open if fname.endswith(".gz") else open
            with op(path, "rt", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except Exception:
                            continue


def _texts(repo, config, extract, need, skip=0):
    out, seen = [], 0
    for ex in _shard_rows(repo, config):
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
        if len(out) >= need:
            return out
    raise RuntimeError(f"FAIL-CLOSED: {repo} yielded {len(out)}/{need} usable texts")


def _held_rec(domain, i, text):
    r = bqp._rec(domain, i, text, "high", "heldout")
    r.id = f"{domain[:2]}hH{i:05d}"          # heldout namespace: never collides with train
    return r


def main() -> int:
    for d in MODALITIES:
        if d not in bqp.HI_SRC:
            print(f"FAIL-CLOSED: unknown modality {d}")
            return 2
    # ---- pass 1: fetch (counts enforced) ----
    hi_all, held_all = {}, {}
    for d in MODALITIES:
        path, cfg, extract = bqp.HI_SRC[d]
        rows = _texts(path, cfg, extract, POOL_HI + HELD_PER)
        held_all[d] = rows[:HELD_PER]
        hi_all[d] = rows[HELD_PER:]
        print(f"[fetch] {d}: hi={len(hi_all[d])} held={len(held_all[d])}")
    # ---- pass 2: records + strict four-kind injection ----
    train, held = [], []
    doms = list(MODALITIES)
    for d in doms:
        for i, t in enumerate(held_all[d]):
            held.append(_held_rec(d, i, t))
        for i, t in enumerate(hi_all[d]):
            train.append(bqp._rec(d, i, t, "high"))
        # 1) truncation
        for i, t in enumerate(hi_all[d][:PER]):
            train.append(bqp._rec(d, 10000 + i, t[: 80 + (i % 70)], "low", "truncation"))
        # 2) template redundancy
        seeds = hi_all[d][PER: PER + max(1, (PER + 11) // 12)]   # ceiling: 42x12=504>=500
        c = 0
        for s_i, seed in enumerate(seeds):
            for r in range(12):
                if c >= PER:
                    break
                train.append(bqp._rec(d, 20000 + c, f"{seed} [v{r}#{s_i}]", "low", "template"))
                c += 1
        if c < PER:
            print(f"FAIL-CLOSED: {d} template kind {c}/{PER}")
            return 2
        # 3) cross-domain: texts of the NEXT domain mislabeled as d (every domain gets it)
        src_dom = doms[(doms.index(d) + 1) % len(doms)]
        src = hi_all[src_dom]
        for i in range(PER):
            train.append(bqp._rec(d, 30000 + i, src[i % len(src)], "low", "crossdomain"))
        # 4) low-tier
        if d in bqp.LOWTIER_SRC:
            lp, lc, lx = bqp.LOWTIER_SRC[d]
            lt = _texts(lp, lc, lx, PER, skip=POOL_HI + HELD_PER)
            for i, t in enumerate(lt):
                train.append(bqp._rec(d, 40000 + i, t, "low", "lowtier"))
        else:
            for i, t in enumerate(hi_all[d][PER: 2 * PER]):
                train.append(bqp._rec(d, 40000 + i, t[:120], "low", "lowtier"))
        print(f"[inject] {d}: hi={POOL_HI} low={N_LOW} held={HELD_PER}")

    train_path = os.path.join(_REPO, "data/processed/qpool_train.jsonl")
    held_path = os.path.join(_REPO, "data/processed/qpool_heldout.jsonl")
    write_jsonl((r.to_dict() for r in train), train_path)
    write_jsonl((r.to_dict() for r in held), held_path)
    _MANIFEST["train_sha256"] = _sha(train_path)
    _MANIFEST["heldout_sha256"] = _sha(held_path)
    _MANIFEST["counts"] = {"train": len(train), "heldout": len(held)}
    mpath = os.path.join(_REPO, "data/processed/pool_manifest.json")
    json.dump(_MANIFEST, open(mpath, "w"), indent=2)
    print(f"BUILD_DONE train={len(train)} held={len(held)} manifest={mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
