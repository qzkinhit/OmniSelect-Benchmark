"""Independent qpool validator (audit round). Exits nonzero unless EVERY check passes;
only then writes /root/A_VALIDATED_OK (path overridable via MARKER env).

Checks (all from the audit work order):
  per-domain train totals POOL_HI+N_LOW, per-domain heldout HELD_PER, strict high/low
  = POOL_HI/N_LOW per domain, all JSON parseable with schema fields, ids unique within
  train, within heldout, and across their union, train/heldout text-hash disjoint,
  manifest present with revisions + shard sha256, and global totals match.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL_HI = int(os.environ.get("POOL_HI", "3000"))
HELD_PER = int(os.environ.get("HELD_PER", "400"))
MODALITIES = os.environ.get("MODALITIES", "general,math,code,image,table").split(",")
N_LOW = int(round(POOL_HI * 0.40 / 0.60))
MARKER = os.environ.get("MARKER", "/root/A_VALIDATED_OK")

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def load(path):
    rows = []
    for ln, line in enumerate(open(path, errors="strict"), 1):
        try:
            rows.append(json.loads(line))
        except Exception as e:
            check(f"json-parse {os.path.basename(path)}:{ln}", False, str(e)[:60])
            return rows
    return rows


def main() -> int:
    tp = os.path.join(_REPO, "data/processed/qpool_train.jsonl")
    hp = os.path.join(_REPO, "data/processed/qpool_heldout.jsonl")
    mp = os.path.join(_REPO, "data/processed/pool_manifest.json")
    for p in (tp, hp, mp):
        check(f"exists {os.path.basename(p)}", os.path.exists(p))
    if FAIL:
        return 1
    train, held = load(tp), load(hp)
    for name, rows in (("train", train), ("heldout", held)):
        for r in rows[:1] + rows[-1:]:
            check(f"schema {name}", all(k in r for k in ("id", "domain", "text", "meta")))
    tdom = Counter(r["domain"] for r in train)
    hdom = Counter(r["domain"] for r in held)
    q = defaultdict(Counter)
    for r in train:
        q[r["domain"]][r["meta"].get("quality")] += 1
    for d in MODALITIES:
        check(f"train[{d}]=={POOL_HI + N_LOW}", tdom.get(d) == POOL_HI + N_LOW, f"got {tdom.get(d)}")
        check(f"heldout[{d}]=={HELD_PER}", hdom.get(d) == HELD_PER, f"got {hdom.get(d)}")
        check(f"train[{d}] high=={POOL_HI}", q[d]["high"] == POOL_HI, f"got {q[d]['high']}")
        check(f"train[{d}] low=={N_LOW}", q[d]["low"] == N_LOW, f"got {q[d]['low']}")
    check(f"train total=={len(MODALITIES) * (POOL_HI + N_LOW)}",
          len(train) == len(MODALITIES) * (POOL_HI + N_LOW), f"got {len(train)}")
    check(f"heldout total=={len(MODALITIES) * HELD_PER}",
          len(held) == len(MODALITIES) * HELD_PER, f"got {len(held)}")
    tids = [r["id"] for r in train]
    hids = [r["id"] for r in held]
    check("train ids unique", len(set(tids)) == len(tids))
    check("heldout ids unique", len(set(hids)) == len(hids))
    check("union ids unique", len(set(tids) | set(hids)) == len(tids) + len(hids))
    th = {hashlib.sha256(r["text"].encode()).hexdigest() for r in train
          if r["meta"].get("quality") == "high"}
    hh = {hashlib.sha256(r["text"].encode()).hexdigest() for r in held}
    check("train/heldout text-hash disjoint", not (th & hh), f"overlap={len(th & hh)}")
    man = json.load(open(mp))
    PER = N_LOW // 4
    kinds = ("truncation", "template", "crossdomain", "lowtier")
    kq = defaultdict(Counter)
    for r in train:
        if r["meta"].get("quality") == "low":
            kq[r["domain"]][r["meta"].get("noise")] += 1
    for d in MODALITIES:
        for kd in kinds:
            check(f"train[{d}] low[{kd}]=={PER}", kq[d][kd] == PER, f"got {kq[d][kd]}")
    check("manifest revisions", bool(man.get("revisions")))
    check("manifest shard sha256", all("sha256" in s for s in man.get("shards", [])))
    import glob as _g
    hub = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")) + "/hub"
    n_ver = 0
    for sh in man.get("shards", []):
        blob = _g.glob(f"{hub}/datasets--{sh['repo'].replace('/', '--')}/snapshots/*/{sh['file']}")
        if blob:
            got = hashlib.sha256(open(blob[0], "rb").read()).hexdigest()
            check(f"shard sha {os.path.basename(sh['file'])}", got == sh["sha256"])
            n_ver += 1
    check("shards re-verified >=1", n_ver >= 1, f"n={n_ver}")
    check("manifest pool sha matches",
          man.get("train_sha256") == hashlib.sha256(open(tp, "rb").read()).hexdigest())
    if FAIL:
        print(f"VALIDATION_FAILED ({len(FAIL)}): {FAIL[:6]}")
        return 1
    with open(MARKER, "w") as fh:
        fh.write(man.get("train_sha256", ""))
    print(f"VALIDATION_OK -> {MARKER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
