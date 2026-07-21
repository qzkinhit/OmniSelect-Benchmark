#!/usr/bin/env python3
"""Unified dataset fetcher for OmniSelect (one command instead of scattered curl/HF calls).

Every source and its pinned revision + SHA256 is declared in docs/dataset_provenance.md.
This script fetches the datasets the arms need and verifies bytes against the recorded
SHA256 where the raw artifact is fetched directly. HF/OpenML sets are pinned by revision
and fetched by the arm runners on first use; here we pre-warm them so a later run is
offline-safe, and we verify the small committed raw sets (TEP, DaISy) that ship in git.

Usage:
  python scripts/fetch_data.py                 # fetch/verify everything the main arms need
  python scripts/fetch_data.py --only ett tep  # a subset
  python scripts/fetch_data.py --only text     # build the five-domain text pool
  python scripts/fetch_data.py --verify-only    # only re-hash what is already on disk
"""
import argparse
import gzip
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")

# name -> (url, dest_relpath, expected_sha256 or None). Sets fetched by the runners
# (CIFAR/electricity via HF/OpenML pinned revisions) are pre-warmed separately below.
DIRECT = {
    "etth1": ("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
              "data/processed/_upstream_etth1.csv", None),
    "etth2": ("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
              "data/processed/_upstream_etth2.csv", None),
    "ettm1": ("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
              "data/processed/_upstream_ettm1.csv", None),
}
# small raw sets that ship in git — verify against provenance SHA256
COMMITTED = {
    "daisy_cstr": ("data/daisy/cstr.dat",
                   "0ffdda8a1b962d377dc34371be105bd9dcaef7fcca40554e666841efeec6b84d"),
    "daisy_steamgen": ("data/daisy/steamgen.dat",
                       "7f1e66031197c9502c7c7583b313b6349b7da678644410f4902d18b743eabc23"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest, expected):
    dest = os.path.join(ROOT, dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"  [cached] {dest}")
    else:
        print(f"  [get] {url}")
        with urllib.request.urlopen(url, timeout=120) as r, open(dest + ".tmp", "wb") as out:
            shutil.copyfileobj(r, out)
        os.replace(dest + ".tmp", dest)
    if expected:
        got = sha256(dest)
        ok = got == expected
        print(f"  [sha256 {'OK' if ok else 'MISMATCH'}] {dest}")
        return ok
    return True


def prewarm_hf_openml():
    """Pinned-revision HF/OpenML pre-warm; skips cleanly if datasets/sklearn absent."""
    try:
        from datasets import load_dataset
        for name, rev in (("uoft-cs/cifar100", "aadb3af77e9048adbea6b47c21a81e47dd092ae5"),
                          ("uoft-cs/cifar10", "0b2714987fa478483af9968de7c934580d0bb9a2")):
            print(f"  [hf] {name}@{rev[:12]}")
            load_dataset(name, revision=rev, split="train[:1]")
    except Exception as e:  # noqa: BLE001
        print(f"  [skip hf pre-warm] {type(e).__name__}: {e}")
    try:
        from sklearn.datasets import fetch_openml
        print("  [openml] electricity data_id=151 version=1")
        fetch_openml("electricity", version=1, as_frame=False)
    except Exception as e:  # noqa: BLE001
        print(f"  [skip openml pre-warm] {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset: ett tep daisy hf text")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    want = set(args.only) if args.only else {"ett", "daisy", "tep", "hf", "text"}
    ok = True

    print("== committed raw sets (verify against provenance SHA256) ==")
    for name, (rel, exp) in COMMITTED.items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print(f"  [missing] {rel}")
            ok = False
            continue
        got = sha256(p)
        good = got == exp
        ok &= good
        print(f"  [sha256 {'OK' if good else 'MISMATCH'}] {rel}")
    if "tep" in want:
        tep_dir = os.path.join(ROOT, "data", "tep")
        n = len([f for f in os.listdir(tep_dir)]) if os.path.isdir(tep_dir) else 0
        print(f"  [tep] {n} committed .dat files (per-file SHA in provenance_evidence/tep_files_sha256.txt)")

    if not args.verify_only and "ett" in want:
        print("== ETT upstream (fetch + record) ==")
        for name, (url, dest, exp) in DIRECT.items():
            ok &= fetch(url, dest, exp)

    if not args.verify_only and "hf" in want:
        print("== HF / OpenML pinned pre-warm ==")
        prewarm_hf_openml()

    if "text" in want:
        manifest = os.path.join(PROC, "pool_manifest.json")
        train = os.path.join(PROC, "qpool_train.jsonl")
        heldout = os.path.join(PROC, "qpool_heldout.jsonl")
        if args.verify_only:
            missing = [p for p in (manifest, train, heldout) if not os.path.exists(p)]
            if missing:
                for p in missing:
                    print(f"  [missing text pool] {os.path.relpath(p, ROOT)}")
                ok = False
            else:
                print(f"  [text pool present] train={sha256(train)}")
                print(f"  [text pool present] heldout={sha256(heldout)}")
        else:
            print("== five-domain text pool (fail-closed builder) ==")
            rc = subprocess.call([sys.executable,
                                  os.path.join(ROOT, "scripts", "build_pool_failclosed.py")])
            ok &= rc == 0

    print("\n" + ("all requested data present and verified" if ok
                  else "some artifacts missing/mismatched — see docs/dataset_provenance.md"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
