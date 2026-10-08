#!/usr/bin/env python3
"""Dataset fetcher and checksum verifier for every track.

Sources, pinned revisions and sha256 values are listed in docs/dataset_provenance.md. ETT CSVs
are downloaded at a pinned ETDataset commit and normalized the way the tracks read them. CIFAR,
ImageNet-100 and OpenML Electricity are pre-warmed at pinned revisions. The committed TEP, DaISy
and CIFAR-100N files are verified. The text pool is built by data/build_text_pool.py.

Usage:
  python data/fetch_data.py                   # fetch or verify everything
  python data/fetch_data.py --only ett tep    # a subset: ett tep daisy cifar_n hf imagenet text
  python data/fetch_data.py --verify-only     # only hash what is on disk
"""
import argparse
import hashlib
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
PROC = os.path.join(ROOT, "data", "processed")

# name -> (url, dest_relpath, raw_sha256, consumed_sha256). Sets fetched by the runners
# (CIFAR/electricity via HF/OpenML pinned revisions) are pre-warmed separately below.
ETT_REVISION = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
DIRECT = {
    "etth1": (f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{ETT_REVISION}/ETT-small/ETTh1.csv",
              "data/processed/etth1.csv",
              "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
              "5c155a1b14dcafcdc64f76b86c30637b80c5db42f2454d70da341cd7a8305575"),
    "etth2": (f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{ETT_REVISION}/ETT-small/ETTh2.csv",
              "data/processed/etth2.csv",
              "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
              "14964a31bcfab7cdb8e5499962525fc58c719dc90c41f9a39ddf80f3def72f52"),
    "ettm1": (f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{ETT_REVISION}/ETT-small/ETTm1.csv",
              "data/processed/ettm1.csv",
              "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e",
              "093cc4efd56a6bf68fb20cc93a2a79a4fbb06f02c8f4e7e5efa5520cc68afce6"),
}
# small raw sets that ship in git , verify against provenance SHA256
COMMITTED = {
    "daisy_cstr": ("data/daisy/cstr.dat",
                   "0ffdda8a1b962d377dc34371be105bd9dcaef7fcca40554e666841efeec6b84d"),
    "daisy_steamgen": ("data/daisy/steamgen.dat",
                       "7f1e66031197c9502c7c7583b313b6349b7da678644410f4902d18b743eabc23"),
}
CIFAR10_REVISION = "0b2714987fa478483af9968de7c934580d0bb9a2"
CIFAR10_ARRAY_SHA256 = {
    "Xtr": "fce2a08ad3fd21447062517659e31459914c58e1dc9f2b61770cc80bd6771dec",
    "ytr": "50db83ec1958f0e21486f6ce97070d1e2d0a6aa2a166d22f88dbb00827c14830",
    "Xte": "a9cbb34ad57a173871e82bf3f009c1f7fbe200fbcb3490f5372ab6c648bbc175",
    "yte": "348fd210099f44280c59f8bae2951dcb3acf9a65679aedccf26570e90be04379",
}
CIFAR100N = (
    "data/cifar_n/CIFAR-100_human.pt",
    "bd2d80409754d420292d622e15e25248ba21e37d27429efac49c8da723f44394",
)
OPENML_ELECTRICITY = {
    "id": "151",
    "version": "1",
    "file_id": "2419",
    "md5_checksum": "8ca97867d960ae029ae3a9ac2c923d34",
    "arff_gzip_sha256": (
        "60df61719ff2065ae144ae7788a2a1fca603e6c5bc83c5bd82bb3b63b0d28c74"
    ),
}
TEXT_PRIMARY_SHA256 = {
    "train": "84e174dbb097288c6b4473af2af8d6cb46a0b00a000b6534545725e53f9939c5",
    "heldout": "1e4a45c9c959995a3c10c840dc2a8b84ce33bd82ba17682c8f50c3b4b3a1e785",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(array):
    import numpy as np

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def line_count(path):
    with open(path, "rb") as handle:
        return sum(1 for _ in handle)


def verify_cifar10_npz(path):
    import numpy as np

    if not os.path.isfile(path):
        print(f"  [missing CIFAR-10 NPZ] {os.path.relpath(path, ROOT)}")
        return False
    with np.load(path) as payload:
        if set(payload.files) != set(CIFAR10_ARRAY_SHA256):
            print(f"  [invalid CIFAR-10 NPZ keys] {payload.files}")
            return False
        observed = {
            name: array_sha256(payload[name]) for name in CIFAR10_ARRAY_SHA256
        }
    ok = observed == CIFAR10_ARRAY_SHA256
    print(
        f"  [CIFAR-10 arrays {'SHA256 OK' if ok else 'MISMATCH'}] "
        f"{os.path.relpath(path, ROOT)}"
    )
    return ok


def materialize_cifar10_npz():
    """Build the exact NPZ consumed by the original-protocol CIFAR-10 runner."""
    import numpy as np
    from datasets import load_dataset

    destination = os.path.join(ROOT, "data", "cifar10_np", "cifar10.npz")
    if os.path.isfile(destination):
        return verify_cifar10_npz(destination)

    train = load_dataset(
        "uoft-cs/cifar10", revision=CIFAR10_REVISION, split="train"
    )
    test = load_dataset(
        "uoft-cs/cifar10", revision=CIFAR10_REVISION, split="test"
    )

    def arrays(dataset):
        image_key = "img" if "img" in dataset.column_names else "image"
        images = np.empty((len(dataset), 32, 32, 3), dtype=np.uint8)
        labels = np.asarray(dataset["label"], dtype=np.int64)
        for index, image in enumerate(dataset[image_key]):
            images[index] = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return images, labels

    Xtr, ytr = arrays(train)
    Xte, yte = arrays(test)
    observed = {
        "Xtr": array_sha256(Xtr),
        "ytr": array_sha256(ytr),
        "Xte": array_sha256(Xte),
        "yte": array_sha256(yte),
    }
    if observed != CIFAR10_ARRAY_SHA256:
        print(f"  [CIFAR-10 materialization mismatch] {observed}")
        return False

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=os.path.dirname(destination), suffix=".tmp.npz"
    )
    os.close(fd)
    try:
        np.savez(temporary, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return verify_cifar10_npz(destination)


def fetch_ett(url, dest, raw_expected, consumed_expected):
    """Materialize the exact pandas-normalized CSV consumed by the runners."""
    import io
    import urllib.request

    import pandas as pd

    dest = os.path.join(ROOT, dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"  [cached] {dest}")
    else:
        print(f"  [get] {url}")
        with urllib.request.urlopen(url) as response:
            raw = response.read()
        observed_raw = hashlib.sha256(raw).hexdigest()
        if observed_raw != raw_expected:
            print(
                "  [raw SHA256 MISMATCH] "
                f"expected {raw_expected}, found {observed_raw}"
            )
            return False
        frame = pd.read_csv(io.BytesIO(raw))
        fd, temporary = tempfile.mkstemp(
            dir=os.path.dirname(dest), suffix=".tmp.csv"
        )
        os.close(fd)
        try:
            frame.to_csv(temporary, index=False)
            os.replace(temporary, dest)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    got = sha256(dest)
    ok = got == consumed_expected
    print(f"  [sha256 {'OK' if ok else 'MISMATCH'}] {dest} ({got})")
    return ok


def prewarm_hf_openml(include_imagenet: bool = True):
    """Pinned-revision HF/OpenML pre-warm. Return False on any missing source.

    ``include_imagenet`` adds ImageNet-100 (about 8 GB), which only the native ImageNet-100 cells read.
    """
    ok = True
    try:
        from datasets import load_dataset
        sources = (
            (
                "uoft-cs/cifar100",
                "aadb3af77e9048adbea6b47c21a81e47dd092ae5",
                ("train", "test"),
            ),
            (
                "uoft-cs/cifar10",
                CIFAR10_REVISION,
                ("train", "test"),
            ),
            (
                "clane9/imagenet-100",
                "0519dc2f402a3a18c6e57f7913db059215eee25b",
                ("train", "validation"),
            ),
        )
        for name, rev, splits in sources:
            if name == "clane9/imagenet-100" and not include_imagenet:
                continue
            print(f"  [hf] {name}@{rev[:12]}")
            for split in splits:
                load_dataset(name, revision=rev, split=split)
        ok &= materialize_cifar10_npz()
    except Exception as e:  # noqa: BLE001
        print(f"  [fail hf pre-warm] {type(e).__name__}: {e}")
        ok = False
    try:
        from transformers import AutoModel, AutoProcessor
        name = "openai/clip-vit-base-patch32"
        revision = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
        print(f"  [hf model] {name}@{revision[:12]}")
        AutoProcessor.from_pretrained(name, revision=revision)
        AutoModel.from_pretrained(name, revision=revision)
    except Exception as e:  # noqa: BLE001
        print(f"  [fail hf model pre-warm] {type(e).__name__}: {e}")
        ok = False
    try:
        from sklearn.datasets import fetch_openml, get_data_home
        print("  [openml] electricity data_id=151 version=1")
        electricity = fetch_openml("electricity", version=1, as_frame=False)
        details = electricity.details
        for key in ("id", "version", "file_id", "md5_checksum"):
            if details.get(key) != OPENML_ELECTRICITY[key]:
                raise ValueError(
                    f"OpenML Electricity {key} drifted: "
                    f"{details.get(key)!r} != {OPENML_ELECTRICITY[key]!r}"
                )
        source = os.path.join(
            get_data_home(),
            "openml",
            "openml.org",
            "data",
            "v1",
            "download",
            OPENML_ELECTRICITY["file_id"],
            "electricity.arff.gz",
        )
        if not os.path.isfile(source):
            raise FileNotFoundError(f"OpenML source cache was not materialized: {source}")
        observed = sha256(source)
        if observed != OPENML_ELECTRICITY["arff_gzip_sha256"]:
            raise ValueError(
                "OpenML Electricity ARFF SHA-256 mismatch: "
                f"{observed}"
            )
        print(f"  [openml SHA256 OK] {source}")
    except Exception as e:  # noqa: BLE001
        print(f"  [fail openml pre-warm] {type(e).__name__}: {e}")
        ok = False
    return ok


def verify_tep():
    manifest = os.path.join(
        ROOT, "data", "tep", "SHA256SUMS.txt"
    )
    expected = {}
    with open(manifest, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            digest, name = line.split(None, 1)
            expected[name.strip()] = digest
    ok = len(expected) == 44
    if not ok:
        print(f"  [tep manifest invalid] expected 44 entries, found {len(expected)}")
    for name, digest in sorted(expected.items()):
        path = os.path.join(ROOT, "data", "tep", name)
        good = os.path.isfile(path) and sha256(path) == digest
        if not good:
            print(f"  [tep {'MISSING' if not os.path.isfile(path) else 'MISMATCH'}] {name}")
        ok &= good
    print(f"  [tep {'SHA256 OK' if ok else 'FAILED'}] {len(expected)} files")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset: ett tep daisy cifar_n hf imagenet text (hf covers CIFAR, CLIP and "
                         "Electricity, imagenet adds ImageNet-100)")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    want = (
        {"cifar100n" if name == "cifar_n" else name for name in args.only}
        if args.only
        else {"ett", "daisy", "tep", "hf", "imagenet", "cifar100n", "text"}
    )
    ok = True

    if "daisy" in want:
        print("== committed DaISy sets (verify against provenance SHA256) ==")
        for name, (rel, exp) in COMMITTED.items():
            p = os.path.join(ROOT, rel)
            if not os.path.exists(p) and os.path.exists(p + ".gz"):
                import gzip
                import shutil

                with gzip.open(p + ".gz", "rb") as f_in, open(p, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            if not os.path.exists(p):
                print(f"  [missing] {rel}")
                ok = False
                continue
            got = sha256(p)
            good = got == exp
            ok &= good
            print(f"  [sha256 {'OK' if good else 'MISMATCH'}] {rel}")
    if "tep" in want:
        print("== TEP set (verify all 44 files) ==")
        ok &= verify_tep()

    if "cifar100n" in want:
        print("== CIFAR-100N human labels (manual download + SHA verify) ==")
        relative, expected = CIFAR100N
        path = os.path.join(ROOT, relative)
        good = os.path.isfile(path) and sha256(path) == expected
        if not good:
            print(
                f"  [{'missing' if not os.path.isfile(path) else 'mismatch'}] "
                f"{relative}; obtain it from UCSC-REAL/cifar-10-100n"
            )
        else:
            print(f"  [sha256 OK] {relative}")
        ok &= good

    if "ett" in want:
        print("== ETT consumed CSVs (fetch/materialize + SHA verify) ==")
        for name, (url, dest, raw_exp, consumed_exp) in DIRECT.items():
            path = os.path.join(ROOT, dest)
            if args.verify_only:
                good = (
                    os.path.isfile(path)
                    and sha256(path) == consumed_exp
                )
                print(f"  [{name} {'SHA256 OK' if good else 'MISSING/MISMATCH'}]")
                ok &= good
            else:
                ok &= fetch_ett(url, dest, raw_exp, consumed_exp)

    if "hf" in want or "imagenet" in want:
        if args.verify_only:
            print("== derived CIFAR-10 original-protocol input ==")
            ok &= verify_cifar10_npz(
                os.path.join(ROOT, "data", "cifar10_np", "cifar10.npz")
            )
        else:
            print("== HF / OpenML pinned pre-warm ==")
            ok &= prewarm_hf_openml(include_imagenet="imagenet" in want)

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
                try:
                    import json

                    with open(manifest, encoding="utf-8") as handle:
                        pool_manifest = json.load(handle)
                    observed = {
                        "train": sha256(train),
                        "heldout": sha256(heldout),
                    }
                    manifest_hashes = {
                        "train": pool_manifest.get("train_sha256"),
                        "heldout": pool_manifest.get("heldout_sha256"),
                    }
                    manifest_counts = pool_manifest.get("counts") or {}
                    observed_counts = {
                        "train": line_count(train),
                        "heldout": line_count(heldout),
                    }
                    expected_counts = {"train": 25_000, "heldout": 2_000}
                    good = (
                        observed == TEXT_PRIMARY_SHA256
                        and manifest_hashes == TEXT_PRIMARY_SHA256
                        and observed_counts == expected_counts
                        and manifest_counts == expected_counts
                    )
                    print(
                        "  [text pool "
                        f"{'SHA256 OK' if good else 'MANIFEST MISMATCH'}] "
                        f"train={observed['train']} heldout={observed['heldout']} "
                        f"counts={observed_counts}"
                    )
                    ok &= good
                except (OSError, ValueError, TypeError) as exc:
                    print(
                        "  [invalid text pool manifest] "
                        f"{type(exc).__name__}: {exc}"
                    )
                    ok = False
        else:
            print("== five-domain text pool (fail-closed builder) ==")
            rc = subprocess.call([sys.executable,
                                  os.path.join(ROOT, "data", "build_text_pool.py")])
            ok &= rc == 0

    print("\n" + ("all requested data present and verified" if ok
                  else "some artifacts are missing or mismatched, see docs/dataset_provenance.md"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
