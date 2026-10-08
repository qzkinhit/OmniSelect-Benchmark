"""ImageNet-100 npz for the native track from the downloaded parquet shards, without the Arrow cache.

The per-image transform is the one of data/prepare_imagenet100.py: decode, convert to RGB, resize so
the shorter side is ``size`` pixels (PIL bilinear), center-crop a ``size`` square, uint8. Rows are
read in shard order and row order, which is the order of load_dataset on the same revision. The npz
(Xtr, ytr, Xte, yte with the validation split as test) goes to OUT/imagenet100_<size>.npz and a
manifest JSON next to it records the source revision, the shard files with their sha256, the class
names of the parquet schema, the counts, the array sha256 values and the sha256 of the npz file.

    python data/imagenet100_npz.py --snapshot <hf snapshot dir> --out data/imagenet100 --size 256
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REVISION = "0519dc2f402a3a18c6e57f7913db059215eee25b"


def transform(data: bytes, size: int) -> np.ndarray:
    """The transform of data/prepare_imagenet100.py convert() for one encoded image."""
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return np.asarray(img.crop((left, top, left + size, top + size)), dtype=np.uint8)


def shard_images(args: tuple[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """uint8 images and int64 labels of one parquet shard."""
    import pyarrow.parquet as pq

    path, size = args
    table = pq.read_table(path, columns=["image", "label"])
    images = table.column("image").to_pylist()
    labels = np.asarray(table.column("label").to_pylist(), dtype=np.int64)
    out = np.empty((len(images), size, size, 3), dtype=np.uint8)
    for i, cell in enumerate(images):
        out[i] = transform(cell["bytes"], size)
    return out, labels


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def class_names(path: str) -> list[str] | None:
    import pyarrow.parquet as pq

    meta = pq.read_schema(path).metadata or {}
    raw = meta.get(b"huggingface")
    if not raw:
        return None
    feats = json.loads(raw).get("info", {}).get("features", {})
    return (feats.get("label") or {}).get("names")


def split(files: list[str], size: int, workers: int) -> tuple[np.ndarray, np.ndarray]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(shard_images, [(f, size) for f in files]))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--snapshot", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    t0 = time.time()
    train_files = sorted(glob.glob(os.path.join(a.snapshot, "data", "train-*.parquet")))
    val_files = sorted(glob.glob(os.path.join(a.snapshot, "data", "validation-*.parquet")))
    if not train_files or not val_files:
        raise SystemExit(f"missing shards under {a.snapshot}/data")
    Xtr, ytr = split(train_files, a.size, a.workers)
    Xte, yte = split(val_files, a.size, a.workers)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"imagenet100_{a.size}.npz")
    fd, tmp = tempfile.mkstemp(dir=a.out, suffix=".tmp.npz")
    os.close(fd)
    np.savez(tmp, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    os.replace(tmp, path)
    manifest = {
        "source": {"repo": "clane9/imagenet-100", "repo_type": "dataset", "revision": REVISION,
                   "endpoint": os.environ.get("HF_ENDPOINT"),
                   "shards": [{"file": os.path.relpath(f, a.snapshot), "sha256": file_sha256(f),
                               "bytes": os.path.getsize(f)} for f in train_files + val_files]},
        "transform": "data/prepare_imagenet100.py convert(): RGB, shorter side to size (PIL bilinear), "
                     "center crop size x size, uint8; validation split stored as Xte, yte",
        "size": a.size,
        "counts": {"train": int(len(ytr)), "test": int(len(yte)), "classes": int(len(np.unique(ytr)))},
        "class_names": class_names(train_files[0]),
        "array_sha256": {"Xtr": array_sha256(Xtr), "ytr": array_sha256(ytr), "Xte": array_sha256(Xte),
                         "yte": array_sha256(yte)},
        "npz": {"file": os.path.basename(path), "sha256": file_sha256(path), "bytes": os.path.getsize(path)},
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(a.out, f"imagenet100_{a.size}.manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({k: manifest[k] for k in ("counts", "npz", "seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
