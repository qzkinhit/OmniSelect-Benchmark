"""Convert ImageNet-100 (clane9/imagenet-100 at the pinned revision) to one uint8 npz.

Every image is resized so its shorter side is ``--size`` pixels and center-cropped to a square
of that size (default 128, which leaves room for the 112 px random-resized crop of the native
track). The npz holds Xtr, ytr, Xte, yte (validation split as test) and is written to
data/imagenet100/imagenet100_<size>.npz. With ``--delete-cache`` the Hugging Face cache of the
dataset is removed afterwards. The native track reads this file with --track-set imagenet_source=npz.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile

import numpy as np

REVISION = "0519dc2f402a3a18c6e57f7913db059215eee25b"


def convert(split, size: int) -> tuple[np.ndarray, np.ndarray]:
    """uint8 (N, size, size, 3) images and int64 labels of one split."""
    from PIL import Image

    images = np.empty((len(split), size, size, 3), dtype=np.uint8)
    for i, row in enumerate(split):
        img = row["image"].convert("RGB")
        w, h = img.size
        scale = size / min(w, h)
        img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BILINEAR)
        w, h = img.size
        left, top = (w - size) // 2, (h - size) // 2
        images[i] = np.asarray(img.crop((left, top, left + size, top + size)), dtype=np.uint8)
    return images, np.asarray(split["label"], dtype=np.int64)


def main(argv: list[str] | None = None) -> int:
    """Write the npz and optionally delete the Hugging Face cache of the dataset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenet100"))
    parser.add_argument("--delete-cache", action="store_true")
    args = parser.parse_args(argv)
    from datasets import load_dataset

    train = load_dataset("clane9/imagenet-100", revision=REVISION, split="train")
    test = load_dataset("clane9/imagenet-100", revision=REVISION, split="validation")
    Xtr, ytr = convert(train, args.size)
    Xte, yte = convert(test, args.size)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"imagenet100_{args.size}.npz")
    fd, tmp = tempfile.mkstemp(dir=args.out, suffix=".tmp.npz")
    os.close(fd)
    np.savez(tmp, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    os.replace(tmp, path)
    print(f"wrote {path}: train {Xtr.shape}, test {Xte.shape}")
    if args.delete_cache:
        cache = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "datasets",
                             "clane9___imagenet-100")
        shutil.rmtree(cache, ignore_errors=True)
        hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub",
                           "datasets--clane9--imagenet-100")
        shutil.rmtree(hub, ignore_errors=True)
        print("deleted the Hugging Face cache of clane9/imagenet-100")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
