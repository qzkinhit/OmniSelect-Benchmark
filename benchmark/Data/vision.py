"""Vision data: pinned CIFAR snapshots from Hugging Face, CLIP embeddings, CIFAR-100N labels.

``split_indices`` reproduces the canonical index draw (default_rng(seed) permutes the train split,
pool then validation, and a second permutation picks test). ``encode_images`` computes
L2-normalized CLIP or DINOv2 image features with the PIL processor, float32 and TF32 disabled, after
``checked_device`` compares an 8-image fixture with the 2026-07 cache rows. ``load_or_encode``
caches features, labels and encoder settings in one npz under data/processed with the canonical
file name. ``cifar100n_labels`` verifies and reads the human labels.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from omniselect.utils.hashing import arrays_sha256, file_sha256

logger = logging.getLogger(__name__)

DATASET_REVISIONS = {
    "uoft-cs/cifar100": "aadb3af77e9048adbea6b47c21a81e47dd092ae5",
    "uoft-cs/cifar10": "0b2714987fa478483af9968de7c934580d0bb9a2",
}
ENCODER_REVISIONS = {"openai/clip-vit-base-patch32": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
                     "facebook/dinov2-small": "ed25f3a31f01632728cabb09d1542f84ab7b0056"}
CIFAR100N_SHA256 = "bd2d80409754d420292d622e15e25248ba21e37d27429efac49c8da723f44394"
CIFAR10N_SHA256 = "873e69c39cb9b5e97fb6bae2d60bb59b38a5cbc31d2b868f7903dcb2b9dd2310"


@dataclass
class VisionArrays:
    """Features, labels and dataset indices of pool, validation and test."""

    Xp: np.ndarray
    Xval: np.ndarray
    Xt: np.ndarray
    yp: np.ndarray
    yval: np.ndarray
    yt: np.ndarray
    pool_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    cache_path: str
    from_cache: bool
    encoding: dict[str, Any] = field(default_factory=dict)


def split_indices(n_train: int, n_test: int, pool_n: int, val_n: int, test_n: int, seed: int):
    """Canonical draw: perm of train -> pool, validation. Perm of test -> test (one rng stream)."""
    if pool_n + val_n > n_train or test_n > n_test:
        raise ValueError(f"split sizes exceed the dataset: pool+val={pool_n + val_n}/{n_train}, test={test_n}/{n_test}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_train)
    return perm[:pool_n], perm[pool_n:pool_n + val_n], rng.permutation(n_test)[:test_n]


def cache_path(data_root: str, dataset: str, encoder: str, pool_n: int, val_n: int, test_n: int, seed: int) -> str:
    """data/processed/vision_<ds>_<encoder>_dr<rev12>_er<rev12>_p<P>v<V>t<T>_s<seed>.npz."""
    dr = DATASET_REVISIONS[dataset][:12]
    er = ENCODER_REVISIONS.get(encoder, "unpinned")[:12]
    name = (f"vision_{dataset.split('/')[-1]}_{encoder.split('/')[-1]}_dr{dr}_er{er}_"
            f"p{pool_n}v{val_n}t{test_n}_s{seed}.npz")
    return os.path.join(data_root, "processed", name)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "clip_vitb32_cifar100_ref8.npz"
FIXTURE_ENCODER = "openai/clip-vit-base-patch32"
MIN_FIXTURE_COSINE = 0.9999


@contextmanager
def deterministic_encoding(device: str) -> Iterator[dict[str, Any]]:
    """Disable TF32 in cuBLAS and cuDNN, request deterministic kernels, restore the flags on exit.

    Yields the flags in force during encoding (recorded in the run record).
    """
    import torch

    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield {"device": device, "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
               "cudnn_deterministic": True, "cudnn_benchmark": False, "float32_matmul_precision": "highest",
               "dtype": "float32"}
    finally:
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.deterministic,
         torch.backends.cudnn.benchmark) = saved[:4]
        torch.set_float32_matmul_precision(saved[4])


def _image_features(model, pixel_values):
    import torch

    if hasattr(model, "get_image_features"):
        out = model.get_image_features(pixel_values=pixel_values)
    else:
        out = model(pixel_values=pixel_values)
    if not torch.is_tensor(out):
        embeds = getattr(out, "image_embeds", None)
        out = embeds if embeds is not None else out.pooler_output
    return out


def encode_images(images: list, encoder: str, device: str, batch: int = 128,
                  info: Optional[dict[str, Any]] = None) -> np.ndarray:
    """L2-normalized image features of a CLIP or DINOv2 encoder at its pinned revision.

    The processor is the slow PIL implementation (use_fast=False), the model runs in float32 with
    TF32 disabled (``deterministic_encoding``). CLIP features are ``get_image_features``, DINOv2
    features are the pooled CLS token. ``info`` receives the processor class, the flags, the
    torch and transformers versions.
    """
    import torch
    import transformers
    from transformers import AutoImageProcessor, AutoModel

    revision = ENCODER_REVISIONS.get(encoder)
    kwargs = {"revision": revision} if revision else {}
    proc = AutoImageProcessor.from_pretrained(encoder, use_fast=False, **kwargs)
    feats = []
    with deterministic_encoding(device) as flags:
        model = AutoModel.from_pretrained(encoder, torch_dtype=torch.float32, **kwargs).to(device).eval()
        with torch.no_grad():
            for start in range(0, len(images), batch):
                inputs = proc(images=images[start:start + batch], return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.float32)
                feats.append(_image_features(model, pixel_values).float().cpu().numpy())
    X = np.concatenate(feats, 0)
    if info is not None:
        info.update({"encoder": encoder, "encoder_revision": revision, "processor": type(proc).__name__,
                     "use_fast": False, **flags, "torch": torch.__version__,
                     "transformers": transformers.__version__})
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def fixture_images() -> tuple[list, np.ndarray]:
    """The 8 CIFAR-100 pool images of seed 0 and their CLIP ViT-B/32 rows from the 2026-07 cache."""
    from PIL import Image

    with np.load(FIXTURE, allow_pickle=False) as z:
        return [Image.fromarray(img) for img in z["images"]], z["reference"].astype(np.float64)


def verify_encoding(encoder: str, device: str) -> dict[str, Any]:
    """Encode the fixture on ``device`` and compare row cosines with the reference rows.

    For CLIP ViT-B/32 the reference is the 2026-07 cache. For any other encoder it is the CPU
    encoding of the same images. ``passed`` is min cosine >= MIN_FIXTURE_COSINE.
    """
    images, reference = fixture_images()
    if encoder != FIXTURE_ENCODER:
        if device == "cpu":
            return {"device": device, "reference": "cpu", "min_cosine": 1.0, "passed": True}
        reference = encode_images(images, encoder, "cpu").astype(np.float64)
    features = encode_images(images, encoder, device).astype(np.float64)
    reference = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    cosine = float((features * reference).sum(axis=1).min())
    return {"device": device, "reference": "2026-07 cache rows" if encoder == FIXTURE_ENCODER else "cpu",
            "min_cosine": cosine, "passed": cosine >= MIN_FIXTURE_COSINE}


def checked_device(encoder: str, device: str) -> tuple[str, dict[str, Any]]:
    """``device`` when its fixture encoding matches the reference, else cpu with a logged warning."""
    check = verify_encoding(encoder, device)
    if check["passed"]:
        return device, check
    logger.warning("encoder %s on %s gives fixture cosine %.6f < %.4f; encoding on cpu instead", encoder, device,
                   check["min_cosine"], MIN_FIXTURE_COSINE)
    return "cpu", {**check, "fallback": "cpu"}


def features_sha256(arrays: dict[str, Any]) -> str:
    """sha256 over dtype, shape and bytes of Xp, Xval and Xt."""
    return arrays_sha256(arrays["Xp"], arrays["Xval"], arrays["Xt"])


def load_or_encode(data_root: str, dataset: str, encoder: str, pool_n: int, val_n: int, test_n: int, seed: int,
                   device: str) -> VisionArrays:
    """Features and labels from the cache, or from the pinned dataset and encoder (then cached).

    A cache written by the 2026-07 runners holds only Xp, Xval and Xt. Its labels are then read
    from the pinned dataset and added to the file.
    """
    path = cache_path(data_root, dataset, encoder, pool_n, val_n, test_n, seed)
    cached: dict[str, Any] = {}
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            cached = {k: z[k] for k in z.files}
    if {"Xp", "Xval", "Xt", "yp", "yval", "yt", "pool_idx", "val_idx", "test_idx"} <= set(cached):
        return VisionArrays(**{k: cached[k] for k in ("Xp", "Xval", "Xt", "yp", "yval", "yt", "pool_idx",
                                                       "val_idx", "test_idx")}, cache_path=path, from_cache=True,
                            encoding=encoding_record(cached))
    from datasets import load_dataset

    revision = DATASET_REVISIONS[dataset]
    train = load_dataset(dataset, split="train", revision=revision)
    test = load_dataset(dataset, split="test", revision=revision)
    pool_idx, val_idx, test_idx = split_indices(len(train), len(test), pool_n, val_n, test_n, seed)
    img_key = "img" if "img" in train.column_names else "image"
    lab_key = "fine_label" if "fine_label" in train.column_names else "label"
    labels_train = np.asarray(train[lab_key])
    labels_test = np.asarray(test[lab_key])
    arrays = {
        "yp": labels_train[pool_idx], "yval": labels_train[val_idx], "yt": labels_test[test_idx],
        "pool_idx": pool_idx, "val_idx": val_idx, "test_idx": test_idx,
    }
    if {"Xp", "Xval", "Xt"} <= set(cached):
        arrays.update({k: cached[k] for k in ("Xp", "Xval", "Xt")})
        if "encoding_json" in cached:
            arrays["encoding_json"] = cached["encoding_json"]
    else:
        device, check = checked_device(encoder, device)
        info: dict[str, Any] = {"fixture_check": check}
        arrays["Xp"] = encode_images([train[int(i)][img_key] for i in pool_idx], encoder, device, info=info)
        arrays["Xval"] = encode_images([train[int(i)][img_key] for i in val_idx], encoder, device)
        arrays["Xt"] = encode_images([test[int(i)][img_key] for i in test_idx], encoder, device)
        arrays["encoding_json"] = np.array(json.dumps(info, sort_keys=True))
    _atomic_savez(path, arrays)
    return VisionArrays(**{k: arrays[k] for k in ("Xp", "Xval", "Xt", "yp", "yval", "yt", "pool_idx", "val_idx",
                                                   "test_idx")}, cache_path=path, from_cache=False,
                        encoding=encoding_record(arrays))


def encoding_record(arrays: dict[str, Any]) -> dict[str, Any]:
    """Encoder settings stored with the cache (None for older caches without them) and the feature sha256."""
    raw = arrays.get("encoding_json")
    settings = json.loads(str(raw)) if raw is not None else None
    return {"settings": settings, "features_sha256": features_sha256(arrays),
            "note": None if settings is not None else "cache written without encoding settings"}


def cifar100n_labels(data_root: str) -> tuple[np.ndarray, np.ndarray, str]:
    """(noisy, clean) CIFAR-100 train labels from data/cifar_n/CIFAR-100_human.pt after a sha256 check."""
    path = os.path.join(data_root, "cifar_n", "CIFAR-100_human.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing {path}; see data/cifar_n/README.md")
    observed = file_sha256(path)
    if observed != CIFAR100N_SHA256:
        raise ValueError(f"CIFAR-100N label sha256 mismatch: {observed}")
    import torch

    payload = torch.load(path, weights_only=False)
    return np.asarray(payload["noisy_label"]), np.asarray(payload["clean_label"]), path


def _atomic_savez(path: str, arrays: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def cifar10n_labels(data_root: str, key: str = "worse_label") -> tuple[np.ndarray, np.ndarray, str]:
    """(noisy, clean) CIFAR-10 train labels of CIFAR-10N (``key``: worse_label, aggre_label, random_label1)."""
    path = os.path.join(data_root, "cifar_n", "CIFAR-10_human.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing {path}; see data/cifar_n/README.md")
    observed = file_sha256(path)
    if observed != CIFAR10N_SHA256:
        raise ValueError(f"CIFAR-10N label sha256 mismatch: {observed}")
    import torch

    payload = torch.load(path, weights_only=False)
    return np.asarray(payload[key]), np.asarray(payload["clean_label"]), path


def human_labels(data_root: str, dataset: str) -> tuple[np.ndarray, np.ndarray, str]:
    """(noisy, clean) human train labels of the dataset: CIFAR-100N noisy_label or CIFAR-10N worse_label."""
    if dataset.endswith("cifar100"):
        return cifar100n_labels(data_root)
    return cifar10n_labels(data_root)


def _train_split(dataset: str):
    from datasets import load_dataset

    return load_dataset(dataset, split="train", revision=DATASET_REVISIONS[dataset])


def superclass_map(dataset: str) -> Optional[np.ndarray]:
    """Fine class -> superclass of CIFAR-100 from the pinned snapshot (None for CIFAR-10)."""
    if not dataset.endswith("cifar100"):
        return None
    train = _train_split(dataset)
    fine = np.asarray(train["fine_label"])
    coarse = np.asarray(train["coarse_label"])
    out = np.full(int(fine.max()) + 1, -1)
    out[fine] = coarse
    return out


def load_images(dataset: str, split: str, indices) -> list:
    """RGB PIL images of the pinned snapshot at ``indices`` of ``split`` (train or test)."""
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split, revision=DATASET_REVISIONS[dataset])
    key = "img" if "img" in ds.column_names else "image"
    return [ds[int(i)][key].convert("RGB") for i in indices]


def corrupted_features(data_root: str, dataset: str, encoder: str, pool_idx: np.ndarray, rows: np.ndarray,
                       mechanism: str, device: str, *, sigma: float = 1.5, quality: int = 20) -> np.ndarray:
    """Features of the pool images ``pool_idx[rows]`` after blur or JPEG, cached in data/processed.

    The cache key is the dataset, the encoder, the mechanism and its parameter, and the sha of the
    train indices, so a record's corrupted feature is encoded once per encoder.
    """
    from tracks.common.injection import blur_image, jpeg_image

    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    train_idx = np.asarray(pool_idx)[rows]
    param = f"s{sigma}" if mechanism == "blur" else f"q{quality}"
    key = arrays_sha256(train_idx)[:12]
    path = os.path.join(data_root, "processed", f"vision_unseen_{dataset.split('/')[-1]}_{encoder.split('/')[-1]}_"
                        f"{mechanism}{param}_{key}.npz")
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            if np.array_equal(z["train_idx"], train_idx):
                return z["X"]
    images = load_images(dataset, "train", train_idx)
    fn = (lambda im: blur_image(im, sigma)) if mechanism == "blur" else (lambda im: jpeg_image(im, quality))
    device, _ = checked_device(encoder, device)
    X = encode_images([fn(im) for im in images], encoder, device).astype(np.float32)
    _atomic_savez(path, {"X": X, "train_idx": train_idx})
    return X


