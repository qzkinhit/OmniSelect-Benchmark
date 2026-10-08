"""CLIP encoding reproduces the 2026-07 cache rows of 8 CIFAR-100 pool images (cosine >= 0.9999 per row).

The fixture benchmark/Data/fixtures/clip_vitb32_cifar100_ref8.npz holds the images and the rows
Xp[0:8] of vision_cifar100_clip-vit-base-patch32_p4000v800t2000_s0.npz (2026-07-04). The test runs
on CPU, and on CUDA and MPS when present. It is skipped when the pinned CLIP weights are not cached.
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmark.Data import vision as vdata


def _clip_cached() -> bool:
    try:
        from transformers import AutoImageProcessor

        revision = vdata.ENCODER_REVISIONS[vdata.FIXTURE_ENCODER]
        AutoImageProcessor.from_pretrained(vdata.FIXTURE_ENCODER, revision=revision, use_fast=False,
                                           local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 - any loading failure means the weights are absent
        return False


def _devices() -> list[str]:
    import torch

    out = ["cpu"]
    if torch.cuda.is_available():
        out.append("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        out.append("mps")
    return out


@pytest.mark.skipif(not _clip_cached(), reason="pinned CLIP ViT-B/32 weights are not in the local cache")
@pytest.mark.parametrize("device", _devices())
def test_fixture_rows_match_the_2026_07_cache(device):
    images, reference = vdata.fixture_images()
    info: dict = {}
    features = vdata.encode_images(images, vdata.FIXTURE_ENCODER, device, info=info)
    reference = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    cosine = (features.astype(np.float64) * reference).sum(axis=1)
    assert cosine.min() >= 0.9999, cosine
    # the slow processor class is CLIPImageProcessor (transformers 4.x) or CLIPImageProcessorPil (5.x)
    assert info["processor"].startswith("CLIPImageProcessor") and "Fast" not in info["processor"]
    assert info["use_fast"] is False
    assert info["cudnn_allow_tf32"] is False and info["cuda_matmul_allow_tf32"] is False
    check = vdata.verify_encoding(vdata.FIXTURE_ENCODER, device)
    assert check["passed"] and check["reference"] == "2026-07 cache rows"


def test_deterministic_encoding_restores_torch_flags():
    import torch

    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
              torch.get_float32_matmul_precision())
    torch.backends.cudnn.allow_tf32 = True
    with vdata.deterministic_encoding("cpu") as flags:
        assert torch.backends.cudnn.allow_tf32 is False and torch.backends.cuda.matmul.allow_tf32 is False
        assert flags["dtype"] == "float32"
    assert torch.backends.cudnn.allow_tf32 is True
    torch.backends.cudnn.allow_tf32 = before[1]
    assert (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()) == (before[0], before[2])


def test_encoding_record_hashes_features():
    arrays = {"Xp": np.ones((2, 3), np.float32), "Xval": np.zeros((1, 3), np.float32), "Xt": np.ones((1, 3))}
    record = vdata.encoding_record(arrays)
    assert record["settings"] is None and len(record["features_sha256"]) == 64
    arrays["Xp"] = arrays["Xp"] * 2
    assert vdata.encoding_record(arrays)["features_sha256"] != record["features_sha256"]
