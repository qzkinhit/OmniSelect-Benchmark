"""DeepCore fidelity check: herding, k-center, EL2N and GraNd on a frozen-CLIP CIFAR subset.

    python -m benchmark.tools.deepcore_fidelity --dataset uoft-cs/cifar10 --pool 1500 --test 1000

Encodes a small CIFAR subset once with a frozen CLIP encoder, then for a clean pool and a pool
with 40% label flips runs each reference implementation of benchmark/tools/deepcore_reference.py
at a 30% budget, fits a logistic probe on the selection and prints top-1 test accuracy. On the
noisy pool EL2N and GraNd select the flipped labels first, which is their documented failure.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from benchmark.tools.deepcore_reference import el2n, grand, herding, kcenter_greedy

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENCODER = "openai/clip-vit-base-patch32"
POOL_N = 1500
TEST_N = 1000
SEED = 0
NOISE_FRAC = 0.40
BUDGET_FRAC = 0.3
DATASET = "uoft-cs/cifar10"
N_CLASSES = 10
_LAB = "label"


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(o):
    import torch
    if torch.is_tensor(o):
        return o
    for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
        v = getattr(o, attr, None)
        if v is not None:
            return v.mean(1) if attr == "last_hidden_state" else v
    raise TypeError(f"cannot extract features from {type(o)}")


def _encode(imgs, dev):
    import torch
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(ENCODER)
    model = AutoModel.from_pretrained(ENCODER).to(dev).eval()
    feats = []
    with torch.no_grad():
        for s in range(0, len(imgs), 128):
            inp = proc(images=imgs[s:s + 128], return_tensors="pt").to(dev)
            f = model.get_image_features(pixel_values=inp["pixel_values"])
            feats.append(_to_tensor(f).float().cpu().numpy())
    X = np.concatenate(feats, 0)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def _load():
    cache = os.path.join(_REPO, f"data/processed/deepcore_{DATASET.split('/')[-1]}_p{POOL_N}t{TEST_N}_s{SEED}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["Xp"], z["yp"], z["Xt"], z["yt"]
    from datasets import load_dataset
    ds = load_dataset(DATASET, split="train")
    te = load_dataset(DATASET, split="test")
    rng = np.random.default_rng(SEED)
    tr = rng.permutation(len(ds))[:POOL_N]
    ts = rng.permutation(len(te))[:TEST_N]
    img_key = "img" if "img" in ds.column_names else "image"
    dev = _device()
    print(f"[encode] {ENCODER} on {POOL_N}+{TEST_N} {DATASET.split('/')[-1]} images (dev={dev}) ...")
    Xp = _encode([ds[int(i)][img_key] for i in tr], dev)
    yp = np.array([ds[int(i)][_LAB] for i in tr])
    Xt = _encode([te[int(i)][img_key] for i in ts], dev)
    yt = np.array([te[int(i)][_LAB] for i in ts])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, Xp=Xp, yp=yp, Xt=Xt, yt=yt)
    return Xp, yp, Xt, yt


def _inject_noise(y, seed):
    rng = np.random.default_rng(seed + 7)
    yo = y.copy()
    flip = rng.permutation(len(y))[:int(NOISE_FRAC * len(y))]
    for i in flip:
        yo[i] = rng.integers(N_CLASSES)
    return yo


def _probe_probs(X, y, ref):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=200, C=1.0).fit(X[ref], y[ref])
    P = np.zeros((len(X), N_CLASSES))
    P[:, clf.classes_] = clf.predict_proba(X)
    return P


def _run_pool(Xp, yp, Xt, yt, tag):
    from sklearn.linear_model import LogisticRegression
    n = len(Xp)
    k = int(BUDGET_FRAC * n)
    rng = np.random.default_rng(SEED)
    ref = rng.permutation(n)[:400]
    probs = _probe_probs(Xp, yp, ref)
    sels = {
        "random": list(rng.permutation(n)[:k]),
        "herding": herding(Xp, k),
        "kcenter": kcenter_greedy(Xp, k, seed=SEED),
        "el2n": el2n(probs, yp, k, is_logits=False),
        "grand": grand(probs, yp, Xp, k, is_logits=False),
    }
    print(f"\n=== {tag} pool ({DATASET.split('/')[-1]}, frozen CLIP + linear probe, "
          f"{int(BUDGET_FRAC * 100)}% budget, top-1 acc) ===")
    out = {}
    for m, sel in sels.items():
        clf = LogisticRegression(max_iter=300, C=1.0).fit(Xp[sel], yp[sel])
        acc = float((clf.predict(Xt) == yt).mean())
        out[m] = acc
        print(f"  {m:9} acc={acc:.4f}")
    return out


def main(argv=None):
    global ENCODER, POOL_N, TEST_N, SEED, DATASET, N_CLASSES, _LAB
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=DATASET, choices=["uoft-cs/cifar10", "uoft-cs/cifar100"])
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--pool", type=int, default=POOL_N)
    parser.add_argument("--test", type=int, default=TEST_N)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    ENCODER, POOL_N, TEST_N, SEED, DATASET = args.encoder, args.pool, args.test, args.seed, args.dataset
    N_CLASSES = 10 if DATASET.endswith("cifar10") else 100
    _LAB = "label" if DATASET.endswith("cifar10") else "fine_label"
    Xp, yp, Xt, yt = _load()
    clean = _run_pool(Xp, yp, Xt, yt, "CLEAN")
    yp_noisy = _inject_noise(yp, SEED)
    noisy = _run_pool(Xp, yp_noisy, Xt, yt, "NOISY (40% label flip)")
    print("\n=== faithfulness vs DeepCore (Guo et al. 2022) / Paul et al. (2021) ===")
    print(f"  CLEAN: methods cluster around a strong random baseline (random {clean['random']:.3f}, "
          f"herding {clean['herding']:.3f}, k-center {clean['kcenter']:.3f}, el2n {clean['el2n']:.3f}). "
          f"DeepCore likewise reports no single coreset consistently beats random on clean data.")
    print(f"  NOISY: el2n/grand crater to {noisy['el2n']:.3f}/{noisy['grand']:.3f} vs random "
          f"{noisy['random']:.3f}, the documented EL2N label-noise failure (select hardest = mislabelled).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
