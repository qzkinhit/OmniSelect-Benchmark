"""Learner swaps of the frozen-feature vision track: a DINOv2 probe and a from-scratch ResNet-18.

``FeatureProbeLearner`` is the logistic probe of clip_probe.py on another encoder's features (DINOv2
ViT-S/14). ``ResNetScratchLearner`` trains a CIFAR-stem ResNet-18 from one shared initialization on
the selected 32 px pool images (SGD momentum 0.9, weight decay 5e-4, cosine schedule, random crop
and flip) and scores softmax probabilities on the validation and test images.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from tracks.common.downstream import Learner, classification_units

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class FeatureProbeLearner(Learner):
    """LogisticRegression(C=1.0) on precomputed features of the pool, scored per split."""

    def __init__(self, plan, Xp, yp, split_arrays, name: str):
        super().__init__(plan=plan, keep_models=True, name=name)
        self.Xp, self.yp, self.split_arrays = Xp, yp, split_arrays

    def fit(self, subset, stage):
        from sklearn.linear_model import LogisticRegression

        fid = self.fidelity(stage)
        kwargs: dict[str, Any] = {"max_iter": fid["max_iter"], "C": 1.0}
        if fid.get("seed") is not None:
            kwargs["random_state"] = fid["seed"]
        return LogisticRegression(**kwargs).fit(self.Xp[subset], self.yp[subset])

    def score(self, model, split):
        X, y = self.split_arrays[split]
        return classification_units(model, X, y)

    def save(self, model, directory: Path, name: str) -> str:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.pkl"
        with open(path, "wb") as handle:
            pickle.dump(model, handle)
        return str(path)


def cifar_resnet18(n_classes: int):
    """torchvision ResNet-18 with a 3x3 stride-1 stem and no max pooling (32 px inputs)."""
    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(num_classes=int(n_classes))
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    return model


class ResNetScratchLearner(Learner):
    """ResNet-18 from scratch on uint8 HWC images. The initialization and the loader order use the stage seed."""

    def __init__(self, plan, images_pool: np.ndarray, yp: np.ndarray, split_images: dict[str, tuple],
                 n_classes: int, device: str, batch_size: int = 128, lr: float = 0.05, workers: int = 0):
        super().__init__(plan=plan, keep_models=False, name="resnet18_scratch")
        self.images_pool, self.yp, self.split_images = images_pool, np.asarray(yp), split_images
        self.n_classes, self.device = int(n_classes), device
        self.batch_size, self.lr, self.workers = int(batch_size), float(lr), int(workers)

    def _loader(self, images, labels, train: bool, seed: int = 0):
        import torch
        import torchvision.transforms as T

        norm = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm]) if train \
            else T.Compose([T.ToTensor(), norm])

        class Images(torch.utils.data.Dataset):
            def __len__(self):
                return len(images)

            def __getitem__(self, i):
                from PIL import Image

                return tf(Image.fromarray(images[i])), int(labels[i])

        generator = torch.Generator()
        generator.manual_seed(int(seed))
        return torch.utils.data.DataLoader(Images(), batch_size=self.batch_size, shuffle=train,
                                           num_workers=self.workers, generator=generator if train else None)

    def fit(self, subset, stage):
        import torch
        import torch.nn.functional as F

        fid = self.fidelity(stage)
        seed = int(fid["seed"] if fid.get("seed") is not None else 0)
        torch.manual_seed(seed)
        model = cifar_resnet18(self.n_classes).to(self.device)
        idx = np.asarray(subset, dtype=int)
        loader = self._loader(self.images_pool[idx], self.yp[idx], train=True, seed=seed)
        epochs = int(fid["epochs"])
        opt = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                F.cross_entropy(model(xb.to(self.device)), yb.to(self.device)).backward()
                opt.step()
            sched.step()
        return model.eval()

    def score(self, model, split):
        import torch
        import torch.nn.functional as F

        images, labels = self.split_images[split]
        loader = self._loader(images, labels, train=False)
        proba = []
        with torch.no_grad():
            for xb, _ in loader:
                proba.append(F.softmax(model(xb.to(self.device)), 1).cpu().numpy())
        p = np.concatenate(proba) if proba else np.zeros((0, self.n_classes))
        y = np.asarray(labels)
        pred = p.argmax(1)
        return {"target": y, "prediction": pred, "proba": p.astype(np.float32), "classes": np.arange(p.shape[1]),
                "correct": (pred == y).astype(np.float32)}
