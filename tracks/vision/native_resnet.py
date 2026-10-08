"""Native vision track: ResNet-18 trained from scratch on CIFAR-10 (32 px) or ImageNet-100 (112 px).

Signals come from shared early checkpoints: ``score_runs`` ResNet-18 inits trained ``score_epoch``
epochs on the whole pool give averaged EL2N and last-layer GraNd scores, penultimate features
and labels. Every selected subset is trained for ``train_epoch`` epochs from one shared
initialization and evaluated on the validation records carved from the train split and on the
dataset's test split. The canonical decision rule is the validation argmax without a margin.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from omniselect.core.datatypes import Modality, UnifiedRecord
from omniselect.core.selection.console import MultiActorConsole
from omniselect.core.signals import InfluenceSignal, RedundancySignal, minmax
from omniselect.core.signals.knn import knn_agreement_and_novelty, l2_normalize
from omniselect.tools.pairing import exact_kmeans_representatives, reset_rng
from tracks.common.downstream import Learner, resolve_device, stage_plan
from tracks.common.experiment import SkipCell
from tracks.common.splits import validation_splits
from tracks.common.task import Signals, TaskData, Track, TrackConfig

CIFAR10_SHA256 = {
    "Xtr": "fce2a08ad3fd21447062517659e31459914c58e1dc9f2b61770cc80bd6771dec",
    "ytr": "50db83ec1958f0e21486f6ce97070d1e2d0a6aa2a166d22f88dbb00827c14830",
    "Xte": "a9cbb34ad57a173871e82bf3f009c1f7fbe200fbcb3490f5372ab6c648bbc175",
    "yte": "348fd210099f44280c59f8bae2951dcb3acf9a65679aedccf26570e90be04379",
}
IMAGENET100_REVISION = "0519dc2f402a3a18c6e57f7913db059215eee25b"
NATIVE_METHODS = ("random", "el2n", "grand", "ccs", "auth_only", "herding", "kcenter", "coreset", "semdedup",
                  "density", "quadmix_pub", "influence_only", "mmdataselect", "dmf_pub", "mmds_adapt")
SMOKE = {"pool_n": 600, "val_n": 200, "test_n": 500, "score_epoch": 1, "score_runs": 1, "train_epoch": 1,
         "batch_size": 128}


@dataclass
class NativeConfig(TrackConfig):
    """Native-protocol constants. ``train_epoch`` defaults to 160, the value of the canonical runs."""

    keep: float = 0.3
    score_epoch: int = 10
    score_runs: int = 3
    train_epoch: int = 160
    in_res: int = 112
    batch_size: int = 256
    lr: float = 0.1
    workers: int = 4
    imagenet_source: str = "hf"       # hf (canonical, full-resolution images) | npz (data/prepare_imagenet100.py)
    imagenet_npz_size: int = 128
    grid_protocols: tuple[str, ...] = ()


class ArrayImages:
    """torch Dataset over uint8 HWC arrays with a transform."""

    def __init__(self, X, y, transform):
        self.X, self.y, self.tf = X, y, transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        from PIL import Image

        return self.tf(Image.fromarray(self.X[i])), int(self.y[i])


class HFImages:
    """torch Dataset over a Hugging Face image dataset with a transform."""

    def __init__(self, ds, transform):
        self.ds, self.tf = ds, transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        r = self.ds[int(i)]
        return self.tf(r["image"].convert("RGB")), int(r["label"])


def load_cifar10(data_root: str):
    """(train, train eval view, test, train labels, provenance) from data/cifar10_np/cifar10.npz."""
    import torchvision.transforms as T

    path = os.path.join(data_root, "cifar10_np", "cifar10.npz")
    if not os.path.isfile(path):
        raise SkipCell(f"missing {path}; run python data/fetch_data.py --only hf")
    with np.load(path) as payload:
        arrays = {name: payload[name] for name in ("Xtr", "ytr", "Xte", "yte")}
    for name, digest in CIFAR10_SHA256.items():
        value = np.ascontiguousarray(arrays[name])
        observed = hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()
        if observed != digest:
            raise ValueError(f"CIFAR-10 array {name} sha256 mismatch")
    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    tf_tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    tf_te = T.Compose([T.ToTensor(), norm])
    return (ArrayImages(arrays["Xtr"], arrays["ytr"], tf_tr), ArrayImages(arrays["Xtr"], arrays["ytr"], tf_te),
            ArrayImages(arrays["Xte"], arrays["yte"], tf_te), np.asarray(arrays["ytr"]), {"cifar10_npz": path})


def load_imagenet100(in_res: int):
    """(train, train eval view, validation-as-test, train labels, provenance) from clane9/imagenet-100."""
    import torchvision.transforms as T
    from datasets import load_dataset

    try:
        dtr = load_dataset("clane9/imagenet-100", revision=IMAGENET100_REVISION, split="train")
        dte = load_dataset("clane9/imagenet-100", revision=IMAGENET100_REVISION, split="validation")
    except (OSError, ConnectionError) as exc:
        raise SkipCell(f"clane9/imagenet-100 unavailable: {exc}") from exc
    norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    tf_tr = T.Compose([T.RandomResizedCrop(in_res, scale=(0.35, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    tf_te = T.Compose([T.Resize(int(in_res * 1.15)), T.CenterCrop(in_res), T.ToTensor(), norm])
    return (HFImages(dtr, tf_tr), HFImages(dtr, tf_te), HFImages(dte, tf_te), np.asarray(dtr["label"]),
            {"imagenet100_revision": IMAGENET100_REVISION})


def load_imagenet100_npz(data_root: str, size: int, in_res: int):
    """ImageNet-100 from the npz of data/prepare_imagenet100.py (images pre-resized to ``size``)."""
    import torchvision.transforms as T

    path = os.path.join(data_root, "imagenet100", f"imagenet100_{size}.npz")
    if not os.path.isfile(path):
        raise SkipCell(f"missing {path}; run python data/prepare_imagenet100.py --size {size}")
    with np.load(path) as z:
        Xtr, ytr, Xte, yte = z["Xtr"], z["ytr"], z["Xte"], z["yte"]
    norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    tf_tr = T.Compose([T.RandomResizedCrop(in_res, scale=(0.35, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    tf_te = T.Compose([T.Resize(int(in_res * 1.15)), T.CenterCrop(in_res), T.ToTensor(), norm])
    return (ArrayImages(Xtr, ytr, tf_tr), ArrayImages(Xtr, ytr, tf_te), ArrayImages(Xte, yte, tf_te), ytr,
            {"imagenet100_npz": path, "imagenet100_revision": IMAGENET100_REVISION})


def make_net(dataset: str, device: str):
    """torchvision ResNet-18. CIFAR uses a 3x3 stem without max pooling."""
    import torch.nn as nn
    from torchvision.models import resnet18

    if dataset == "imagenet100":
        return resnet18(num_classes=100).to(device)
    m = resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m.to(device)


def train_resnet(model, ds, idx, epochs: int, seed: int, device: str, bs: int, lr: float, workers: int):
    """SGD (momentum 0.9, wd 5e-4) with cosine schedule. The loader order is seeded by ``seed``."""
    import torch
    import torch.nn.functional as F

    sub = torch.utils.data.Subset(ds, [int(i) for i in idx])
    g = torch.Generator()
    g.manual_seed(seed)
    dl = torch.utils.data.DataLoader(sub, batch_size=bs, shuffle=True, num_workers=workers, generator=g)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        sch.step()
    return model


def score_checkpoint(model, eval_ds, idx, n_total: int, n_classes: int, device: str):
    """EL2N, last-layer GraNd proxy (EL2N times penultimate norm), features and labels for ``idx``."""
    import torch
    import torch.nn.functional as F

    model.eval()
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(eval_ds, [int(i) for i in idx]), batch_size=256)
    el2n, grand = np.zeros(n_total), np.zeros(n_total)
    feats, labs = np.zeros((n_total, 512), dtype=np.float32), np.zeros(n_total, dtype=int)
    buf = {}
    hook = model.avgpool.register_forward_hook(lambda m, i, o: buf.__setitem__("z", o.flatten(1).detach()))
    pos = 0
    with torch.no_grad():
        for xb, yb in dl:
            p = F.softmax(model(xb.to(device)), 1).cpu().numpy()
            err = np.linalg.norm(p - np.eye(n_classes)[yb.numpy()], axis=1)
            z = buf["z"].cpu().numpy()
            norm = np.linalg.norm(z, axis=1)
            for j, g in enumerate(idx[pos:pos + len(yb)]):
                el2n[g], grand[g], feats[g], labs[g] = err[j], err[j] * max(norm[j], 1e-6), z[j], int(yb[j])
            pos += len(yb)
    hook.remove()
    return el2n, grand, feats, labs


def ccs_native(diff: np.ndarray, keep_n: int, cutoff: float = 0.1, bins: int = 50, seed: int = 0) -> np.ndarray:
    """CCS of the native protocol: equal-width strata with the last bin closed, sparse strata first."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    kept = np.argsort(diff)[: int(n * (1 - cutoff))]
    dd = diff[kept]
    edges = np.linspace(float(dd.min()), float(dd.max()), bins + 1)
    strata = []
    for b in range(bins):
        top = edges[b + 1]
        m = kept[(dd >= edges[b]) & ((dd <= top) if b == bins - 1 else (dd < top))]
        strata.append(m)
    out: list[int] = []
    budget = keep_n
    for i, stratum in enumerate(sorted(strata, key=len)):
        alloc = budget // (len(strata) - i) if (len(strata) - i) else 0
        take = min(len(stratum), alloc)
        if take > 0:
            out += [int(x) for x in rng.permutation(stratum)[:take]]
        budget -= take
    return np.array(out[:keep_n])


class ResNetLearner(Learner):
    """Trains ResNet-18 from the shared initialization. Scores are predictions and softmax probabilities."""

    def __init__(self, plan, tcfg, train_ds, eval_ds, test_ds, pool_global, val_global, splits, base_state, device):
        super().__init__(plan=plan, keep_models=False, name="resnet18_scratch")
        self.tcfg, self.train_ds, self.eval_ds, self.test_ds = tcfg, train_ds, eval_ds, test_ds
        self.pool_global, self.val_global, self.splits = pool_global, val_global, splits
        self.base_state, self.device = base_state, device

    def fit(self, subset, stage):
        reset_rng(self.tcfg.seed, "original-protocol-final-fit")
        model = make_net(self.tcfg.dataset, self.device)
        model.load_state_dict(self.base_state)
        order = self.pool_global[np.asarray(subset, dtype=int)]
        return train_resnet(model, self.train_ds, order, self.fidelity(stage)["epochs"], self.tcfg.seed, self.device,
                            self.tcfg.batch_size, self.tcfg.lr, self.tcfg.workers)

    def score(self, model, split):
        import torch
        import torch.nn.functional as F

        if split == "test":
            ds = self.test_ds
            idx = np.arange(len(ds)) if not self.tcfg.test_n else np.arange(min(self.tcfg.test_n, len(ds)))
        else:
            ds, idx = self.eval_ds, self.val_global[self.splits.get(split)]
        dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, [int(i) for i in idx]), batch_size=256)
        truth, proba = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in dl:
                proba.append(F.softmax(model(xb.to(self.device)), 1).cpu().numpy())
                truth.append(yb.numpy())
        p = np.concatenate(proba)
        y = np.concatenate(truth)
        pred = p.argmax(1)
        return {"target": y, "prediction": pred, "proba": p.astype(np.float32), "classes": np.arange(p.shape[1]),
                "correct": (pred == y).astype(np.float32)}


class NativeTrack(Track):
    """From-scratch ResNet-18 protocol."""

    name = "native"
    utility = "accuracy"
    config_cls = NativeConfig
    default_learner = "resnet18_scratch"
    sorted_training = False
    datasets = {
        "cifar10": {"pool_n": 45000, "val_n": 5000, "test_n": 0, "train_epoch": 160, "methods": NATIVE_METHODS,
                    "smoke": SMOKE},
        "imagenet100": {"pool_n": 120000, "val_n": 5000, "test_n": 0, "train_epoch": 40, "in_res": 112,
                        "methods": NATIVE_METHODS, "smoke": SMOKE},
    }

    def omni_overrides(self, protocol: str, tcfg: NativeConfig) -> dict[str, Any]:
        base = {"cache.by_subset_hash": True, "fidelity.scoring_equals_reported": True,
                "synthesis.coordinate_ascent": False}
        if protocol == "canonical":
            return {**base, "splits.mode": "rank_only", "splits.fractions": (0.0, 1.0, 0.0), "gate.kind": "argmax",
                    "screening.kind": "off", "synthesis.consensus": False}
        return base

    def load(self, tcfg: NativeConfig, ocfg) -> TaskData:
        if tcfg.dataset == "cifar10":
            train, eval_view, test, labels, prov = load_cifar10(tcfg.data_root)
        elif tcfg.imagenet_source == "npz":
            train, eval_view, test, labels, prov = load_imagenet100_npz(tcfg.data_root, tcfg.imagenet_npz_size,
                                                                        tcfg.in_res)
        else:
            train, eval_view, test, labels, prov = load_imagenet100(tcfg.in_res)
        if tcfg.val_n + tcfg.pool_n > len(train):
            raise ValueError("val_n + pool_n exceeds the training split")
        rng = np.random.default_rng(tcfg.seed)
        perm = rng.permutation(len(train))
        val_idx, pool_idx = perm[:tcfg.val_n], perm[tcfg.val_n:tcfg.val_n + tcfg.pool_n]
        n = len(pool_idx)
        keep_n = int(n * tcfg.keep)
        records = [UnifiedRecord(id=str(i), modality=Modality.TEXT, domain="image", text="") for i in range(n)]
        n_test = len(test) if not tcfg.test_n else min(tcfg.test_n, len(test))
        return TaskData(
            pool_ids=[int(i) for i in pool_idx], val_ids=[int(i) for i in val_idx],
            test_ids=[f"test:{i}" for i in range(n_test)], tags=np.array(["high"] * n, dtype=object),
            splits=validation_splits(len(val_idx), tcfg.seed, ocfg), budget=keep_n, utility=self.utility,
            records=records, labels=labels[pool_idx],
            n_classes=100 if tcfg.dataset == "imagenet100" else 10,
            arrays={"train": train, "eval": eval_view, "test": test, "pool_idx": pool_idx, "val_idx": val_idx,
                    "rng": rng, "n_total": len(train)},
            provenance=prov,
        )

    def signals(self, data: TaskData, tcfg: NativeConfig, ocfg) -> Signals:
        a = data.arrays
        device = resolve_device(tcfg.device)
        full = a["pool_idx"]
        pool_sha = hashlib.sha256(np.ascontiguousarray(np.sort(full)).tobytes()).hexdigest()
        # v2.1 (alignment channel) and v2.2 (cleanliness score): the score runs also embed the validation
        # records, and the last run's output layer is kept, in a cache file of its own
        va = bool(ocfg.signals.alignment or ocfg.cooperative.enabled)
        path = os.path.join(tcfg.data_root, "processed",
                            f"native_scores_{tcfg.dataset}_s{tcfg.seed}_e{tcfg.score_epoch}_r{tcfg.score_runs}_"
                            f"p{len(full)}_v{len(a['val_idx'])}_{pool_sha[:12]}{'_va' if va else ''}.npz")
        fc_w = fc_b = None
        if os.path.exists(path):
            with np.load(path) as z:
                el2n, grand, feats, labs = z["el2n"], z["grand"], z["feats"], z["labs"]
                if va:
                    fc_w, fc_b = z["fc_w"], z["fc_b"]
        else:
            import torch

            idx_score = np.concatenate([full, a["val_idx"]]) if va else full
            el2n, grand = np.zeros(a["n_total"]), np.zeros(a["n_total"])
            for run in range(tcfg.score_runs):
                torch.manual_seed(tcfg.seed * 1000 + run)
                model = train_resnet(make_net(tcfg.dataset, device), a["train"], full, tcfg.score_epoch, tcfg.seed,
                                     device, tcfg.batch_size, tcfg.lr, tcfg.workers)
                e_, g_, feats, labs = score_checkpoint(model, a["eval"], idx_score, a["n_total"], data.n_classes,
                                                       device)
                el2n += e_
                grand += g_
            el2n /= tcfg.score_runs
            grand /= tcfg.score_runs
            extra = {}
            if va:
                fc_w = model.fc.weight.detach().float().cpu().numpy()
                fc_b = model.fc.bias.detach().float().cpu().numpy()
                extra = {"fc_w": fc_w, "fc_b": fc_b}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp.npz")
            os.close(fd)
            np.savez(tmp, el2n=el2n, grand=grand, feats=feats, labs=labs, **extra,
                     contract=np.asarray(json.dumps({"dataset": tcfg.dataset, "seed": tcfg.seed,
                                                     "score_epoch": tcfg.score_epoch, "score_runs": tcfg.score_runs,
                                                     "pool_sha256": pool_sha, "validation_embedded": va})))
            os.replace(tmp, path)
        fp = feats[full]
        lp = labs[full]
        auth, _ = knn_agreement_and_novelty(l2_normalize(fp), lp, 15, self_value=-2.0)
        _, redundancy = knn_agreement_and_novelty(l2_normalize(fp), None, 15, self_value=-2.0)
        from sklearn.linear_model import LogisticRegression

        ref_n = min(2000, len(full) // 4)
        # The native protocol has no corruption tag. Both protocols draw a uniform pool sample with
        # default_rng(seed + 91). A V_con reference would need features from an extra forward pass.
        ref_local = np.random.default_rng(tcfg.seed + 91).permutation(len(full))[:ref_n]
        source = "pool_uniform_sample"
        clf = LogisticRegression(max_iter=200, C=1.0).fit(fp[ref_local], lp[ref_local])
        proba = clf.predict_proba(fp)
        column = {c: j for j, c in enumerate(clf.classes_)}
        influence = np.array([np.log(proba[i, column[lp[i]]] + 1e-9) if lp[i] in column else -20.0
                              for i in range(len(full))])
        if getattr(tcfg, "influence_within_class", False):
            from omniselect.core.signals.influence import within_class_rank
            influence = within_class_rank(influence, lp)
        console = MultiActorConsole(
            [("redundancy", RedundancySignal()), ("influence", InfluenceSignal())],
            weights=np.log(np.array([1 - tcfg.w_infl, tcfg.w_infl]) + 1e-9),
            conflict_gate=True, anneal=0.6, group_key="domain", trust_region=0.5, ema_beta=0.3, min_weight=0.02,
        )
        imp_dyn = console.importance(data.records, scores=np.stack([minmax(redundancy), minmax(influence)]),
                                     progress=0.5)
        return Signals(auth=auth.astype(float), influence=influence, redundancy=redundancy.astype(float),
                       features=fp.astype(float), reference_ids=[data.pool_ids[int(i)] for i in ref_local],
                       reference_source=source,
                       extras={"imp_dyn": imp_dyn, "el2n_scores": el2n[full], "grand_scores": grand[full],
                               "coverage_features": fp,
                               "alignment_inputs": ({"feats": feats, "labs": labs, "fc_w": fc_w, "fc_b": fc_b}
                                                    if va else None),
                               "validation_embedding": ((feats[a["val_idx"]], labs[a["val_idx"]]) if va else None)},
                       saved={"el2n": el2n[full], "grand": grand[full]})

    def alignment(self, data: TaskData, sig: Signals, tcfg: NativeConfig, ocfg):
        """Alignment from the output layer of the last score-run ResNet-18 on its penultimate features."""
        from benchmark.Methods._gradients import LastLayer, append_bias, softmax_theta
        from tracks.common.alignment import from_last_layer

        inputs = sig.extras.get("alignment_inputs")
        if inputs is None:
            return None
        a = data.arrays
        full, con_idx = a["pool_idx"], a["val_idx"][data.splits.con]
        theta = softmax_theta(inputs["fc_w"].T, inputs["fc_b"], np.arange(data.n_classes), data.n_classes)
        model = LastLayer("softmax", append_bias(inputs["feats"][full]), np.asarray(inputs["labs"][full]), theta,
                          append_bias(inputs["feats"][con_idx]), np.asarray(inputs["labs"][con_idx]),
                          labels=np.asarray(inputs["labs"][full]), source="ResNet-18 score run output layer")
        return from_last_layer(model, "ResNet-18 score run output layer")

    def cooperative_inputs(self, data: TaskData, sig: Signals, tcfg: NativeConfig, ocfg) -> dict[str, Any]:
        """Cosine geometry of the score-run penultimate features. Cleanliness features from those features and
        the dataset labels. V_con features come from the score runs' embedding of the validation records."""
        con = data.splits.con
        inputs: dict[str, Any] = {"veto_features": sig.features, "metric": "cosine", "selector_features": sig.features}
        embedded = sig.extras.get("validation_embedding")
        if embedded is not None:
            feats_val, labs_val = embedded
            inputs["con_veto_features"] = feats_val[con]
            inputs["cleanliness"] = {"kind": "classification", "pool_X": sig.features, "pool_y": data.labels,
                                     "con_X": feats_val[con], "con_y": labs_val[con]}
        return inputs

    def strategy_overrides(self, data: TaskData, sig: Signals, tcfg: NativeConfig, ocfg) -> dict:
        a = data.arrays
        full = a["pool_idx"]
        local = {int(g): i for i, g in enumerate(full)}
        fp = sig.features
        el2n_s, grand_s = sig.extras["el2n_scores"], sig.extras["grand_scores"]

        def random(ctx, k):
            return [local[int(g)] for g in a["rng"].permutation(full)[:k]]

        from benchmark.Methods._torch_select import kcenter_torch, kmeans_coverage_torch, record, torch_device

        device = torch_device(tcfg.selection_device)

        def coreset(ctx, k):
            fp32 = sig.extras["coverage_features"]
            record(ctx, "coreset", device)
            if device is not None:           # full-batch Lloyd k-means on the device (not mini-batch)
                return kmeans_coverage_torch(fp32, k, seed=tcfg.seed, device=device)
            from sklearn.cluster import MiniBatchKMeans

            km = MiniBatchKMeans(n_clusters=min(k, len(fp32)), n_init=3, batch_size=max(1024, k),
                                 random_state=tcfg.seed).fit(fp32)
            return exact_kmeans_representatives(fp32, km.labels_, km.cluster_centers_, k)

        def kcenter(ctx, k):
            from benchmark.Methods.KCenter.method import kcenter_greedy

            record(ctx, "kcenter", device)
            return kcenter_greedy(fp, k) if device is None else kcenter_torch(fp, k, seed=0, device=device)

        def density(ctx, k):
            from benchmark.Methods.Density.method import density_select

            return density_select(fp, k, knn=10, seed=tcfg.seed)

        def dmf_pub(ctx, k):
            from benchmark.Methods.DMF.method import dmf_published_update
            from sklearn.linear_model import LogisticRegression

            n = len(fp)
            n_hold = max(1, int(0.1 * n))
            hold = np.arange(n - n_hold, n)
            sel_local = np.arange(n - n_hold)
            lp = data.labels
            ch = np.stack([minmax(sig.auth[sel_local]), minmax(sig.influence[sel_local]),
                           minmax(sig.redundancy[sel_local])], axis=0)

            def reward(sub):
                rows = sel_local[np.asarray(list(sub), dtype=int)]
                clf = LogisticRegression(max_iter=150, C=1.0).fit(fp[rows], lp[rows])
                return float((clf.predict(fp[hold]) == lp[hold]).mean())

            picked = dmf_published_update(ch, k, val_reward=reward, rounds=6, seed=tcfg.seed)
            return [int(sel_local[i]) for i in picked]

        return {
            "random": random,
            "el2n": lambda ctx, k: [int(i) for i in np.argsort(-el2n_s, kind="stable")[:k]],
            "grand": lambda ctx, k: [int(i) for i in np.argsort(-grand_s, kind="stable")[:k]],
            "ccs": lambda ctx, k: [int(i) for i in ccs_native(el2n_s, k, seed=tcfg.seed)],
            "coreset": coreset, "kcenter": kcenter, "density": density, "dmf_pub": dmf_pub,
        }

    def learner(self, data: TaskData, tcfg: NativeConfig, ocfg) -> ResNetLearner:
        import torch

        a = data.arrays
        device = resolve_device(tcfg.device)
        torch.manual_seed(tcfg.seed)
        base_state = {k: v.clone() for k, v in make_net(tcfg.dataset, device).state_dict().items()}
        schedule = {"epochs": tcfg.train_epoch}
        plan = stage_plan(ocfg, scoring=schedule, reported=schedule,
                          seeds={"con": tcfg.seed, "rank": tcfg.seed, "report": tcfg.seed}, schedule_key="epochs")
        return ResNetLearner(plan, tcfg, a["train"], a["eval"], a["test"], a["pool_idx"], a["val_idx"], data.splits,
                             base_state, device)

    def save_path(self, cell: Path) -> Path:
        return cell / "model"
