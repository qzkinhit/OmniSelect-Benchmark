"""Vision track: frozen CLIP ViT-B/32 embeddings and a logistic-regression probe (CIFAR-100, CIFAR-100N, CIFAR-10).

The pool is drawn from the pinned CIFAR train split, validation from the same split, test from
the test split. Label noise is injected (flip, duplicate, hard, or the unseen blur, JPEG and
class-dependent flips) or read from CIFAR-100N. The probe is scored with 150 iterations and reported
with 300 under the canonical protocol. V2 uses 300 everywhere. Signals are kNN label agreement,
probe log-likelihood influence and kNN novelty on CLIP features. The learner is the CLIP probe,
a probe on DINOv2 ViT-S/14 features, or a ResNet-18 trained from scratch on the images.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.Data import vision as vdata
from omniselect.core.datatypes import Modality, UnifiedRecord
from omniselect.core.selection.console import MultiActorConsole
from omniselect.core.signals import InfluenceSignal, RedundancySignal, minmax
from omniselect.core.signals.influence import probe_log_likelihood
from omniselect.core.signals.knn import knn_agreement_and_novelty
from tracks.common.downstream import Learner, classification_units, resolve_device, stage_plan
from tracks.common.experiment import SkipCell
from tracks.common.injection import (
    CIFAR10_CLASS_MAP,
    contaminate_validation,
    duplicate_embeddings,
    inject_vision_labels,
    unseen_vision_labels,
)
from tracks.common.pairing import fit_seed
from tracks.common.splits import validation_splits
from tracks.common.task import Signals, TaskData, Track, TrackConfig

VISION_METHODS = ("full", "random", "coreset", "auth_only", "herding", "el2n", "grand", "ccs", "density",
                  "quadmix_pub", "dmf_pub", "influence_only", "mmdataselect", "mmds_adapt")
SMOKE = {"pool_n": 300, "val_n": 160, "test_n": 200, "dsdm_runs": 4, "dmf_rounds": 2,
         "grid_lam": (0.0, 0.6), "influence_ref_n": 100, "resnet_epochs": 1, "resnet_screen_epochs": 1}
LEARNERS = ("clip_vitb32", "dinov2_vits14", "resnet18_scratch")


@dataclass
class VisionConfig(TrackConfig):
    """Vision constants: dataset snapshot, encoder, noise source and probe schedules."""

    hf_dataset: str = "uoft-cs/cifar100"
    encoder: str = "openai/clip-vit-base-patch32"
    vis_noise: str = "inject"         # inject | real (CIFAR-100N human labels)
    scoring_max_iter: int = 150
    reported_max_iter: int = 300
    screen_max_iter: int = 60         # low-fidelity screening schedule
    learner_encoder: str = "facebook/dinov2-small"   # encoder of learner dinov2_vits14
    last_layer_warm: int = 100        # GLISTER and GRAD-MATCH: warm-start iterations of the logistic head
    resnet_epochs: int = 60           # learner resnet18_scratch
    resnet_screen_epochs: int = 15
    resnet_batch: int = 128
    resnet_lr: float = 0.05
    blur_sigma: float = 1.5           # unseen mechanism: Gaussian blur standard deviation (pixels)
    jpeg_quality: int = 20            # unseen mechanism: JPEG quality


class ProbeLearner(Learner):
    """LogisticRegression(C=1.0) on pool embeddings. Per-unit outputs on each split."""

    def __init__(self, plan, Xp, yp, split_arrays):
        super().__init__(plan=plan, keep_models=True, name="clip_vitb32")
        self.Xp, self.yp, self.split_arrays = Xp, yp, split_arrays

    def fit(self, subset, stage):
        from sklearn.linear_model import LogisticRegression

        fid = self.fidelity(stage)
        kwargs = {"max_iter": fid["max_iter"], "C": 1.0}
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


def _last_layer_probe(data: TaskData, tcfg: VisionConfig, prefix: str | None = None):
    """Callable building a last layer: a logistic head warm-started on the whole pool.

    The head is LogisticRegression(C=1, max_iter=``last_layer_warm``) on the learner's features
    (GLISTER, GRAD-MATCH) or, with ``prefix=""``, on the CLIP features of the signals (the alignment
    channel), with the observed labels. V_con features and labels are the validation side.
    """
    cache: dict = {}

    def build():
        if "model" not in cache:
            from sklearn.linear_model import LogisticRegression

            from benchmark.Methods._gradients import LastLayer, append_bias, softmax_theta

            a = data.arrays
            pre = prefix if prefix is not None else ("learner_" if tcfg.learner == "dinov2_vits14" else "")
            Xp, Xval = a[pre + "Xp"], a[pre + "Xval"]
            con = data.splits.con
            head = LogisticRegression(C=1.0, max_iter=tcfg.last_layer_warm, random_state=tcfg.seed)
            head.fit(Xp, data.labels)
            theta = softmax_theta(head.coef_.T, head.intercept_, head.classes_, data.n_classes)
            cache["model"] = LastLayer("softmax", append_bias(Xp), np.asarray(data.labels), theta,
                                       append_bias(Xval[con]), np.asarray(a["yval"][con]),
                                       labels=np.asarray(data.labels),
                                       source="logistic head on " + ("learner features" if pre else "CLIP features"))
        return cache["model"]

    return build


class VisionTrack(Track):
    """Frozen-CLIP probe tasks."""

    name = "vision"
    utility = "accuracy"
    config_cls = VisionConfig
    default_learner = "clip_vitb32"
    datasets = {
        "cifar100": {"hf_dataset": "uoft-cs/cifar100", "vis_noise": "inject", "pool_n": 4000, "val_n": 800,
                     "test_n": 2000, "budget_frac": 0.5, "dsdm_runs": 20, "dmf_rounds": 6,
                     "methods": VISION_METHODS, "smoke": SMOKE},
        "cifar100n": {"hf_dataset": "uoft-cs/cifar100", "vis_noise": "real", "pool_n": 4000, "val_n": 800,
                      "test_n": 2000, "budget_frac": 0.5, "dsdm_runs": 20, "dmf_rounds": 6,
                      "methods": VISION_METHODS, "smoke": SMOKE},
        "cifar10_clip": {"hf_dataset": "uoft-cs/cifar10", "vis_noise": "inject", "pool_n": 4000, "val_n": 800,
                         "test_n": 2000, "budget_frac": 0.5, "dsdm_runs": 20, "dmf_rounds": 6,
                         "methods": VISION_METHODS, "smoke": SMOKE},
    }

    def load(self, tcfg: VisionConfig, ocfg) -> TaskData:
        if tcfg.learner not in LEARNERS:
            raise ValueError(f"vision learner must be one of {LEARNERS}, got {tcfg.learner!r}")
        rb = ocfg.robustness
        if rb.mechanism_set == "unseen" and tcfg.vis_noise == "real":
            raise NotImplementedError("mechanism_set=unseen needs injected noise; cifar100n uses human labels")
        device = resolve_device(tcfg.device)
        try:
            arr = vdata.load_or_encode(tcfg.data_root, tcfg.hf_dataset, tcfg.encoder, tcfg.pool_n, tcfg.val_n,
                                       tcfg.test_n, tcfg.seed, device)
        except (OSError, ConnectionError, ImportError) as exc:
            raise SkipCell(f"dataset {tcfg.hf_dataset} or encoder {tcfg.encoder} unavailable: {exc}") from exc
        n_classes_data = 100 if tcfg.hf_dataset.endswith("cifar100") else 10
        provenance: dict[str, Any] = {"hf_dataset": tcfg.hf_dataset,
                                      "dataset_revision": vdata.DATASET_REVISIONS[tcfg.hf_dataset],
                                      "encoder": tcfg.encoder,
                                      "encoder_revision": vdata.ENCODER_REVISIONS.get(tcfg.encoder),
                                      "embedding_cache": arr.cache_path, "cache_hit": arr.from_cache,
                                      "encoding": arr.encoding}
        natural = None
        if rb.val_noise_kind == "natural":
            noisy, clean, path = vdata.human_labels(tcfg.data_root, tcfg.hf_dataset)
            if not np.array_equal(clean[arr.val_idx], arr.yval):
                raise ValueError("human-label alignment failed on the validation indices")
            natural = noisy[arr.val_idx]
            provenance["validation_human_labels"] = path
        yval, val_pos, val_noise = contaminate_validation(
            rb.val_noise_kind, arr.yval, rb.val_noise_rate, tcfg.seed,
            modulus=int(max(arr.yp.max(), arr.yval.max()) + 1), max_offset=10, natural=natural)
        provenance["val_noise"] = val_noise
        noise = tcfg.noise_frac if rb.injection_ratio < 0 else rb.injection_ratio
        plan = None
        if tcfg.vis_noise == "real":
            noisy, clean, path = vdata.cifar100n_labels(tcfg.data_root)
            pool_idx = np.random.default_rng(tcfg.seed).permutation(len(clean))[:tcfg.pool_n]
            if not (np.array_equal(pool_idx, arr.pool_idx) and np.array_equal(arr.yp, clean[pool_idx])):
                raise ValueError("CIFAR-100N alignment failed")
            obs = noisy[pool_idx].astype(arr.yp.dtype)
            tags = np.where(obs != arr.yp, "flip", "high").astype(object)
            dup_idx = np.array([], dtype=int)
            provenance["cifar100n_labels"] = path
        elif rb.mechanism_set == "unseen":
            obs, tags, plan = unseen_vision_labels(
                arr.yp, tcfg.seed, noise, n_classes_data, superclass=vdata.superclass_map(tcfg.hf_dataset),
                class_map=None if n_classes_data == 100 else CIFAR10_CLASS_MAP)
            dup_idx = np.array([], dtype=int)
            provenance["unseen"] = {"blur_sigma": tcfg.blur_sigma, "jpeg_quality": tcfg.jpeg_quality,
                                    "counts": {k: int(len(v)) for k, v in plan.groups.items()}}
        else:
            flip_range = 100 if ocfg.fixes.vision_flip_range == "canonical_100" else n_classes_data
            obs, tags, dup_idx = inject_vision_labels(arr.yp, tcfg.seed, noise, n_classes=flip_range)
        Xp = self._corrupt(duplicate_embeddings(arr.Xp, dup_idx), plan, tcfg, tcfg.encoder, arr.pool_idx, device)
        n = len(Xp)
        budget = int(tcfg.budget_frac * n)
        splits = validation_splits(len(yval), tcfg.seed, ocfg)
        records = [UnifiedRecord(id=str(i), modality=Modality.TEXT, domain="image", text="") for i in range(n)]
        arrays = {"Xp": Xp, "Xval": arr.Xval[val_pos], "yval": yval, "Xt": arr.Xt, "yt": arr.yt}
        if tcfg.learner == "dinov2_vits14":
            feats = vdata.load_or_encode(tcfg.data_root, tcfg.hf_dataset, tcfg.learner_encoder, tcfg.pool_n,
                                         tcfg.val_n, tcfg.test_n, tcfg.seed, device)
            arrays.update({"learner_Xp": self._corrupt(duplicate_embeddings(feats.Xp, dup_idx), plan, tcfg,
                                                       tcfg.learner_encoder, arr.pool_idx, device),
                           "learner_Xval": feats.Xval[val_pos], "learner_Xt": feats.Xt})
            provenance["learner_encoding"] = feats.encoding
        elif tcfg.learner == "resnet18_scratch":
            arrays.update(self._images(tcfg, arr, val_pos, dup_idx, plan))
        return TaskData(
            pool_ids=[f"train:{int(i)}" for i in arr.pool_idx],
            val_ids=[f"train:{int(arr.val_idx[p])}" for p in val_pos],
            test_ids=[f"test:{int(i)}" for i in arr.test_idx], tags=tags, splits=splits, budget=budget,
            utility=self.utility, records=records, labels=obs, labels_clean=arr.yp, n_classes=n_classes_data,
            arrays=arrays, provenance=provenance,
            mechanisms=[str(t) if t != "high" else "" for t in tags] if plan is not None else None,
        )

    @staticmethod
    def _corrupt(X: np.ndarray, plan, tcfg: VisionConfig, encoder: str, pool_idx: np.ndarray, device: str):
        """Replace the rows of blurred and JPEG-compressed pool records by the features of the corrupted images."""
        if plan is None:
            return X
        X = X.copy()
        for mechanism in ("blur", "jpeg"):
            rows = plan.groups.get(mechanism, np.array([], dtype=np.int64))
            if len(rows):
                X[rows] = vdata.corrupted_features(tcfg.data_root, tcfg.hf_dataset, encoder, pool_idx, rows,
                                                   mechanism, device, sigma=tcfg.blur_sigma,
                                                   quality=tcfg.jpeg_quality)
        return X

    @staticmethod
    def _images(tcfg: VisionConfig, arr, val_pos: np.ndarray, dup_idx: np.ndarray, plan) -> dict[str, np.ndarray]:
        """uint8 images of pool (duplicates copy their source image, blur and JPEG applied), validation and test."""
        from tracks.common.injection import blur_image, jpeg_image

        def as_array(images):
            return np.stack([np.asarray(im, dtype=np.uint8) for im in images])

        pool = vdata.load_images(tcfg.hf_dataset, "train", arr.pool_idx)
        if len(dup_idx):
            seeds = dup_idx[: max(1, len(dup_idx) // 8)]
            for j, i in enumerate(dup_idx):
                pool[int(i)] = pool[int(seeds[j % len(seeds)])]
        if plan is not None:
            for i in plan.groups.get("blur", []):
                pool[int(i)] = blur_image(pool[int(i)], tcfg.blur_sigma)
            for i in plan.groups.get("jpeg", []):
                pool[int(i)] = jpeg_image(pool[int(i)], tcfg.jpeg_quality)
        val = vdata.load_images(tcfg.hf_dataset, "train", arr.val_idx)
        test = vdata.load_images(tcfg.hf_dataset, "test", arr.test_idx)
        return {"images_p": as_array(pool), "images_val": as_array(val)[val_pos], "images_t": as_array(test)}

    def signals(self, data: TaskData, tcfg: VisionConfig, ocfg) -> Signals:
        a = data.arrays
        Xp, obs = a["Xp"], data.labels
        auth, redundancy = knn_agreement_and_novelty(Xp, obs, tcfg.knn)
        if ocfg.influence.reference == "v_con":
            con = data.splits.con[: min(len(data.splits.con), tcfg.influence_ref_n)]
            ref_X, ref_y = a["Xval"][con], a["yval"][con]
            ref_ids = [data.val_ids[int(i)] for i in con]
            source = "v_con"
        else:
            rng = np.random.default_rng(tcfg.seed)
            ref = rng.permutation(np.where(data.tags == "high")[0])[: tcfg.influence_ref_n]
            ref_X, ref_y = Xp[ref], obs[ref]
            ref_ids = [data.pool_ids[int(i)] for i in ref]
            source = "pool_clean_tag"
        influence, proba, classes = probe_log_likelihood(ref_X, ref_y, Xp, obs, max_iter=200)
        if getattr(tcfg, "influence_within_class", False):
            from omniselect.core.signals.influence import within_class_rank
            influence = within_class_rank(influence, obs)
        n_cols = int(max(int(obs.max()), int(a["yval"].max()), int(a["yt"].max()))) + 1
        proba_full = np.zeros((len(Xp), n_cols))
        proba_full[:, classes] = proba
        console = MultiActorConsole(
            [("redundancy", RedundancySignal()), ("influence", InfluenceSignal())],
            weights=np.log(np.array([1 - tcfg.w_infl, tcfg.w_infl]) + 1e-9),
            conflict_gate=True, anneal=0.6, group_key="domain", trust_region=0.5, ema_beta=0.3, min_weight=0.02,
        )
        imp_dyn = console.importance(data.records, scores=np.stack([minmax(redundancy), minmax(influence)]),
                                     progress=0.5)
        extras = {"imp_dyn": imp_dyn, "auth_features": Xp, "coverage_features": Xp, "knn": tcfg.knn}
        if tcfg.learner in ("clip_vitb32", "dinov2_vits14"):
            extras["last_layer"] = _last_layer_probe(data, tcfg)
        return Signals(auth=auth, influence=influence, redundancy=redundancy, features=Xp.astype(float),
                       reference_ids=ref_ids, reference_source=source, proba=proba_full, extras=extras)

    def alignment(self, data: TaskData, sig: Signals, tcfg: VisionConfig, ocfg):
        """Alignment from a logistic head on the CLIP features of the signals (learner independent)."""
        from tracks.common.alignment import from_last_layer

        return from_last_layer(_last_layer_probe(data, tcfg, prefix="")(), "logistic head on CLIP features")

    def cooperative_inputs(self, data: TaskData, sig: Signals, tcfg: VisionConfig, ocfg) -> dict[str, Any]:
        """Cosine geometry of the CLIP features. Cleanliness features from the CLIP features and labels."""
        a, con = data.arrays, data.splits.con
        return {"veto_features": sig.features, "metric": "cosine", "con_veto_features": a["Xval"][con],
                "selector_features": sig.features,
                "cleanliness": {"kind": "classification", "pool_X": a["Xp"], "pool_y": data.labels,
                                "con_X": a["Xval"][con], "con_y": a["yval"][con]}}

    def learner(self, data: TaskData, tcfg: VisionConfig, ocfg):
        a = data.arrays
        report_seed = fit_seed(tcfg.seed, "final-fit", ocfg.paired_rng, reset=False)
        if tcfg.learner == "resnet18_scratch":
            from tracks.vision.image_learners import ResNetScratchLearner

            plan = stage_plan(ocfg, scoring={"epochs": tcfg.resnet_epochs}, reported={"epochs": tcfg.resnet_epochs},
                              seeds={"con": report_seed, "rank": report_seed, "report": report_seed},
                              schedule_key="epochs", low={"epochs": tcfg.resnet_screen_epochs})
            split_images = {name: (a["images_val"][data.splits.get(name)], a["yval"][data.splits.get(name)])
                            for name in ("con", "rank", "conf")}
            split_images["test"] = (a["images_t"], a["yt"])
            return ResNetScratchLearner(plan, a["images_p"], data.labels, split_images, data.n_classes,
                                        resolve_device(tcfg.device), batch_size=tcfg.resnet_batch, lr=tcfg.resnet_lr)
        plan = stage_plan(ocfg, scoring={"max_iter": tcfg.scoring_max_iter},
                          reported={"max_iter": tcfg.reported_max_iter},
                          seeds={"con": None, "rank": None, "report": report_seed}, schedule_key="max_iter",
                          low={"max_iter": tcfg.screen_max_iter})
        prefix = "learner_" if tcfg.learner == "dinov2_vits14" else ""
        split_arrays = {name: (a[prefix + "Xval"][data.splits.get(name)], a["yval"][data.splits.get(name)])
                        for name in ("con", "rank", "conf")}
        split_arrays["test"] = (a[prefix + "Xt"], a["yt"])
        if prefix:
            from tracks.vision.image_learners import FeatureProbeLearner

            return FeatureProbeLearner(plan, a["learner_Xp"], data.labels, split_arrays, tcfg.learner)
        return ProbeLearner(plan, a["Xp"], data.labels, split_arrays)
