"""Process track: Tennessee Eastman fault diagnosis (normal plus 21 faults, 22 classes).

Rows are the 52 standardized process variables. The pool comes from the training files, the
validation and test sets from the post-onset test files. Corruption flips labels, adds Gaussian
feature noise, or duplicates rows (canonical), or sticks features and flips labels to the next
class (unseen). The canonical learner is an MLP (128, 64), 120 iterations.
Rf, cnn1d, svm, knn and pca are alternatives. The task metric is macro-F1.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from benchmark.Data import process as pdata
from omniselect.core.datatypes import Modality, UnifiedRecord
from omniselect.core.selection.console import MultiActorConsole
from omniselect.core.signals import InfluenceSignal, RedundancySignal, minmax
from omniselect.core.signals.influence import probe_log_likelihood
from omniselect.core.signals.knn import knn_agreement_and_novelty, l2_normalize
from tracks.common.downstream import Learner, classification_units, stage_plan
from tracks.common.injection import contaminate_validation, inject_rows, unseen_rows
from tracks.common.pairing import fit_seed
from tracks.common.splits import validation_splits
from tracks.common.task import Signals, TaskData, Track, TrackConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
TEP_METHODS = ("full", "random", "coreset", "auth_only", "herding", "el2n", "grand", "ccs", "density",
               "quadmix_pub", "dmf_pub", "influence_only", "mmdataselect", "mmds_adapt")
LEARNERS = ("mlp", "rf", "cnn1d", "svm", "knn", "pca")
SMOKE = {"pool_n": 400, "val_n": 300, "test_n": 300, "mlp_max_iter": 30, "dsdm_runs": 3, "dmf_rounds": 2,
         "influence_ref_n": 150}


@dataclass
class ProcessConfig(TrackConfig):
    """TEP constants: number of faults and learner settings."""

    n_faults: int = 21
    mlp_max_iter: int = 120
    cnn_epochs: int = 80
    screen_mlp_max_iter: int = 30     # low-fidelity screening schedule (MLP)
    last_layer_warm: int = 30         # GLISTER and GRAD-MATCH: warm-start iterations of the MLP
    screen_cnn_epochs: int = 30       # low-fidelity screening schedule (cnn1d)
    manifest: str = "data/tep/SHA256SUMS.txt"


class CNN1D:
    """Two convolution blocks over the 52 variables as a length-52 signal, full-batch Adam."""

    def __init__(self, seed: int = 0, epochs: int = 80):
        self.seed, self.epochs = seed, epochs

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        self.classes_ = np.unique(y)
        index = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([index[v] for v in y])
        d = X.shape[1]
        self.net = nn.Sequential(
            nn.Unflatten(1, (1, d)), nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(32, len(self.classes_)))
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(yi, dtype=torch.long)
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = lossf(self.net(xt), yt)
            loss.backward()
            opt.step()
        return self

    def predict_proba(self, X):
        import torch

        self.net.eval()
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor(X, dtype=torch.float32)), dim=1).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def make_classifier(kind: str, seed: int, tcfg: ProcessConfig, schedule: Optional[dict] = None):
    """Classifier of the given kind with the canonical hyperparameters.

    ``schedule`` (a fidelity dict) overrides the MLP ``max_iter`` and the cnn1d ``epochs``.
    """
    schedule = schedule or {}
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=int(schedule.get("max_iter", tcfg.mlp_max_iter)),
                             early_stopping=False, random_state=seed)
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=120, n_jobs=4, random_state=seed)
    if kind == "cnn1d":
        return CNN1D(seed=seed, epochs=int(schedule.get("epochs", tcfg.cnn_epochs)))
    if kind == "svm":
        from sklearn.svm import SVC

        return SVC(kernel="rbf", C=1.0, random_state=seed)
    if kind == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier(n_neighbors=5)
    if kind == "pca":
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        return make_pipeline(PCA(n_components=20, random_state=seed), LogisticRegression(max_iter=300))
    raise KeyError(f"unknown process learner {kind!r}")


class ProcessLearner(Learner):
    """Fits the configured classifier on pool rows. Per-unit outputs include probabilities when available."""

    def __init__(self, plan, kind, Xp, yp, split_arrays, tcfg):
        super().__init__(plan=plan, keep_models=True, name=kind)
        self.kind, self.Xp, self.yp, self.split_arrays, self.tcfg = kind, Xp, yp, split_arrays, tcfg

    def fit(self, subset, stage):
        fid = self.fidelity(stage)
        return make_classifier(self.kind, fid["seed"], self.tcfg, fid).fit(self.Xp[subset], self.yp[subset])

    def score(self, model, split):
        X, y = self.split_arrays[split]
        return classification_units(model, X, y)

    def save(self, model, directory: Path, name: str) -> str:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.pkl"
        with open(path, "wb") as handle:
            pickle.dump(model, handle)
        return str(path)


def _last_layer_mlp(data: TaskData, tcfg: ProcessConfig):
    """Callable building the GLISTER and GRAD-MATCH last layer of the MLP (128, 64).

    The MLP is warm-started on the whole pool for ``last_layer_warm`` iterations (seed = run seed).
    phi is the second hidden layer (ReLU) with a constant column, Theta the output layer.
    """
    cache: dict = {}

    def build():
        if "model" not in cache:
            from sklearn.neural_network import MLPClassifier

            from benchmark.Methods._gradients import LastLayer, append_bias, softmax_theta

            a = data.arrays
            mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=tcfg.last_layer_warm, early_stopping=False,
                                random_state=tcfg.seed).fit(a["Xp"], data.labels)

            def hidden(X):
                h = np.asarray(X, dtype=float)
                for W, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
                    h = np.maximum(h @ W + b, 0.0)
                return h

            con = data.splits.con
            theta = softmax_theta(mlp.coefs_[-1], mlp.intercepts_[-1], mlp.classes_, data.n_classes)
            cache["model"] = LastLayer("softmax", append_bias(hidden(a["Xp"])), np.asarray(data.labels), theta,
                                       append_bias(hidden(a["Xval"][con])), np.asarray(a["yval"][con]),
                                       labels=np.asarray(data.labels), source="MLP (128, 64) output layer")
        return cache["model"]

    return build


class ProcessTrack(Track):
    """TEP fault diagnosis."""

    name = "process"
    utility = "macro_f1"
    config_cls = ProcessConfig
    default_learner = "mlp"
    datasets = {"tep21": {"n_faults": 21, "pool_n": 4000, "val_n": 2000, "test_n": 3000, "budget_frac": 0.3,
                          "dsdm_runs": 16, "dmf_rounds": 6, "influence_ref_n": 600,
                          "grid_weights": ((1, 0, 0), (0, .5, .5), (.34, .33, .33)), "grid_lam": (0.0, 0.5),
                          "methods": TEP_METHODS, "smoke": SMOKE}}

    def load(self, tcfg: ProcessConfig, ocfg) -> TaskData:
        if tcfg.learner not in LEARNERS:
            raise ValueError(f"process learner must be one of {LEARNERS}, got {tcfg.learner!r}")
        manifest = tcfg.manifest if os.path.isabs(tcfg.manifest) else str(REPO_ROOT / tcfg.manifest)
        arr = pdata.load_tep(tcfg.data_root, manifest, tcfg.n_faults, tcfg.pool_n, tcfg.val_n, tcfg.test_n,
                             tcfg.seed)
        rb = ocfg.robustness
        yval, val_pos, val_noise = contaminate_validation(rb.val_noise_kind, arr.yval, rb.val_noise_rate, tcfg.seed,
                                                          modulus=arr.n_classes, max_offset=arr.n_classes)
        noise = tcfg.noise_frac if rb.injection_ratio < 0 else rb.injection_ratio
        if rb.mechanism_set == "unseen":
            Xp, obs, tags = unseen_rows(arr.Xp, arr.yp, tcfg.seed, arr.n_classes, noise)
        else:
            Xp, obs, tags = inject_rows(arr.Xp, arr.yp, tcfg.seed, arr.n_classes, noise)
        n = len(Xp)
        records = [UnifiedRecord(id=str(i), modality=Modality.TEXT, domain="process", text="") for i in range(n)]
        return TaskData(
            pool_ids=[f"train:{int(i)}" for i in arr.pool_idx],
            val_ids=[f"test:{int(arr.val_idx[p])}" for p in val_pos],
            test_ids=[f"test:{int(i)}" for i in arr.test_idx], tags=tags,
            splits=validation_splits(len(yval), tcfg.seed, ocfg), budget=int(tcfg.budget_frac * n),
            utility=self.utility, records=records, labels=obs, labels_clean=arr.yp, n_classes=arr.n_classes,
            arrays={"Xp": Xp, "Xval": arr.Xval[val_pos], "yval": yval, "Xt": arr.Xt, "yt": arr.yt, "Xcal": arr.Xcal,
                    "ycal": arr.ycal},
            provenance={"tep_files": arr.files, "manifest": manifest, "n_faults": tcfg.n_faults,
                        "calibration_ids": [f"test:{int(i)}" for i in arr.cal_idx], "val_noise": val_noise},
            mechanisms=[str(t) if t != "high" else "" for t in tags] if rb.mechanism_set == "unseen" else None,
        )

    def signals(self, data: TaskData, tcfg: ProcessConfig, ocfg) -> Signals:
        a = data.arrays
        Xp, obs = a["Xp"], data.labels
        Xn = l2_normalize(Xp)
        auth, redundancy = knn_agreement_and_novelty(Xn, obs, tcfg.knn)
        if ocfg.influence.reference == "v_con":
            con = data.splits.con[: min(len(data.splits.con), tcfg.influence_ref_n)]
            ref_X, ref_y, ref_ids, source = a["Xval"][con], a["yval"][con], [data.val_ids[int(i)] for i in con], "v_con"
        else:
            ref = np.random.default_rng(tcfg.seed).permutation(np.where(data.tags == "high")[0])[: tcfg.influence_ref_n]
            ref_X, ref_y, ref_ids, source = Xp[ref], obs[ref], [data.pool_ids[int(i)] for i in ref], "pool_clean_tag"
        influence, proba, classes = probe_log_likelihood(ref_X, ref_y, Xp, obs, max_iter=300)
        if getattr(tcfg, "influence_within_class", False):
            from omniselect.core.signals.influence import within_class_rank
            influence = within_class_rank(influence, obs)
        proba_full = np.zeros((len(Xp), data.n_classes))
        proba_full[:, classes] = proba
        console = MultiActorConsole(
            [("redundancy", RedundancySignal()), ("influence", InfluenceSignal())],
            weights=np.log(np.array([1 - tcfg.w_infl, tcfg.w_infl]) + 1e-9),
            conflict_gate=True, anneal=0.6, group_key="domain", trust_region=0.5, ema_beta=0.3, min_weight=0.02,
        )
        imp_dyn = console.importance(data.records, scores=np.stack([minmax(redundancy), minmax(influence)]),
                                     progress=0.5)
        extras = {"imp_dyn": imp_dyn, "auth_features": Xp, "knn": tcfg.knn}
        if tcfg.learner == "mlp":
            extras["last_layer"] = _last_layer_mlp(data, tcfg)
        return Signals(auth=auth, influence=influence, redundancy=redundancy, features=Xn.astype(float),
                       reference_ids=ref_ids, reference_source=source, proba=proba_full, extras=extras)

    def alignment(self, data: TaskData, sig: Signals, tcfg: ProcessConfig, ocfg):
        """Alignment from the output layer of the MLP (128, 64) warm-started on the pool (learner independent)."""
        from tracks.common.alignment import from_last_layer

        return from_last_layer(_last_layer_mlp(data, tcfg)(), "MLP (128, 64) output layer")

    def cooperative_inputs(self, data: TaskData, sig: Signals, tcfg: ProcessConfig, ocfg) -> dict:
        """Euclidean geometry of the standardized rows. Cleanliness features from those rows and the labels."""
        a, con = data.arrays, data.splits.con
        return {"veto_features": a["Xp"], "metric": "euclidean", "con_veto_features": a["Xval"][con],
                "selector_features": sig.features,
                "cleanliness": {"kind": "classification", "pool_X": a["Xp"], "pool_y": data.labels,
                                "con_X": a["Xval"][con], "con_y": a["yval"][con]}}

    def learner(self, data: TaskData, tcfg: ProcessConfig, ocfg) -> ProcessLearner:
        a = data.arrays
        schedule = {"max_iter": tcfg.mlp_max_iter} if tcfg.learner == "mlp" else {"epochs": tcfg.cnn_epochs}
        low = {"mlp": {"max_iter": tcfg.screen_mlp_max_iter}, "cnn1d": {"epochs": tcfg.screen_cnn_epochs}}
        key = {"mlp": "max_iter", "cnn1d": "epochs"}.get(tcfg.learner, "")
        report_seed = fit_seed(tcfg.seed, "final-fit", ocfg.paired_rng, reset=False)
        plan = stage_plan(ocfg, scoring=schedule, reported=schedule,
                          seeds={"con": tcfg.seed, "rank": tcfg.seed, "report": report_seed}, schedule_key=key,
                          low=low.get(tcfg.learner))
        split_arrays = {name: (a["Xval"][data.splits.get(name)], a["yval"][data.splits.get(name)])
                        for name in ("con", "rank", "conf")}
        split_arrays["test"] = (a["Xt"], a["yt"])
        return ProcessLearner(plan, tcfg.learner, a["Xp"], data.labels, split_arrays, tcfg)
