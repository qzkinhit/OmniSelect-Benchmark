"""Tabular track: OpenML Electricity with TabPFN-v2 in context (xgboost and rf as alternatives).

The selected pool rows are TabPFN's context set. There is no gradient training, so a fit costs
forward passes only and counts 0 FDE. Corruption flips labels, adds feature noise, or duplicates
rows (canonical), or sticks features and flips positives to the negative class (unseen). The task
metric is ROC AUC from predict_proba. The Tab-AICL acquisition rules read TabPFN
probabilities from a 300-row seed context drawn with default_rng(seed + 99).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmark.Data import tabular as tdata
from omniselect.core.datatypes import Modality, UnifiedRecord
from omniselect.core.selection.console import MultiActorConsole
from omniselect.core.signals import InfluenceSignal, RedundancySignal, minmax
from omniselect.core.signals.influence import probe_log_likelihood
from omniselect.core.signals.knn import knn_agreement_and_novelty, l2_normalize
from tracks.common.downstream import Learner, classification_units, stage_plan
from tracks.common.experiment import SkipCell
from tracks.common.injection import contaminate_validation, inject_rows, unseen_rows
from tracks.common.splits import validation_splits
from tracks.common.task import Signals, TaskData, Track, TrackConfig

TAB_METHODS = ("full", "random", "coreset", "auth_only", "herding", "el2n", "grand", "ccs", "density",
               "quadmix_pub", "dmf_pub", "influence_only", "mmdataselect", "mmds_adapt")
SMOKE = {"pool_n": 300, "val_n": 300, "test_n": 300, "dsdm_runs": 3, "dmf_rounds": 2, "influence_ref_n": 100}


@dataclass
class TabularConfig(TrackConfig):
    """Tabular constants."""

    openml_name: str = "electricity"
    aicl_seed_rows: int = 300
    last_layer_warm: int = 100        # alignment channel: iterations of the logistic head


def make_model(kind: str, seed: int):
    """TabPFN-v2 (cuda when available, else cpu), XGBoost or a depth-4 random forest."""
    if kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=4, tree_method="hist",
                             eval_metric="logloss", verbosity=0)
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=200, max_depth=4, n_jobs=4, random_state=seed)
    import torch
    from tabpfn import TabPFNClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TabPFNClassifier.create_default_for_version("v2", device=device, ignore_pretraining_limits=True)


class TabularLearner(Learner):
    """Context-set learner. Per-unit outputs carry class probabilities for AUC."""

    def __init__(self, plan, kind, Xp, yp, split_arrays, seed):
        super().__init__(plan=plan, keep_models=True, name=kind)
        self.kind, self.Xp, self.yp, self.split_arrays, self.seed = kind, Xp, yp, split_arrays, seed

    def cost_factor(self, stage: str) -> float:
        return 0.0 if self.kind == "tabpfn" else 1.0

    def fit(self, subset, stage):
        return make_model(self.kind, self.seed).fit(self.Xp[subset], self.yp[subset])

    def score(self, model, split):
        X, y = self.split_arrays[split]
        return classification_units(model, X, y, predict_from_proba=True)

    def save(self, model, directory: Path, name: str) -> str:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.pkl"
        with open(path, "wb") as handle:
            pickle.dump(model, handle)
        return str(path)


def tabpfn_ready() -> str | None:
    """None when TabPFN imports and its v2 classifier weights load, else the reason."""
    try:
        import numpy as _np
        from tabpfn import TabPFNClassifier

        clf = TabPFNClassifier.create_default_for_version("v2", device="cpu", ignore_pretraining_limits=True)
        clf.fit(_np.random.default_rng(0).normal(size=(20, 3)), _np.array([0, 1] * 10))
        return None
    except Exception as exc:  # noqa: BLE001 - reported as the skip reason
        return f"{type(exc).__name__}: {exc}"


class TabularTrack(Track):
    """OpenML Electricity."""

    name = "tabular"
    utility = "auc"
    config_cls = TabularConfig
    default_learner = "tabpfn"
    datasets = {"electricity": {"openml_name": "electricity", "pool_n": 3000, "val_n": 2500, "test_n": 2000,
                                "budget_frac": 0.5, "dsdm_runs": 12, "dmf_rounds": 4, "influence_ref_n": 300,
                                "grid_weights": ((1, 0, 0), (0, .5, .5), (.34, .33, .33)), "grid_lam": (0.0, 0.6),
                                "methods": TAB_METHODS, "smoke": SMOKE}}

    def config(self, dataset, learner=None, seed=0, smoke=False, **overrides):
        """TrackConfig. For the xgboost learner xgboost is imported here, before the run record imports torch,
        because loading torch's OpenMP runtime first makes xgboost segfault on macOS."""
        cfg = super().config(dataset, learner=learner, seed=seed, smoke=smoke, **overrides)
        if cfg.learner == "xgboost":
            import xgboost  # noqa: F401
        return cfg

    def load(self, tcfg: TabularConfig, ocfg) -> TaskData:
        if tcfg.learner not in ("tabpfn", "xgboost", "rf"):
            raise ValueError(f"tabular learner must be tabpfn, xgboost or rf, got {tcfg.learner!r}")

        if tcfg.learner == "tabpfn":
            reason = tabpfn_ready()
            if reason is not None:
                raise SkipCell(f"TabPFN-v2 weights unavailable: {reason}")
        try:
            arr = tdata.load_table(tcfg.openml_name, tcfg.pool_n, tcfg.val_n, tcfg.test_n, tcfg.seed)
        except (OSError, ConnectionError) as exc:
            raise SkipCell(f"OpenML {tcfg.openml_name} unavailable: {exc}") from exc
        rb = ocfg.robustness
        yval, val_pos, val_noise = contaminate_validation(rb.val_noise_kind, arr.yval, rb.val_noise_rate, tcfg.seed,
                                                          modulus=arr.n_classes, max_offset=arr.n_classes)
        noise = tcfg.noise_frac if rb.injection_ratio < 0 else rb.injection_ratio
        if rb.mechanism_set == "unseen":
            Xp, obs, tags = unseen_rows(arr.Xp, arr.yp, tcfg.seed, arr.n_classes, noise)
        else:
            Xp, obs, tags = inject_rows(arr.Xp, arr.yp, tcfg.seed, arr.n_classes, noise)
        n = len(Xp)
        records = [UnifiedRecord(id=str(i), modality=Modality.TEXT, domain="table", text="") for i in range(n)]
        return TaskData(
            pool_ids=[int(i) for i in arr.pool_idx], val_ids=[int(arr.val_idx[p]) for p in val_pos],
            test_ids=[int(i) for i in arr.test_idx], tags=tags,
            splits=validation_splits(len(yval), tcfg.seed, ocfg), budget=int(tcfg.budget_frac * n),
            utility=self.utility, records=records, labels=obs, labels_clean=arr.yp, n_classes=arr.n_classes,
            arrays={"Xp": Xp, "Xval": arr.Xval[val_pos], "yval": yval, "Xt": arr.Xt, "yt": arr.yt},
            provenance={"openml": arr.manifest, "val_noise": val_noise},
            mechanisms=[str(t) if t != "high" else "" for t in tags] if rb.mechanism_set == "unseen" else None,
        )

    def signals(self, data: TaskData, tcfg: TabularConfig, ocfg) -> Signals:
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
        influence, proba, classes = probe_log_likelihood(ref_X, ref_y, Xp, obs, max_iter=200)
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
        cache: dict = {}
        n, budget = len(Xp), data.budget

        def aicl_probs():
            if "p" not in cache:
                seed_sub = np.random.default_rng(tcfg.seed + 99).permutation(n)[: min(tcfg.aicl_seed_rows, budget)]
                cache["p"] = make_model(tcfg.learner, tcfg.seed).fit(Xp[seed_sub], obs[seed_sub]).predict_proba(Xp)
            return cache["p"]

        return Signals(auth=auth, influence=influence, redundancy=redundancy, features=Xn.astype(float),
                       reference_ids=ref_ids, reference_source=source, proba=proba_full,
                       extras={"imp_dyn": imp_dyn, "auth_features": Xn, "knn": tcfg.knn, "aicl_probs": aicl_probs})

    def alignment(self, data: TaskData, sig: Signals, tcfg: TabularConfig, ocfg):
        """Alignment from a logistic head on TabPFN embeddings of the pool and V_con.

        TabPFN-v2 conditions on the Tab-AICL seed context (default_rng(seed + 99), ``aicl_seed_rows``
        rows) and embeds every pool and V_con row as a test token (mean over estimators). When TabPFN
        is unavailable the head runs on the L2-normalized feature rows.
        """
        from sklearn.linear_model import LogisticRegression

        from benchmark.Methods._gradients import LastLayer, append_bias, softmax_theta
        from tracks.common.alignment import from_last_layer

        a = data.arrays
        Xp, obs, con = a["Xp"], np.asarray(data.labels), data.splits.con
        Xv, yv = a["Xval"][con], np.asarray(a["yval"][con])
        try:
            n = len(Xp)
            seed_sub = np.random.default_rng(tcfg.seed + 99).permutation(n)[: min(tcfg.aicl_seed_rows, data.budget)]
            clf = make_model("tabpfn", tcfg.seed).fit(Xp[seed_sub], obs[seed_sub])
            emb = np.asarray(clf.get_embeddings(np.vstack([Xp, Xv]), data_source="test"), dtype=float)
            emb = emb.mean(axis=0) if emb.ndim == 3 else emb
            Ep, Ev, source = emb[:n], emb[n:], "logistic head on TabPFN test embeddings"
        except Exception:  # noqa: BLE001 - TabPFN weights or its embedding API unavailable
            Ep, Ev, source = l2_normalize(Xp), l2_normalize(Xv), "logistic head on normalized features"
        head = LogisticRegression(C=1.0, max_iter=tcfg.last_layer_warm, random_state=tcfg.seed).fit(Ep, obs)
        theta = softmax_theta(head.coef_.T, head.intercept_, head.classes_, data.n_classes)
        model = LastLayer("softmax", append_bias(Ep), obs, theta, append_bias(Ev), yv, labels=obs, source=source)
        return from_last_layer(model, source)

    def cooperative_inputs(self, data: TaskData, sig: Signals, tcfg: TabularConfig, ocfg) -> dict:
        """Euclidean geometry of the standardized rows. Cleanliness features from those rows and the labels."""
        a, con = data.arrays, data.splits.con
        return {"veto_features": a["Xp"], "metric": "euclidean", "con_veto_features": a["Xval"][con],
                "selector_features": sig.features,
                "cleanliness": {"kind": "classification", "pool_X": a["Xp"], "pool_y": data.labels,
                                "con_X": a["Xval"][con], "con_y": a["yval"][con]}}

    def learner(self, data: TaskData, tcfg: TabularConfig, ocfg) -> TabularLearner:
        a = data.arrays
        plan = stage_plan(ocfg, scoring={"context": "full"}, reported={"context": "full"},
                          seeds={"con": tcfg.seed, "rank": tcfg.seed, "report": tcfg.seed},
                          low={"context": "subsample"})
        split_arrays = {name: (a["Xval"][data.splits.get(name)], a["yval"][data.splits.get(name)])
                        for name in ("con", "rank", "conf")}
        split_arrays["test"] = (a["Xt"], a["yt"])
        return TabularLearner(plan, tcfg.learner, a["Xp"], data.labels, split_arrays, tcfg.seed)
