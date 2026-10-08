"""Forecasting track: DLinear trained from scratch on selected windows (ETTh1, ETTh2, ETTm1, DaISy).

Windows of length L = 96 predict H = 24 steps of the z-normalized series. Pool windows come from
the first 70% of start positions. Validation and test windows follow either the canonical layout
or v2 contiguous time blocks. Pool corruption is canonical (Gaussian, flat, shuffle, near-duplicate)
or unseen (scale drift, bias drift, stuck sensor). Signals are |lag-1 autocorrelation|, the negative
error of a reference forecaster, and kNN window novelty. Scoring fits use 40 epochs (canonical) or 60
(v2). The learner is DLinear or fine-tuned chronos-bolt tiny or small.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmark.Data import timeseries as tsdata
from omniselect.core.datatypes import Modality, UnifiedRecord
from omniselect.core.selection.console import MultiActorConsole
from omniselect.core.signals import InfluenceSignal, RedundancySignal, minmax
from omniselect.core.signals.knn import knn_agreement_and_novelty, l2_normalize
from tracks.common.downstream import Learner, resolve_device, stage_plan
from tracks.common.injection import inject_windows, unseen_windows
from tracks.common.pairing import fit_seed
from tracks.common.splits import block_splits, forecast_starts, validation_splits
from tracks.common.task import Signals, TaskData, Track, TrackConfig

TS_METHODS = ("full", "random", "coreset", "auth_only", "influence_only", "mmdataselect", "herding", "kcenter",
              "semdedup", "density", "quadmix_pub", "dmf_pub", "mmds_adapt")
SMOKE = {"pool_n": 300, "val_n": 200, "test_n": 200, "epochs": 5, "scoring_epochs": 3, "reference_epochs": 3,
         "dsdm_runs": 3, "dmf_rounds": 2, "grid_lam": (0.0, 0.6), "influence_ref_n": 100}


@dataclass
class ForecastConfig(TrackConfig):
    """Forecasting constants: window lengths, epochs, learner schedules."""

    L: int = 96
    H: int = 24
    epochs: int = 60                  # reported DLinear schedule
    scoring_epochs: int = 40          # canonical scoring schedule
    reference_epochs: int = 40        # influence reference forecaster
    screen_epochs: int = 15           # low-fidelity screening schedule (DLinear)
    last_layer_warm: int = 10         # GLISTER and GRAD-MATCH: warm-start epochs of DLinear
    chronos_screen_epochs: int = 1    # low-fidelity screening schedule (chronos-bolt)
    recent_validation: bool = False   # historical TS_VAL_MODE=recent (negative result)
    chronos_epochs: int = 3
    chronos_max_epochs: int = 4
    chronos_lr: float = 1e-4
    chronos_model: str = "amazon/chronos-bolt-tiny"
    chronos_revision: str = "a0e552de83495b5c28c14c71c374f3e33280b340"
    chronos_grad_clip: float = 1.0    # gradient norm clip of chronos fine-tuning
    chronos_patience: int = 1         # epochs without a lower V_con loss before chronos stops
    chronos_predict_batch: int = 64   # inference only, cached Chronos models reside on the CPU
    drift_scale: float = 1.3          # unseen mechanism: multiplicative drift at the end of the segment
    drift_bias: float = 1.0           # unseen mechanism: additive drift in units of the pool window sd


LEARNERS = ("dlinear", "chronos_tiny", "chronos_small")
LEARNER_ALIASES = {"chronos": "chronos_tiny"}
CHRONOS_MODELS = {
    "chronos_tiny": ("amazon/chronos-bolt-tiny", "a0e552de83495b5c28c14c71c374f3e33280b340"),
    "chronos_small": ("amazon/chronos-bolt-small", "772f3d25d38aec6d914c8949dab4462e2d46f5d8"),
}


def autocorr1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of a window (0 for a constant window)."""
    x = x - x.mean()
    d = (x[:-1] ** 2).sum()
    return float((x[:-1] * x[1:]).sum() / (d + 1e-9)) if d > 0 else 0.0


def train_dlinear(X: np.ndarray, Y: np.ndarray, device: str, seed: int, epochs: int):
    """DLinear (moving-average trend plus seasonal linear maps), Adam 1e-3, batch 256, MSE."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class DLinear(nn.Module):
        def __init__(self, L, H, k=25):
            super().__init__()
            self.k = k
            self.lt = nn.Linear(L, H)
            self.ls = nn.Linear(L, H)

        def forward(self, x):
            pad = torch.nn.functional.pad(x, (self.k // 2, self.k // 2), mode="replicate")
            trend = torch.nn.functional.avg_pool1d(pad.unsqueeze(1), self.k, 1).squeeze(1)[:, : x.shape[1]]
            return self.lt(trend) + self.ls(x - trend)

    model = DLinear(X.shape[1], Y.shape[1]).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    yb = torch.tensor(Y, dtype=torch.float32, device=device)
    n = len(xb)
    order = hashlib.sha256()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        order.update(perm.detach().cpu().numpy().tobytes())
        for s in range(0, n, 256):
            idx = perm[s:s + 256]
            opt.zero_grad()
            loss = ((model(xb[idx]) - yb[idx]) ** 2).mean()
            loss.backward()
            opt.step()
    model.train_order_sha12 = order.hexdigest()[:12]
    return model.eval()


def predict(model, X: np.ndarray, device: str, batch_size: int | None = None) -> np.ndarray:
    """Model forecasts for the windows X."""
    import torch

    with torch.no_grad():
        if batch_size is not None:
            if batch_size < 1:
                raise ValueError("prediction batch size must be positive")
            return np.concatenate([
                model(torch.tensor(X[s:s + batch_size], dtype=torch.float32, device=device)).cpu().numpy()
                for s in range(0, len(X), batch_size)
            ])
        return model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()


def dlinear_features(model, X: np.ndarray) -> np.ndarray:
    """[trend, 1, seasonal, 1] of the DLinear decomposition, so that the forecast is features @ Theta."""
    import torch

    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32)
        k = model.k
        pad = torch.nn.functional.pad(x, (k // 2, k // 2), mode="replicate")
        trend = torch.nn.functional.avg_pool1d(pad.unsqueeze(1), k, 1).squeeze(1)[:, : x.shape[1]].numpy()
    ones = np.ones((len(X), 1))
    return np.hstack([trend, ones, np.asarray(X) - trend, ones]).astype(float)


def _last_layer_dlinear(data: TaskData, tcfg: "ForecastConfig", device: str):
    """Callable building the GLISTER and GRAD-MATCH model of DLinear, which is linear in its features.

    DLinear is warm-started on the whole pool for ``last_layer_warm`` epochs. phi = [trend, 1, seasonal, 1],
    Theta stacks the trend and seasonal maps and biases, the loss is half the squared error.
    """
    cache: dict = {}

    def build():
        if "model" not in cache:
            from benchmark.Methods._gradients import LastLayer

            a = data.arrays
            model = train_dlinear(a["Xp"], a["Yp"], device, tcfg.seed, tcfg.last_layer_warm).cpu()
            theta = np.vstack([model.lt.weight.detach().numpy().T, model.lt.bias.detach().numpy()[None],
                               model.ls.weight.detach().numpy().T, model.ls.bias.detach().numpy()[None]])
            con = data.splits.con
            cache["model"] = LastLayer("squared", dlinear_features(model, a["Xp"]), np.asarray(a["Yp"], float),
                                       theta.astype(float), dlinear_features(model, a["Xval"][con]),
                                       np.asarray(a["Yval"][con], float), labels=None, source="DLinear")
        return cache["model"]

    return build


class ForecastLearner(Learner):
    """DLinear or chronos-bolt learner. Per-unit outputs are forecasts, targets and last inputs."""

    def __init__(self, plan, kind, Xp, Yp, split_arrays, device, tcfg):
        super().__init__(plan=plan, keep_models=True, name=kind)
        self.kind, self.Xp, self.Yp, self.split_arrays = kind, Xp, Yp, split_arrays
        self.device, self.tcfg = device, tcfg

    def fit(self, subset, stage):
        fid = self.fidelity(stage)
        seed = fid["seed"]
        if fid.get("reset_rng"):
            from omniselect.tools.pairing import reset_rng

            reset_rng(*fid["reset_rng"])
        if self.kind.startswith("chronos"):
            from tracks.timeseries.chronos import train_chronos

            Xc, Yc, _ = self.split_arrays["con"]
            val = (Xc, Yc) if len(Xc) else None
            model = train_chronos(self.Xp[subset], self.Yp[subset], self.device, seed, fid["epochs"], self.tcfg,
                                  val=val)
            # FitCache retains every candidate for train-once scoring. Keep fitted weights on CPU
            # and discard unused final-step gradients so candidate count does not grow GPU memory.
            model.model.zero_grad(set_to_none=True)
            return model.to("cpu")
        return train_dlinear(self.Xp[subset], self.Yp[subset], self.device, seed, fid["epochs"])

    def score(self, model, split):
        X, Y, starts = self.split_arrays[split]
        if self.kind.startswith("chronos"):
            model.to(self.device)
            try:
                prediction = predict(model, X, self.device, self.tcfg.chronos_predict_batch)
            finally:
                model.to("cpu")
        else:
            prediction = predict(model, X, self.device)
        return {"prediction": prediction, "target": Y, "last_value": X[:, -1],
                "start": np.asarray(starts)}

    def save(self, model, directory: Path, name: str) -> str:
        directory.mkdir(parents=True, exist_ok=True)
        import torch

        path = directory / "model.pt"
        torch.save(model.state_dict() if hasattr(model, "state_dict") else pickle.dumps(model), path)
        return str(path)


class ForecastTrack(Track):
    """Forecasting windows with a from-scratch DLinear (or fine-tuned chronos-bolt)."""

    name = "timeseries"
    utility = "neg_mase"
    config_cls = ForecastConfig
    default_learner = "dlinear"
    canonical_sorted_scoring = True
    datasets = {
        name: {"pool_n": 3000, "val_n": 1000, "test_n": 1500, "budget_frac": 0.3, "dsdm_runs": 12, "dmf_rounds": 3,
               "methods": TS_METHODS, "smoke": SMOKE}
        for name in ("ETTh1", "ETTh2", "ETTm1", "daisy_cstr", "daisy_steamgen")
    }

    def config(self, dataset, learner=None, seed=0, smoke=False, **overrides):
        """TrackConfig with the chronos-bolt model and revision of the learner (chronos means chronos_tiny)."""
        learner = LEARNER_ALIASES.get(learner, learner)
        if "learner" in overrides:
            overrides["learner"] = LEARNER_ALIASES.get(overrides["learner"], overrides["learner"])
        cfg = super().config(dataset, learner=learner, seed=seed, smoke=smoke, **overrides)
        if cfg.learner not in LEARNERS:
            raise ValueError(f"forecasting learner must be one of {LEARNERS}, got {cfg.learner!r}")
        if cfg.learner in CHRONOS_MODELS and "chronos_model" not in overrides:
            cfg.chronos_model, cfg.chronos_revision = CHRONOS_MODELS[cfg.learner]
        return cfg

    def load(self, tcfg: ForecastConfig, ocfg) -> TaskData:
        raw, path = tsdata.load_series(tcfg.dataset, tcfg.data_root)
        series = tsdata.normalize(raw)
        rng = np.random.default_rng(tcfg.seed)
        draw = forecast_starts(rng, len(series), tcfg.L, tcfg.H, tcfg.pool_n, tcfg.val_n, tcfg.test_n, ocfg,
                               recent_validation=tcfg.recent_validation)
        Xp, Yp = tsdata.windows(series, draw["pool"], tcfg.L, tcfg.H)
        Xval, Yval = tsdata.windows(series, draw["val"], tcfg.L, tcfg.H)
        Xt, Yt = tsdata.windows(series, draw["test"], tcfg.L, tcfg.H)
        noise = tcfg.noise_frac if ocfg.robustness.injection_ratio < 0 else ocfg.robustness.injection_ratio
        if ocfg.robustness.mechanism_set == "unseen":
            Xp, Yp, tags = unseen_windows(Xp, Yp, tcfg.seed, noise, scale=tcfg.drift_scale, bias=tcfg.drift_bias)
        else:
            Xp, Yp, tags = inject_windows(Xp, Yp, tcfg.seed, noise)
        if draw["layout"] == "time_blocks":
            splits = block_splits(draw["blocks"]["counts"])
        else:
            splits = validation_splits(len(Xval), tcfg.seed, ocfg)
        n = len(Xp)
        records = [UnifiedRecord(id=str(i), modality=Modality.TEXT, domain="timeseries", text="") for i in range(n)]
        s0 = float(np.abs(Yp - Xp[:, -1:]).mean())
        block_steps = ocfg.gate.ts_block_steps or (tcfg.L + tcfg.H)
        # the rng stream continues after the split draws. The canonical influence reference reads it
        return TaskData(
            pool_ids=[int(i) for i in draw["pool"]], val_ids=[int(i) for i in draw["val"]],
            test_ids=[int(i) for i in draw["test"]], tags=tags, splits=splits, budget=int(tcfg.budget_frac * n),
            mechanisms=([str(t) if t != "high" else "" for t in tags]
                        if ocfg.robustness.mechanism_set == "unseen" else None),
            utility=self.utility, records=records, blocks=draw["blocks"], s0=s0, block_steps=block_steps,
            arrays={"Xp": Xp, "Yp": Yp, "Xval": Xval, "Yval": Yval, "Xt": Xt, "Yt": Yt, "rng": rng,
                    "val_starts": draw["val"], "test_starts": draw["test"]},
            provenance={"series_source": path, "series_len": int(len(series)), "layout": draw["layout"],
                        "L": tcfg.L, "H": tcfg.H},
        )

    def signals(self, data: TaskData, tcfg: ForecastConfig, ocfg) -> Signals:
        a = data.arrays
        Xp, Yp = a["Xp"], a["Yp"]
        auth = np.array([abs(autocorr1(x)) for x in Xp])
        Xn = l2_normalize(Xp)
        _, redundancy = knn_agreement_and_novelty(Xn, None, tcfg.knn)
        device = resolve_device(tcfg.device)
        if ocfg.influence.reference == "v_con":
            con = data.splits.con[: min(len(data.splits.con), tcfg.influence_ref_n)]
            ref_X, ref_Y = a["Xval"][con], a["Yval"][con]
            ref_ids = [data.val_ids[int(i)] for i in con]
            source = "v_con"
        else:
            ref = a["rng"].permutation(np.where(data.tags == "high")[0])[: tcfg.influence_ref_n]
            ref_X, ref_Y = Xp[ref], Yp[ref]
            ref_ids = [data.pool_ids[int(i)] for i in ref]
            source = "pool_clean_tag"
        learner = self.learner(data, tcfg, ocfg)
        if learner.kind.startswith("chronos"):
            from tracks.timeseries.chronos import train_chronos

            ref_model = train_chronos(ref_X, ref_Y, device, tcfg.seed, tcfg.reference_epochs, tcfg)
        else:
            ref_model = train_dlinear(ref_X, ref_Y, device, tcfg.seed, tcfg.reference_epochs)
        influence = -np.abs(predict(ref_model, Xp, device) - Yp).mean(axis=1)
        console = MultiActorConsole(
            [("redundancy", RedundancySignal()), ("influence", InfluenceSignal())],
            weights=np.log(np.array([1 - tcfg.w_infl, tcfg.w_infl]) + 1e-9),
            conflict_gate=True, anneal=0.6, group_key="domain", trust_region=0.5, ema_beta=0.3, min_weight=0.02,
        )
        imp_dyn = console.importance(data.records, scores=np.stack([minmax(redundancy), minmax(influence)]),
                                     progress=0.5)
        extras = {"imp_dyn": imp_dyn}
        if tcfg.learner == "dlinear":
            extras["last_layer"] = _last_layer_dlinear(data, tcfg, device)
        saved = {}
        if {"el2n_adapt", "grand_adapt", "ccs_adapt"} & set(tcfg.methods):
            if tcfg.learner != "dlinear":
                raise ValueError("forecast score adaptations require the disclosed DLinear checkpoint")
            from benchmark.Methods.ScoreAdapt.method import squared_error_scores

            early = extras["last_layer"]()
            scores = squared_error_scores(early.phi, early.phi @ early.theta, early.target)
            extras["adapt_scores"] = {**scores, "info": {
                "score_source": "pool-warmed DLinear", "score_epochs": tcfg.last_layer_warm,
                "score_seed": tcfg.seed, "selection_reads": "pool inputs and observed pool targets only",
                "el2n_definition": "L2 norm of multi-horizon residual",
                "grand_definition": "parameter-gradient Frobenius norm of half squared residual loss",
                "initializations": 1,
            }}
            saved = {"el2n_adapt": scores["el2n"], "grand_adapt": scores["grand"]}
        return Signals(auth=auth, influence=influence, redundancy=redundancy, features=Xn.astype(float),
                       reference_ids=ref_ids, reference_source=source, extras=extras, saved=saved)

    def alignment(self, data: TaskData, sig: Signals, tcfg: ForecastConfig, ocfg):
        """Alignment from DLinear warm-started on the pool (linear in [trend, 1, seasonal, 1], learner independent)."""
        from tracks.common.alignment import from_last_layer

        return from_last_layer(_last_layer_dlinear(data, tcfg, resolve_device(tcfg.device))(), "DLinear")

    def cooperative_inputs(self, data: TaskData, sig: Signals, tcfg: ForecastConfig, ocfg) -> dict:
        """Euclidean geometry of the L2-normalized windows (the selector features). Cleanliness features
        from the input windows and their targets."""
        a, con = data.arrays, data.splits.con
        return {"veto_features": sig.features, "metric": "euclidean", "con_veto_features": l2_normalize(a["Xval"][con]),
                "selector_features": sig.features,
                "cleanliness": {"kind": "forecasting", "Xp": a["Xp"], "Yp": a["Yp"], "Xc": a["Xval"][con],
                                "Yc": a["Yval"][con]}}

    def strategy_overrides(self, data, sig, tcfg, ocfg):
        def auth2_only(ctx, k):
            def rank01(v):
                r = np.argsort(np.argsort(v, kind="stable"), kind="stable")
                return r / (len(v) - 1 + 1e-9)
            inlier = 1.0 - sig.redundancy
            score = np.minimum(rank01(sig.auth), rank01(inlier))
            return [int(i) for i in np.argsort(-score, kind="stable")[:k]]
        return {"auth2_only": auth2_only}

    def learner(self, data: TaskData, tcfg: ForecastConfig, ocfg) -> ForecastLearner:
        a = data.arrays
        paired = ocfg.paired_rng
        seeds = {stage: fit_seed(tcfg.seed, name, paired, reset=False)
                 for stage, name in (("con", "v1-fit"), ("rank", "v2-fit"), ("report", "final-fit"))}
        if tcfg.learner.startswith("chronos"):
            plan = stage_plan(ocfg, scoring={"epochs": min(tcfg.chronos_epochs, tcfg.chronos_max_epochs)},
                              reported={"epochs": min(tcfg.chronos_epochs, tcfg.chronos_max_epochs)}, seeds=seeds,
                              schedule_key="epochs", low={"epochs": tcfg.chronos_screen_epochs})
        else:
            plan = stage_plan(ocfg, scoring={"epochs": tcfg.scoring_epochs}, reported={"epochs": tcfg.epochs},
                              seeds=seeds, schedule_key="epochs", low={"epochs": tcfg.screen_epochs})
        if paired:
            names = {"con": "v1-fit", "rank": "v2-fit", "conf": "v2-fit", "member": "v1-fit", "report": "final-fit",
                     "screen": "v1-fit"}
            for stage, stage_fid in plan.stages.items():
                label = "final-fit" if ocfg.fidelity.scoring_equals_reported else names[stage]
                stage_fid["reset_rng"] = [tcfg.seed, label]
        vs, ts = a["val_starts"], a["test_starts"]
        split_arrays = {name: (a["Xval"][data.splits.get(name)], a["Yval"][data.splits.get(name)],
                               vs[data.splits.get(name)]) for name in ("con", "rank", "conf")}
        split_arrays["test"] = (a["Xt"], a["Yt"], ts)
        return ForecastLearner(plan, tcfg.learner, a["Xp"], a["Yp"], split_arrays, resolve_device(tcfg.device), tcfg)
