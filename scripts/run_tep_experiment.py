"""Process-industry arm: data selection for a fault-diagnosis model on the Tennessee
Eastman Process (TEP), the de-facto benchmark for chemical-process monitoring and fault
detection/diagnosis (hundreds of papers since Downs & Vogel 1993; the Braatz simulation
files used here are the standard ML-ready version).

Task: 52 process variables (41 measured + 11 manipulated) -> classify normal vs fault
1..21 (22 classes). This is the SAME unified data-selection framework instantiated on a
recognized industrial benchmark with established baselines (PCA/PLS/SVM/RF fault diagnosis).

Pipeline mirrors the other arms:
  1. quality-variance pool from the TEP training files (d00 normal + d01..d21 faults), 60%
     clean + 40% controlled noise (label-flip / sensor-corruption / near-duplicate);
  2. three channels on the standardized 52-d features (same fusion controller): authenticity
     = kNN label agreement, influence = -loss of a clean-reference classifier, redundancy =
     feature-space novelty;
  3. each method picks a budgeted training subset; train a RandomForest fault classifier;
     report macro-F1 + accuracy on the held-out TEP test files (post-fault samples);
  4. multiple seeds. M2-feasible (RF + sklearn). Env: POOL_N, TEST_N, VAL_N, NOISE_FRAC,
     BUDGET_FRAC, SEED, METHODS, N_FAULTS.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

TEP_DIR = os.path.join(_REPO, "data/tep")
N_FAULTS = int(os.environ.get("N_FAULTS", "21"))      # use faults 1..N_FAULTS (+ normal)
POOL_N = int(os.environ.get("POOL_N", "4000"))
TEST_N = int(os.environ.get("TEST_N", "3000"))
VAL_N = int(os.environ.get("VAL_N", "2000"))
NOISE_FRAC = float(os.environ.get("NOISE_FRAC", "0.40"))
BUDGET_FRAC = float(os.environ.get("BUDGET_FRAC", "0.3"))
KNN = int(os.environ.get("KNN", "15"))
SEED = int(os.environ.get("SEED", "0"))
METHODS = os.environ.get("METHODS", "full,random,coreset,auth_only,influence_only,mmdataselect,mmds_adapt").split(",")
LAM = float(os.environ.get("LAM", "0.5"))
AUTH_Q = float(os.environ.get("AUTH_Q", "0.25"))
W_INFL = float(os.environ.get("W_INFL", "0.5"))
PAIRED_RNG = os.environ.get("PAIRED_RNG", "0") == "1"
SAVE_DOWNSTREAM_CHECKPOINTS = os.environ.get("SAVE_DOWNSTREAM_CHECKPOINTS", "1") == "1"


def _read(fname):
    a = np.loadtxt(os.path.join(TEP_DIR, fname))
    return a.T if a.shape[0] == 52 else a   # d00.dat is stored 52 x 500 (transposed)


def _export_split_ids(arm, seed, payload):
    """Env-gated split-id manifest dump (POST_DATA_LOCK audit item 四.3): when
    SPLIT_EXPORT_DIR is set, persist the exact seeded split indices + rng recipe
    used by this run, then continue normally. No effect when the env var is unset."""
    d = os.environ.get("SPLIT_EXPORT_DIR", "")
    if not d:
        return
    import json as _j
    sub = os.path.join(d, arm)
    os.makedirs(sub, exist_ok=True)
    path = os.path.join(sub, f"split_ids_seed{seed}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        _j.dump(payload, fh, indent=2)
    os.replace(tmp, path)
    print(f"[split-export] wrote {path}")


def load_tep(seed):
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte = [], [], [], []
    Xtr.append(_read("d00.dat")); ytr.append(np.zeros(len(Xtr[-1]), int))        # normal train
    Xte.append(_read("d00_te.dat")); yte.append(np.zeros(len(Xte[-1]), int))     # normal test
    for k in range(1, N_FAULTS + 1):
        tr = _read(f"d{k:02d}.dat"); Xtr.append(tr); ytr.append(np.full(len(tr), k, int))
        te = _read(f"d{k:02d}_te.dat")[160:]   # fault is introduced at sample 161
        Xte.append(te); yte.append(np.full(len(te), k, int))
    Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
    Xte = np.vstack(Xte); yte = np.concatenate(yte)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    pi = rng.permutation(len(Xtr))[:POOL_N]
    ti = rng.permutation(len(Xte)); vi, tei = ti[:VAL_N], ti[VAL_N : VAL_N + TEST_N]
    # dedicated detection-threshold calibration split (audit: >=200, ideally 500+, normal
    # samples, fully disjoint from validation/V1/V2 AND test): ALL remaining held-out te
    # indices beyond val+test. Pre-registered - same seeded permutation, original data
    # untouched, no re-splitting of val or test.
    ci = ti[VAL_N + TEST_N:]
    n_classes = N_FAULTS + 1
    _export_split_ids("tep", seed, {
        "arm": "tep", "seed": int(seed), "dataset": f"tep{N_FAULTS}",
        "n_train_source": int(len(Xtr)), "n_test_source": int(len(Xte)),
        "pool_ids": [int(i) for i in pi],
        "val_ids": [int(i) for i in vi],
        "test_ids": [int(i) for i in tei],
        "calibration_ids": [int(i) for i in ci],
        "counts": {"pool": int(len(pi)), "val": int(len(vi)), "test": int(len(tei)),
                   "calibration": int(len(ci))},
        "rng_recipe": ("rng=np.random.default_rng(seed); pool_ids=rng.permutation(len(Xtr))[:POOL_N]; "
                       "ti=rng.permutation(len(Xte)); val_ids=ti[:VAL_N]; "
                       "test_ids=ti[VAL_N:VAL_N+TEST_N]; calibration_ids=ti[VAL_N+TEST_N:]; "
                       f"POOL_N={POOL_N}, VAL_N={VAL_N}, TEST_N={TEST_N}, N_FAULTS={N_FAULTS}; "
                       "ids index the stacked TEP arrays: train = d00.dat + d01..dNN.dat in fault order, "
                       "test = d00_te.dat + d01_te..dNN_te.dat with fault files truncated to rows [160:]"),
    })
    return Xtr[pi], ytr[pi], Xte[vi], yte[vi], Xte[tei], yte[tei], Xte[ci], yte[ci], n_classes


def inject_noise(X, labels, seed, n_classes):
    rng = np.random.default_rng(seed + 7)
    n = len(labels)
    obs = labels.copy(); Xn = X.copy()
    tag = np.array(["high"] * n, dtype=object)
    n_low = int(round(NOISE_FRAC * n))
    low = rng.permutation(n)[:n_low]
    per = max(1, n_low // 3)
    flip, corr, dup = low[:per], low[per : 2 * per], low[2 * per :]
    for i in flip:
        obs[i] = rng.integers(n_classes); tag[i] = "flip"
    for i in corr:
        Xn[i] = Xn[i] + rng.standard_normal(X.shape[1]) * 2.0; tag[i] = "corrupt"
    if len(dup):
        seeds = dup[: max(1, len(dup) // 8)]
        for j, i in enumerate(dup):
            Xn[i] = Xn[seeds[j % len(seeds)]] + 0.01 * rng.standard_normal(X.shape[1]); tag[i] = "dup"
    return Xn, obs, tag




def _trial_dir(arm, dataset, seed, extra_cfg):
    tags = "-".join(f"{k}={v}" for k, v in sorted(extra_cfg.items()) if v not in ("", None)) or "base"
    run_id = os.environ.get("RUN_ID", "")
    if run_id:
        tags = f"run_id={run_id}-" + tags
    return os.path.join(
        _REPO,
        "outputs",
        arm,
        str(dataset).replace("/", "_"),
        tags,
        f"seed_{seed}",
    )


def _trial_dump(results, arm, dataset, seed, extra_cfg):
    """Isolated per-trial artifact: outputs/{arm}/{dataset}/{tags}/seed_{seed}/results.json,
    written atomically (tmp + os.replace), with full config metadata so trials from
    parallel lanes can never clobber each other."""
    import hashlib as _h
    import json as _j
    import tempfile as _tf
    _rid = os.environ.get("RUN_ID", "")
    d = _trial_dir(arm, dataset, seed, extra_cfg)
    os.makedirs(d, exist_ok=True)
    _baseline_path = os.path.join(_REPO, "src/mmdataselect/selectors/external_baselines.py")
    payload = {"arm": arm, "dataset": dataset, "seed": seed, "config": extra_cfg,
               "code_sha256_12": _h.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12],
               "baseline_impl_sha256": _h.sha256(open(_baseline_path, "rb").read()).hexdigest(),
               "fidelity_mode": os.environ.get("FIDELITY_MODE", "unified-protocol"),
               "published_core_protocol": {
                   "quadmix": {"equations": "1-3", "lambda": 100.0, "omega": 0.05,
                               "eta": 1.0, "epsilon": 0.001,
                               "domain_replacement": "kmeans-8-shared-representation",
                               "budget_adapter": "gumbel-top-k-without-replacement"},
                   "dmf": {"equations": "6-8", "rounds": 6, "eta": 0.5,
                           "post_update_constraint": "simplex-projection"},
               } if os.environ.get("FIDELITY_MODE", "").startswith("published-core") else None,
               "results": results, "adapt_manifest": globals().get("_ADAPT_MANIFEST"),
               "pairing_manifest": globals().get("_PAIRING_MANIFEST")}
    fd, tmp = _tf.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        _j.dump(payload, fh, indent=2)
    result_path = os.path.join(d, "results.json")
    os.replace(tmp, result_path)
    from mmdataselect.utils.repro_bundle import write_repro_bundle
    write_repro_bundle(
        d,
        repo_root=_REPO,
        runner_path=os.path.abspath(__file__),
        arm=arm,
        dataset=str(dataset),
        seed=seed,
        config={**extra_cfg, "run_id": _rid, "methods": METHODS},
        result_path=result_path,
        selections=globals().get("_REPRO_SELECTIONS", {}),
        selection_source=globals().get("_REPRO_SELECTION_SOURCE"),
        predictions=globals().get("_REPRO_PREDICTIONS"),
        split_manifest=globals().get("_PAIRING_MANIFEST"),
        evaluation_data=globals().get("_REPRO_EVALUATION_DATA"),
        checkpoint_paths=globals().get("_REPRO_CHECKPOINT_PATHS", {}),
    )
    return result_path

def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    from mmdataselect.utils.repro_bundle import save_downstream_checkpoint
    from mmdataselect.datatypes import Modality, UnifiedRecord
    from mmdataselect.fusion.console import MultiActorConsole
    from mmdataselect.fusion.adaptive import AdaptiveController
    from mmdataselect.selectors.budget_select import BudgetSelector
    from mmdataselect.selectors.external_baselines import (
        ccs, d4, density_select, dmf_dynamic, dmf_published_update, dsdm_scores,
        el2n, grand_expected, herding, kcenter_greedy, quadmix,
        quadmix_published_core, semdedup)
    from mmdataselect.signals import InfluenceSignal, RedundancySignal, minmax
    from mmdataselect.utils.pairing import arrays_sha256, order_sha12, reset_rng, sel_sha12, stable_seed

    Xp, yp_clean, Xval, yval, Xt, yt, Xcal, ycal, n_classes = load_tep(SEED)
    _vn = float(os.environ.get("VAL_NOISE", "0"))
    if _vn > 0:
        _rngv = np.random.default_rng(10_000 + int(_vn * 100))
        _m = _rngv.random(len(yval)) < _vn
        yval = yval.copy()
        yval[_m] = (yval[_m] + _rngv.integers(1, n_classes, _m.sum())) % n_classes
    Xp, obs_lab, tag = inject_noise(Xp, yp_clean, SEED, n_classes)
    n = len(Xp); budget = int(BUDGET_FRAC * n)
    rng = np.random.default_rng(SEED)

    Xn = Xp / (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-8)
    # Chunked kNN (audit #OOM): a full n x n similarity matrix is 57.6GB at n=120000 (float32),
    # plus another ~115GB for a full argsort -- this OOM-killed a container with a 72GB cgroup
    # limit with no Python traceback (see baselines/deepcore_original/run_original_protocol.py's
    # matching fix). Currently n stays small here so this was latent, not yet hit, but the fix is
    # a straight port of the same chunked, argpartition-based rule (mathematically identical to
    # the unchunked S/argsort version -- verified by direct comparison on random data).
    _chunk = 2048
    auth = np.zeros(n, dtype=np.float64)
    redundancy = np.zeros(n, dtype=np.float64)
    for _s0 in range(0, n, _chunk):
        _sims = Xn[_s0:_s0 + _chunk] @ Xn.T
        for _r in range(_sims.shape[0]):
            _sims[_r, _s0 + _r] = -1.0                              # drop self (matches fill_diagonal(-1.0))
        _idx = np.argpartition(-_sims, KNN, axis=1)[:, :KNN]        # k nearest neighbours
        _rows = np.arange(_sims.shape[0])[:, None]
        auth[_s0:_s0 + _chunk] = (obs_lab[_idx] == obs_lab[_s0:_s0 + _chunk, None]).mean(axis=1)
        redundancy[_s0:_s0 + _chunk] = 1.0 - _sims[_rows, _idx].mean(axis=1)
    ref = rng.permutation(np.where(tag == "high")[0])[:600]
    ref_clf = LogisticRegression(max_iter=300, C=1.0).fit(Xp[ref], obs_lab[ref])
    proba = ref_clf.predict_proba(Xp); cls_idx = {c: k for k, c in enumerate(ref_clf.classes_)}
    influence = np.array([np.log(proba[i, cls_idx[obs_lab[i]]] + 1e-9) if obs_lab[i] in cls_idx else -20.0
                          for i in range(n)])
    proba_full = np.zeros((n, n_classes)); proba_full[:, ref_clf.classes_] = proba  # for EL2N / GraNd

    feats = Xn.astype(float)
    recs = [UnifiedRecord(id=str(i), modality=Modality.TEXT, domain="process", text="") for i in range(n)]
    console = MultiActorConsole(
        [("redundancy", RedundancySignal()), ("influence", InfluenceSignal())],
        weights=np.log(np.array([1 - W_INFL, W_INFL]) + 1e-9),
        conflict_gate=True, anneal=0.6, group_key="domain", trust_region=0.5, ema_beta=0.3, min_weight=0.02,
    )
    imp_dyn = console.importance(recs, scores=np.stack([minmax(redundancy), minmax(influence)], axis=0), progress=0.5)

    from sklearn.neural_network import MLPClassifier
    MODEL = os.environ.get("MODEL", "rf")   # rf (robust, bagging) | mlp (noise-sensitive) | cnn (1D-CNN over the 52 vars)
    CNN_EPOCHS = int(os.environ.get("CNN_EPOCHS", "80"))

    class _CNN1D:
        """A small 1D-CNN fault classifier over the 52 process variables (treated as a length-52
        single-channel signal). Standard deep fault-diagnosis architecture for TEP."""
        def __init__(self, seed=0, epochs=80):
            self.seed = seed; self.epochs = epochs

        def fit(self, X, y):
            import torch
            import torch.nn as nn
            torch.manual_seed(self.seed)
            self.classes_ = np.unique(y)
            ymap = {c: i for i, c in enumerate(self.classes_)}
            yi = np.array([ymap[v] for v in y]); d = X.shape[1]
            self.net = nn.Sequential(
                nn.Unflatten(1, (1, d)), nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Linear(32, len(self.classes_)))
            opt = torch.optim.Adam(self.net.parameters(), lr=1e-3); lossf = nn.CrossEntropyLoss()
            Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(yi, dtype=torch.long)
            self.net.train()
            for _ in range(self.epochs):
                opt.zero_grad(); loss = lossf(self.net(Xt), yt); loss.backward(); opt.step()
            return self

        def predict(self, X):
            import torch
            self.net.eval()
            with torch.no_grad():
                out = self.net(torch.tensor(X, dtype=torch.float32))
            return self.classes_[out.argmax(1).numpy()]

    def _fit(sub, model_seed=SEED):
        if MODEL == "cnn":
            return _CNN1D(seed=model_seed, epochs=CNN_EPOCHS).fit(Xp[sub], obs_lab[sub])
        if MODEL == "mlp":
            return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=120, early_stopping=False,
                                 random_state=model_seed).fit(Xp[sub], obs_lab[sub])
        if MODEL == "svm":               # classical TEP fault-diagnosis baseline
            from sklearn.svm import SVC
            return SVC(kernel="rbf", C=1.0, random_state=model_seed).fit(Xp[sub], obs_lab[sub])
        if MODEL == "knn":               # classical TEP fault-diagnosis baseline
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(n_neighbors=5).fit(Xp[sub], obs_lab[sub])
        if MODEL == "pca":               # PCA-based process monitoring (linear) + classifier head
            from sklearn.decomposition import PCA
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            return make_pipeline(PCA(n_components=min(20, Xp.shape[1]), random_state=model_seed),
                                 LogisticRegression(max_iter=300)).fit(Xp[sub], obs_lab[sub])
        return RandomForestClassifier(n_estimators=120, n_jobs=4, random_state=model_seed).fit(Xp[sub], obs_lab[sub])

    def _f1(clf, X, y):
        return float(f1_score(y, clf.predict(X), average="macro"))

    _vperm = np.random.default_rng(SEED + 41).permutation(len(yval))
    _v1, _v2 = _vperm[: len(_vperm) // 2], _vperm[len(_vperm) // 2 :]
    Xval1, yval1, Xval2, yval2 = Xval[_v1], yval[_v1], Xval[_v2], yval[_v2]

    _r2 = np.random.default_rng(SEED + 43).permutation(len(yval2))
    _s1, _s2 = _r2[: len(_r2) // 2], _r2[len(_r2) // 2 :]

    def gain(sub):                   # ADJUDICATION on V2
        m = _fit(sub)
        if os.environ.get("ROBUST_VAL", "0") == "1":
            return float(min(_f1(m, Xval2[_s1], yval2[_s1]), _f1(m, Xval2[_s2], yval2[_s2])))
        return _f1(m, Xval2, yval2)

    def gain1(sub):                  # CONSTRUCTION on V1
        return _f1(_fit(sub), Xval1, yval1)

    def select(method):
        if PAIRED_RNG:
            reset_rng(SEED, "select", method)
        if method == "full":
            return list(range(n))
        if method == "random":
            rr = np.random.default_rng(stable_seed(SEED, "select", method)) if PAIRED_RNG else rng
            return list(rr.permutation(n)[:budget])
        if method == "coreset":
            from sklearn.cluster import KMeans
            k = min(budget, n); km = KMeans(n_clusters=k, n_init=3, random_state=SEED).fit(Xn)
            out = []
            for c in range(k):
                m = np.where(km.labels_ == c)[0]
                if len(m):
                    out.append(int(m[np.argmin(np.linalg.norm(Xn[m] - km.cluster_centers_[c], axis=1))]))
            return out[:budget]
        if method == "auth_only":
            return list(np.argsort(-auth)[:budget])
        if method == "auth2_only":      # mechanism-matched v2, label arm (min(oof, knn))
            from mmdataselect.signals.authenticity_v2 import auth_label
            return list(np.argsort(-auth_label(Xp, obs_lab, seed=SEED))[:budget])
        if method == "auth3_only":      # v2 full: label arm AND corruption (inlier) arm
            from mmdataselect.signals.authenticity_v2 import auth_full
            return list(np.argsort(-auth_full(Xp, obs_lab, seed=SEED))[:budget])
        if method == "influence_only":
            return list(np.argsort(-influence)[:budget])
        if method == "herding":          # geometric coreset (Welling 2009 / DeepCore)
            return herding(Xn, budget)
        if method == "kcenter":          # k-center greedy coreset (Sener & Savarese 2018 / DeepCore)
            return kcenter_greedy(Xn, budget, seed=SEED)
        if method == "el2n":             # EL2N score-based pruning (Paul et al. 2021)
            return el2n(proba_full, obs_lab, budget, is_logits=False)
        if method == "grand":            # GraNd faithful = expected gradient norm over early probes (Paul et al. 2021)
            return grand_expected(Xn, obs_lab, budget, seed=SEED)
        if method == "ccs":              # Coverage-centric Coreset Selection (Zheng et al., ICLR 2023)
            return ccs(proba_full, obs_lab, budget, is_logits=False)
        if method == "semdedup":         # semantic deduplication (Abbas et al. 2023)
            return semdedup(Xn, budget, seed=SEED)
        if method == "density":          # Density coverage sampler (Sachdeva et al. 2024)
            return density_select(Xn, budget)
        if method == "quadmix":          # QuaDMix-style quality x diversity joint selection (2024)
            return quadmix(auth, Xn, budget, seed=SEED)
        if method == "quadmix_pub":      # QuaDMix Eqs. 1--3, fixed-budget transfer
            return quadmix_published_core(auth, Xn, budget, seed=SEED)
        if method == "d4":               # D4 (Tirumala et al. 2023): dedup then diversify
            return d4(feats, budget, seed=SEED)
        if method == "dsdm":             # DsDm proxy datamodels (Engstrom et al. 2024); probe on V1 only
            w = dsdm_scores(gain1, n, k_runs=int(os.environ.get("DSDM_RUNS", "16")), seed=SEED)
            return [int(i) for i in np.argsort(-w)[:budget]]
        if method == "dmf":              # DMF faithful: dynamic multi-channel reweighting (Yang et al. 2025)
            ch = np.stack([minmax(auth), minmax(influence), minmax(redundancy)], axis=0)
            return dmf_dynamic(ch, budget, val_reward=gain1, seed=SEED)
        if method == "dmf_pub":          # Multi-Actor Eqs. 6--8, published-update transfer
            ch = np.stack([minmax(auth), minmax(influence), minmax(redundancy)], axis=0)
            return dmf_published_update(ch, budget, val_reward=gain1, rounds=6, seed=SEED)
        if method == "mmdataselect":
            thr = float(np.quantile(auth, AUTH_Q)); imp = imp_dyn.copy(); imp[auth < thr] = -1e9
            return BudgetSelector(lam=LAM).select(recs, imp, budget, features=feats)
        if method == "mmds_adapt":
            _drop = os.environ.get("DROP_CHANNEL", "")
            if _drop:
                # channel-drop ablation: controller sees only the remaining two channels;
                # external baselines leave the portfolio so the ablation is interpretable;
                # dropping auth also disables the authenticity gate (it IS that channel).
                _chmap = {"auth": 0, "infl": 1, "red": 2}
                _keep = [v for k, v in _chmap.items() if k != _drop]
                _S3full = np.stack([auth, influence, redundancy], axis=0)
                _S3 = _S3full[_keep]
                ctrl = AdaptiveController(lam_grid=(0.0, 0.5),
                                          prefilter_grid=((0.0,) if _drop == "auth" else (0.0, AUTH_Q)),
                                          seed=SEED)
                _extras = [("random", lambda k, _r=np.random.default_rng(SEED + 1): list(_r.permutation(n)[:k]))]
                sel = ctrl.select(recs, _S3, budget, features=feats, held_out_gain=gain,
                                  extra_strategies=_extras, construct_gain=gain1)
                print(f"    [adapt-drop:{_drop}] picked '{ctrl.chosen_['strategy']}' (val={ctrl.chosen_['val_gain']:.3f})")
                globals()["_ADAPT_MANIFEST"] = {"leaderboard": list(getattr(ctrl, "leaderboard_", []) or []), "chosen": dict(getattr(ctrl, "chosen_", {}) or {}), "sel_sha12": __import__("hashlib").sha256(str(sorted(int(i) for i in sel)).encode()).hexdigest()[:12] if "sel" in dir() else None}
                return sel
            ctrl = AdaptiveController(weight_grid=[(1, 0, 0), (0, .5, .5), (.34, .33, .33)],
                                     lam_grid=(0.0, 0.5), prefilter_grid=(0.0, AUTH_Q), seed=SEED)
            extras = [(b, (lambda bb: (lambda k: select(bb)))(b)) for b in
                      ("coreset", "auth_only", "auth2_only", "auth3_only", "influence_only",
                       "mmdataselect", "herding", "kcenter", "el2n", "grand", "ccs",
                       "semdedup", "density", "quadmix_pub",  # style-proxy quadmix removed from portfolio (PROTOCOL_INVALID)
                       "dmf", "dmf_pub", "d4", "dsdm")]
            extras.append(("random", lambda k: select("random")))
            extras.append(("auth_bottom", lambda k: [int(i) for i in np.argsort(auth)[:k]]))
            sel = ctrl.select(recs, np.stack([auth, influence, redundancy], axis=0), budget,
                              features=feats, held_out_gain=gain, extra_strategies=extras,
                              construct_gain=gain1,
                              policy_search=(os.environ.get("ADAPT_GRPO", "0") == "1"))
            print(f"    [adapt] picked '{ctrl.chosen_['strategy']}' (val_f1={ctrl.chosen_['val_gain']:.3f})")
            globals()["_ADAPT_MANIFEST"] = {"leaderboard": list(getattr(ctrl, "leaderboard_", []) or []), "chosen": dict(getattr(ctrl, "chosen_", {}) or {}), "sel_sha12": __import__("hashlib").sha256(str(sorted(int(i) for i in sel)).encode()).hexdigest()[:12] if "sel" in dir() else None}
            return sel
        raise ValueError(method)

    print(f"TEP fault-diagnosis | pool={n} test={len(Xt)} budget={budget} classes={n_classes} seed={SEED}")
    print(f"  tags: {dict(zip(*np.unique(tag, return_counts=True)))}")
    globals()["_PAIRING_MANIFEST"] = {
        "pool_sha256": arrays_sha256(Xp, obs_lab, tag.astype(str)),
        "validation_sha256": arrays_sha256(Xval, yval),
        "test_sha256": arrays_sha256(Xt, yt),
        "shared_initialization_rule": "stable_seed(paper_seed, final-fit)",
        "training_input_order": "sorted selected integer ids",
    }
    globals()["_REPRO_SELECTIONS"] = {}
    globals()["_REPRO_PREDICTIONS"] = {}
    globals()["_REPRO_SELECTION_SOURCE"] = {
        "features": Xp,
        "observed_labels": obs_lab,
        "quality_tags": tag.astype(str),
    }
    globals()["_REPRO_EVALUATION_DATA"] = {
        "validation_features": Xval,
        "validation_labels": yval,
        "calibration_features": Xcal,
        "calibration_labels": ycal,
        "test_features": Xt,
        "test_labels": yt,
    }
    run_config = {
        "model": MODEL,
        "noise_frac": NOISE_FRAC,
        "drop": os.environ.get("DROP_CHANNEL", ""),
        "pool": POOL_N,
        "budget": BUDGET_FRAC,
        "paired_rng": int(PAIRED_RNG),
    }
    trial_dir = _trial_dir("tep", f"tep{N_FAULTS}", SEED, run_config)
    globals()["_REPRO_CHECKPOINT_PATHS"] = {}
    results = []
    _sel_only = os.environ.get("SELECT_ONLY", "0") == "1"
    for m in METHODS:
        t0 = time.time()
        sel = [int(i) for i in select(m)]
        if PAIRED_RNG:
            sel = sorted(sel)
        globals()["_REPRO_SELECTIONS"][m] = sel
        if _sel_only:  # selection-manifest replay (audit item 二): NO training, full IDs
            results.append({"method": m, "n_selected": len(sel), "selected_ids": sel,
                            "training_order": sel, "sel_sha12": sel_sha12(sel),
                            "train_order_sha12": order_sha12(sel)})
            print(f"  [select-only] {m:16} n={len(sel)} sel={sel_sha12(sel)}")
            continue
        fit_seed = reset_rng(SEED, "final-fit") if PAIRED_RNG else SEED
        clf = _fit(sel, fit_seed)
        pred = clf.predict(Xt)
        globals()["_REPRO_PREDICTIONS"][m] = {
            "y_true": yt,
            "y_pred": pred,
        }
        checkpoint_path = None
        if SAVE_DOWNSTREAM_CHECKPOINTS:
            checkpoint_path = save_downstream_checkpoint(
                clf,
                os.path.join(trial_dir, "checkpoints", str(m).replace("/", "_")),
                metadata={
                    "arm": "tep",
                    "dataset": f"tep{N_FAULTS}",
                    "seed": int(SEED),
                    "method": str(m),
                    "fit_seed": int(fit_seed),
                    "selection_sha12": sel_sha12(sel),
                    "config": run_config,
                },
            )
            globals()["_REPRO_CHECKPOINT_PATHS"][m] = checkpoint_path
        f1 = _f1(clf, Xt, yt); acc = float((pred == yt).mean())
        # FDR (fault detection rate, process-monitoring standard): among true-fault
        # samples, the fraction not classified as normal (class 0). FAR: among
        # true-normal samples, the fraction wrongly flagged as some fault.
        fmask = yt != 0
        fdr = float((pred[fmask] != 0).mean()) if fmask.any() else float("nan")
        far = float((pred[~fmask] != 0).mean()) if (~fmask).any() else float("nan")
        calib = {}
        if os.environ.get("TEP_CALIB", "0") == "1" and hasattr(clf, "predict_proba"):
            # audit: FDR must come with a VALIDATION-calibrated normal threshold, never
            # tuned on test. score = 1 - P(normal); threshold = the largest cut whose
            # validation FAR <= target. PRIMARY operating point: FAR<=5% (frozen
            # 2026-07-16 before the calibrated 3-seed rerun; 1%/10% secondary). The same
            # rule applies to every method symmetrically; test sees each frozen threshold
            # exactly once. Raw multiclass-argmax FDR/FAR above is diagnostic only and
            # must never be quoted as the detection operating point.
            from sklearn.metrics import (roc_auc_score, average_precision_score,
                                         balanced_accuracy_score, confusion_matrix)
            cls = list(clf.classes_)
            if 0 not in cls:
                # structural N/A: the method's selected training set contains no normal
                # (class-0) samples, so P(normal) and a detection threshold are undefined.
                # This is a property of the baseline (fault-heavy selection), documented,
                # not silently skipped.
                calib["na_reason"] = ("class 0 (normal) absent from selected training set; "
                                      "detection operating point undefined")
            if 0 in cls:
                k0 = cls.index(0)
                # calibration set = the V1 half of validation. The REPORTED strategy
                # decision comes from V2 (adjudication); calibrating the detection
                # threshold on V1 keeps threshold data disjoint from the data behind
                # the reported decision. Same rule for every method symmetrically.
                sv = 1.0 - clf.predict_proba(Xcal)[:, k0]
                st = 1.0 - clf.predict_proba(Xt)[:, k0]
                nv = np.sort(sv[ycal == 0])
                assert len(nv) >= 200, (
                    "calibration split has only %d normal samples (<200); cannot support "
                    "the validation-calibrated FAR target" % len(nv))
                bt = (yt != 0).astype(int)
                calib["calibration_set"] = ("dedicated held-out calibration split (all te "
                                            "indices beyond val+test; disjoint from "
                                            "validation/V1/V2 and test; pre-registered)")
                calib["n_cal_normal"] = int(len(nv)); calib["n_cal_fault"] = int((ycal != 0).sum())
                calib["n_test_normal"] = int((yt == 0).sum()); calib["n_test_fault"] = int((yt != 0).sum())
                for tgt in (0.01, 0.05, 0.10):
                    # EMPIRICAL threshold search (audit: plain quantile does not
                    # guarantee the cap under ties/discreteness): smallest unique
                    # score whose empirical calibration FAR <= target; assert it.
                    key = int(tgt * 100)
                    if len(nv):
                        uniq = np.unique(nv)
                        # FAR(t) = #(nv >= t)/n, decreasing in t
                        fars = (len(nv) - np.searchsorted(nv, uniq, side="left")) / len(nv)
                        ok = np.where(fars <= tgt)[0]
                        if len(ok):
                            thr = float(uniq[ok[0]]); vfar = float(fars[ok[0]])
                        else:  # even the max score misses the cap -> threshold above max
                            thr = float(np.nextafter(uniq[-1], np.inf)); vfar = 0.0
                    else:
                        thr, vfar = float("inf"), 0.0
                    assert vfar <= tgt + 1e-12, f"val FAR {vfar} exceeds cap {tgt}"
                    tf = st >= thr
                    calib[f"threshold@far{key}"] = round(thr, 6)
                    calib[f"val_far@far{key}"] = round(vfar, 4)
                    calib[f"val_fp@far{key}"] = int(round(vfar * len(nv)))
                    calib[f"fdr@far{key}"] = round(float(tf[yt != 0].mean()), 4)
                    calib[f"far@far{key}"] = round(float(tf[yt == 0].mean()), 4)
                    calib[f"balacc@far{key}"] = round(float(balanced_accuracy_score(bt, tf.astype(int))), 4)
                    if tgt == 0.05:  # primary point: full binary confusion matrix
                        calib["confusion@far5"] = confusion_matrix(bt, tf.astype(int)).tolist()
                calib["auroc"] = round(float(roc_auc_score(bt, st)), 4)
                calib["auprc"] = round(float(average_precision_score(bt, st)), 4)
                calib["primary_operating_point"] = ("validation-calibrated 5% FAR target "
                                                    "(empirical threshold on the dedicated "
                                                    "calibration split, frozen pre-test)")
                print(f"    [calib] {m}: " + str({k: v for k, v in calib.items() if not k.startswith(('n_', 'confusion'))}))
        hi = float(np.mean(tag[sel] == "high"))
        row = {"method": m, "n": len(sel), "clean%": round(hi, 3), "f1": round(f1, 4),
               "acc": round(acc, 4), "fdr": round(fdr, 4), "far": round(far, 4), "calib": calib}
        if checkpoint_path is not None:
            row["checkpoint_path"] = str(checkpoint_path)
        if PAIRED_RNG:
            row.update({"sel_sha12": sel_sha12(sel), "fit_seed": fit_seed,
                        "train_order_sha12": order_sha12(sel)})
        results.append(row)
        print(f"  {m:16} n={len(sel):5} clean%={hi:.2f} macroF1={f1:.4f} acc={acc:.4f} FDR={fdr:.4f} FAR={far:.4f} ({time.time()-t0:.1f}s)")
    print(f"\n==== TEP (macro-F1 higher better; FDR fault detection / FAR false alarm) ====")
    for r in sorted(results, key=lambda r: -r.get("f1", 0)):
        if "f1" not in r:
            continue  # select-only manifest rows carry no metrics
        print(f"  {r['method']:16} F1={r['f1']:.4f} acc={r['acc']:.4f} FDR={r['fdr']:.4f} FAR={r['far']:.4f} clean%={r['clean%']:.2f} n={r['n']}")
    _trial_dump(results, "tep", f"tep{N_FAULTS}", SEED, run_config)
    out = os.path.join(_REPO, "outputs", "tep"); os.makedirs(out, exist_ok=True)
    import json
    if not os.environ.get("RUN_ID"):
        json.dump(results, open(os.path.join(out, f"results_seed{SEED}.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
