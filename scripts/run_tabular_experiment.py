"""Tabular arm: data selection for a tabular foundation model (TabPFN v2), the SAME
unified framework instantiated on tabular signals. Selection = choosing TabPFN's
in-context support set under a fixed budget (faithful to the in-context-selection
framing of ICD-TabPFN / Tab-AICL), NOT gradient training.

Pipeline mirrors the vision arm:
  1. quality-variance pool from a real OpenML table: 60% clean rows + 40% controlled
     noise (label-flip / feature-corruption / near-duplicate), each tagged;
  2. three orthogonal channels on the standardized features (same fusion controller):
       authenticity = kNN label agreement, influence = -loss of a clean-reference probe,
       redundancy   = feature-space novelty;
  3. each method picks a budgeted support set; TabPFN predicts the held-out test split;
  4. report test ROC AUC + accuracy, multiple seeds.

M2-feasible: TabPFN forward-pass inference only (no GPU training). Env: TAB_DATASET,
POOL_N, TEST_N, NOISE_FRAC, BUDGET_FRAC, KNN, SEED, METHODS.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # avoid OpenMP runtime clash (xgboost vs torch)

DATASET = os.environ.get("TAB_DATASET", "electricity")
MODEL = os.environ.get("MODEL", "tabpfn")
_VALID_MODELS = {"tabpfn", "xgboost", "rf"}
if MODEL not in _VALID_MODELS:
    raise SystemExit(f"invalid MODEL={MODEL!r}; expected one of {sorted(_VALID_MODELS)}")   # tabpfn (robust FM) | xgboost (noise-sensitive, ablation)
POOL_N = int(os.environ.get("POOL_N", "3000"))
TEST_N = int(os.environ.get("TEST_N", "2000"))
NOISE_FRAC = float(os.environ.get("NOISE_FRAC", "0.40"))
BUDGET_FRAC = float(os.environ.get("BUDGET_FRAC", "0.5"))
KNN = int(os.environ.get("KNN", "15"))
VAL_N = int(os.environ.get("VAL_N", "2500"))   # bigger independent val -> stable config selection
SEED = int(os.environ.get("SEED", "0"))
METHODS = os.environ.get("METHODS", "full,random,coreset,auth_only,influence_only,mmdataselect,mmds_adapt").split(",")
LAM = float(os.environ.get("LAM", "0.5"))
AUTH_Q = float(os.environ.get("AUTH_Q", "0.25"))
W_INFL = float(os.environ.get("W_INFL", "0.5"))
PAIRED_RNG = os.environ.get("PAIRED_RNG", "0") == "1"


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


def load_table(seed):
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    d = fetch_openml(DATASET, version=1, as_frame=True)
    X = d.data.select_dtypes(include="number").to_numpy(dtype=float)
    X = np.nan_to_num(X)
    y_raw = d.target.to_numpy()
    classes = {c: i for i, c in enumerate(sorted(set(y_raw)))}
    y = np.array([classes[c] for c in y_raw])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    # pool gets noise injected; val + test stay CLEAN and NATURAL (same distribution), disjoint.
    pool_i = perm[:POOL_N]
    val_i = perm[POOL_N : POOL_N + VAL_N]
    test_i = perm[POOL_N + VAL_N : POOL_N + VAL_N + TEST_N]
    _export_split_ids("tabular", seed, {
        "arm": "tabular", "seed": int(seed), "dataset": DATASET,
        "n_source_rows": int(len(X)),
        "pool_ids": [int(i) for i in pool_i],
        "val_ids": [int(i) for i in val_i],
        "test_ids": [int(i) for i in test_i],
        "counts": {"pool": int(len(pool_i)), "val": int(len(val_i)), "test": int(len(test_i))},
        "rng_recipe": ("rng=np.random.default_rng(seed); perm=rng.permutation(len(X)); "
                       "pool_ids=perm[:POOL_N]; val_ids=perm[POOL_N:POOL_N+VAL_N]; "
                       "test_ids=perm[POOL_N+VAL_N:POOL_N+VAL_N+TEST_N]; "
                       f"POOL_N={POOL_N}, VAL_N={VAL_N}, TEST_N={TEST_N}; "
                       "ids index the row order of sklearn fetch_openml('" + DATASET + "', version=1) "
                       "numeric-column matrix"),
    })
    sc = StandardScaler().fit(X[pool_i])
    return (sc.transform(X[pool_i]), y[pool_i], sc.transform(X[val_i]), y[val_i],
            sc.transform(X[test_i]), y[test_i], len(classes))


def inject_noise(X, labels, seed, n_classes):
    rng = np.random.default_rng(seed + 7)
    n = len(labels)
    obs = labels.copy()
    Xn = X.copy()
    tag = np.array(["high"] * n, dtype=object)
    n_low = int(round(NOISE_FRAC * n))
    low = rng.permutation(n)[:n_low]
    per = max(1, n_low // 3)
    flip, corr, dup = low[:per], low[per : 2 * per], low[2 * per :]
    for i in flip:                              # label-flip -> authenticity
        obs[i] = rng.integers(n_classes)
        tag[i] = "flip"
    for i in corr:                              # feature corruption -> authenticity/influence
        Xn[i] = Xn[i] + rng.standard_normal(Xn.shape[1]) * 2.0
        tag[i] = "corrupt"
    if len(dup):                                # near-duplicate -> redundancy
        seeds = dup[: max(1, len(dup) // 8)]
        for j, i in enumerate(dup):
            Xn[i] = Xn[seeds[j % len(seeds)]] + 0.01 * rng.standard_normal(Xn.shape[1])
            tag[i] = "dup"
    return Xn, obs, tag




def _trial_dump(results, arm, dataset, seed, extra_cfg):
    """Isolated per-trial artifact: outputs/{arm}/{dataset}/{tags}/seed_{seed}/results.json,
    written atomically (tmp + os.replace), with full config metadata so trials from
    parallel lanes can never clobber each other."""
    import hashlib as _h
    import json as _j
    import tempfile as _tf
    tags = "-".join(f"{k}={v}" for k, v in sorted(extra_cfg.items()) if v not in ("", None)) or "base"
    _rid = os.environ.get("RUN_ID", "")
    if _rid:
        tags = f"run_id={_rid}-" + tags
    d = os.path.join(_REPO, "outputs", arm, str(dataset).replace("/", "_"), tags, f"seed_{seed}")
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
    os.replace(tmp, os.path.join(d, "results.json"))
    return os.path.join(d, "results.json")

def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    # TabPFN (and its torch dep) is imported lazily only when MODEL=tabpfn, so the xgboost
    # ablation never loads torch -> no OpenMP runtime clash / segfault.

    from mmdataselect.datatypes import Modality, UnifiedRecord
    from mmdataselect.fusion.console import MultiActorConsole
    from mmdataselect.selectors.budget_select import BudgetSelector
    from mmdataselect.selectors.external_baselines import (
        ccs, d4, density_select, dmf_dynamic, dmf_published_update, dsdm_scores,
        el2n, grand_expected, herding, kcenter_greedy, quadmix,
        quadmix_published_core, semdedup)
    from mmdataselect.signals import minmax
    from mmdataselect.signals import InfluenceSignal, RedundancySignal, minmax
    from mmdataselect.utils.pairing import arrays_sha256, order_sha12, reset_rng, sel_sha12, stable_seed

    import sys as _sys
    _sys.path.insert(0, _REPO)
    from baselines.tab_aicl.method import tabpfn_coreset, tabpfn_hybrid, tabpfn_margin  # Tab-AICL (Ma et al. 2026)

    Xp, yp_clean, Xval, yval, Xt, yt, n_classes = load_table(SEED)
    Xp, obs_lab, tag = inject_noise(Xp, yp_clean, SEED, n_classes)
    n = len(Xp)
    budget = int(BUDGET_FRAC * n)
    rng = np.random.default_rng(SEED)

    # normalized features for similarity/diversity
    Xn = Xp / (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-8)
    # Chunked kNN (audit #OOM): a full n x n similarity matrix would be tens of GB at pool
    # scales in the hundred-thousands (see baselines/deepcore_original/run_original_protocol.py's
    # matching fix / run_vision_experiment.py). n stays small here today so this was latent, not
    # yet hit, but the fix is a straight port of the same chunked, argpartition-based rule
    # (mathematically identical to the unchunked S/argsort version -- verified by direct
    # comparison on random data).
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
    ref = rng.permutation(np.where(tag == "high")[0])[:300]
    ref_clf = LogisticRegression(max_iter=200).fit(Xp[ref], obs_lab[ref])
    proba = ref_clf.predict_proba(Xp)
    cls_idx = {c: k for k, c in enumerate(ref_clf.classes_)}
    influence = np.array([np.log(proba[i, cls_idx[obs_lab[i]]] + 1e-9) if obs_lab[i] in cls_idx else -20.0
                          for i in range(n)])
    proba_full = np.zeros((n, n_classes)); proba_full[:, ref_clf.classes_] = proba  # for EL2N / GraNd

    cand = np.arange(n)  # independent clean val (Xval/yval) means the whole pool is selectable
    feats = Xn.astype(float)
    recs = [UnifiedRecord(id=str(i), modality=Modality.TEXT, domain="table", text="") for i in range(n)]
    console = MultiActorConsole(
        [("redundancy", RedundancySignal()), ("influence", InfluenceSignal())],
        weights=np.log(np.array([1 - W_INFL, W_INFL]) + 1e-9),
        conflict_gate=True, anneal=0.6, group_key="domain", trust_region=0.5, ema_beta=0.3, min_weight=0.02,
    )
    imp_dyn = console.importance(recs, scores=np.stack([minmax(redundancy), minmax(influence)], axis=0), progress=0.5)

    def _tabpfn(sub):
        # tabpfn = robust in-context FM; xgboost = noise-sensitive tree model (ablation: shows the
        # best signal depends on the MODEL, not just the modality — auth helps the noise-sensitive one).
        if MODEL == "xgboost":
            from xgboost import XGBClassifier
            c = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=4,
                              tree_method="hist", eval_metric="logloss", verbosity=0)
            c.fit(Xp[sub], obs_lab[sub])
            return c
        if MODEL == "rf":
            # post-experiment reproducibility bugfix (audit 1910): rf was accepted by _VALID_MODELS
            # but fell through to the TabPFN import; no archived result used MODEL=rf, so no
            # historical number changes
            from sklearn.ensemble import RandomForestClassifier
            c = RandomForestClassifier(n_estimators=200, max_depth=4, n_jobs=4, random_state=SEED)
            c.fit(Xp[sub], obs_lab[sub])
            return c
        from tabpfn import TabPFNClassifier   # lazy: only loads torch for the tabpfn path
        import torch as _t
        _tabdev = "cuda" if _t.cuda.is_available() else "cpu"
        c = TabPFNClassifier.create_default_for_version("v2", device=_tabdev, ignore_pretraining_limits=True)
        c.fit(Xp[sub], obs_lab[sub])
        return c

    _vperm = np.random.default_rng(SEED + 41).permutation(len(yval))
    _v1, _v2 = _vperm[: len(_vperm) // 2], _vperm[len(_vperm) // 2 :]
    Xval1, yval1, Xval2, yval2 = Xval[_v1], yval[_v1], Xval[_v2], yval[_v2]

    def _auc_on(sub, Xv, yv):
        from sklearn.metrics import roc_auc_score
        p = _tabpfn(sub).predict_proba(Xv)
        try:
            return float(roc_auc_score(yv, p[:, 1]) if p.shape[1] == 2
                         else roc_auc_score(yv, p, multi_class="ovr"))
        except Exception:
            return float((p.argmax(1) == yv).mean())

    # stratified halves of V2 (each half keeps every class) so AUC is well-defined per half
    _rng43 = np.random.default_rng(SEED + 43)
    _s1, _s2 = [], []
    for _c in np.unique(yval2):
        _idx = _rng43.permutation(np.where(yval2 == _c)[0])
        _s1 += list(_idx[: len(_idx) // 2]); _s2 += list(_idx[len(_idx) // 2 :])
    _s1, _s2 = np.array(_s1, dtype=int), np.array(_s2, dtype=int)

    def _probe_val_acc(sub):   # ADJUDICATION gain on V2 only
        if os.environ.get("ROBUST_VAL", "0") == "1":
            a1, a2 = _auc_on(sub, Xval2[_s1], yval2[_s1]), _auc_on(sub, Xval2[_s2], yval2[_s2])
            return float(min(a1, a2))
        return _auc_on(sub, Xval2, yval2)

    def _probe_val_acc1(sub):  # CONSTRUCTION gain on V1 only
        return _auc_on(sub, Xval1, yval1)

    _aicl = {}
    def _aicl_probs():   # TabPFN forward-pass class probs on the pool, for Tab-AICL margin/hybrid
        if "p" not in _aicl:
            srng = np.random.default_rng(SEED + 99)
            seed_sub = srng.permutation(n)[:min(300, budget)]
            _aicl["p"] = _tabpfn(seed_sub).predict_proba(Xp)
        return _aicl["p"]

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
            k = min(budget, n)
            km = KMeans(n_clusters=k, n_init=3, random_state=SEED).fit(Xn)
            order = []
            for c in range(k):
                m = np.where(km.labels_ == c)[0]
                if len(m):
                    order.append(int(m[np.argmin(np.linalg.norm(Xn[m] - km.cluster_centers_[c], axis=1))]))
            return order[:budget]
        if method == "auth_only":
            return list(np.argsort(-auth)[:budget])
        if method == "auth2_only":      # mechanism-matched v2, label arm (min(oof, knn))
            from mmdataselect.signals.authenticity_v2 import auth_label
            return list(np.argsort(-auth_label(Xn, obs_lab, seed=SEED))[:budget])
        if method == "auth3_only":      # v2 full: label arm AND corruption (inlier) arm
            from mmdataselect.signals.authenticity_v2 import auth_full
            return list(np.argsort(-auth_full(Xn, obs_lab, seed=SEED))[:budget])
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
            w = dsdm_scores(_probe_val_acc1, n, k_runs=int(os.environ.get("DSDM_RUNS", "12")), seed=SEED)
            return [int(i) for i in np.argsort(-w)[:budget]]
        if method == "dmf":              # DMF faithful: dynamic multi-channel reweighting (Yang et al. 2025)
            ch = np.stack([minmax(auth), minmax(influence), minmax(redundancy)], axis=0)
            return dmf_dynamic(ch, budget, val_reward=_probe_val_acc1, rounds=4, seed=SEED)
        if method == "dmf_pub":          # Multi-Actor Eqs. 6--8, published-update transfer
            ch = np.stack([minmax(auth), minmax(influence), minmax(redundancy)], axis=0)
            return dmf_published_update(ch, budget, val_reward=_probe_val_acc1, rounds=6, seed=SEED)
        if method == "tabpfn_coreset":   # Tab-AICL representativeness (Ma et al. 2026, direct TabPFN-selection prior)
            return tabpfn_coreset(Xn, budget, seed=SEED)
        if method == "tabpfn_margin":    # Tab-AICL uncertainty acquisition (TabPFN prediction margin)
            return tabpfn_margin(_aicl_probs(), budget)
        if method == "tabpfn_hybrid":    # Tab-AICL representativeness + informativeness
            return tabpfn_hybrid(Xn, _aicl_probs(), budget, seed=SEED)
        if method == "mmdataselect":
            thr = float(np.quantile(auth, AUTH_Q))
            imp = imp_dyn.copy()
            imp[auth < thr] = -1e9
            return BudgetSelector(lam=LAM).select(recs, imp, budget, features=feats)
        if method == "mmds_adapt":       # ADAPTIVE controller (framework core), TabPFN held-out gain
            from mmdataselect.fusion.adaptive import AdaptiveController

            # smaller fusion grid keeps the TabPFN-evaluated portfolio fast; the exact baselines
            # (passed below) carry the pure-channel + coverage candidates.
            ctrl = AdaptiveController(weight_grid=[(1, 0, 0), (0, .5, .5), (.34, .33, .33)],
                                      lam_grid=(0.0, 0.6), prefilter_grid=(0.0, AUTH_Q),
                                      prefilter_channel=0, seed=SEED)
            sel = ctrl.select(recs, np.stack([auth, influence, redundancy], axis=0), budget,
                              features=feats, held_out_gain=_probe_val_acc,
                              construct_gain=_probe_val_acc1,
                              policy_search=(os.environ.get("ADAPT_GRPO", "0") == "1"),
                              extra_strategies=[(b, (lambda bb: (lambda k: select(bb)))(b)) for b in
                                                ("mmdataselect", "auth2_only", "auth3_only",
                                                 "herding", "kcenter", "el2n", "grand", "ccs",
                                                 "semdedup", "density", "quadmix_pub",  # style-proxy quadmix removed from portfolio (PROTOCOL_INVALID)
                                                 "dmf", "dmf_pub",
                                                 "tabpfn_coreset", "tabpfn_margin", "tabpfn_hybrid",
                                                 "d4", "dsdm")]
                                                + [("random", lambda k: select("random")),
                                                   ("auth_bottom", lambda k: [int(i) for i in np.argsort(auth)[:k]])])
            print(f"    [adapt] picked '{ctrl.chosen_['strategy']}' (val_auc={ctrl.chosen_['val_gain']:.3f})")
            globals()["_ADAPT_MANIFEST"] = {"leaderboard": list(getattr(ctrl, "leaderboard_", []) or []), "chosen": dict(getattr(ctrl, "chosen_", {}) or {}), "sel_sha12": __import__("hashlib").sha256(str(sorted(int(i) for i in sel)).encode()).hexdigest()[:12] if "sel" in dir() else None}
            return sel
        raise ValueError(method)

    print(f"tabular={DATASET} model={MODEL} | pool={n} test={len(Xt)} budget={budget} classes={n_classes} seed={SEED}")
    print(f"  tags: {dict(zip(*np.unique(tag, return_counts=True)))}")
    globals()["_PAIRING_MANIFEST"] = {
        "pool_sha256": arrays_sha256(Xp, obs_lab, tag.astype(str)),
        "validation_sha256": arrays_sha256(Xval, yval),
        "test_sha256": arrays_sha256(Xt, yt),
        "shared_initialization_rule": "stable_seed(paper_seed, final-fit)",
        "training_input_order": "sorted selected integer ids",
    }
    results = []
    _sel_only = os.environ.get("SELECT_ONLY", "0") == "1"
    for m in METHODS:
        t0 = time.time()
        sel = [int(i) for i in select(m)]
        if PAIRED_RNG:
            sel = sorted(sel)
        if _sel_only:  # selection-manifest replay (audit item 二): NO training, full IDs
            results.append({"method": m, "n_selected": len(sel), "selected_ids": sel,
                            "training_order": sel, "sel_sha12": sel_sha12(sel),
                            "train_order_sha12": order_sha12(sel)})
            print(f"  [select-only] {m:16} n={len(sel)} sel={sel_sha12(sel)}")
            continue
        fit_seed = reset_rng(SEED, "final-fit") if PAIRED_RNG else SEED
        clf = _tabpfn(sel)   # MODEL-dispatched downstream model (tabpfn / xgboost)
        proba_t = clf.predict_proba(Xt)
        pred = proba_t.argmax(1)
        acc = float((pred == yt).mean())
        try:
            auc = float(roc_auc_score(yt, proba_t[:, 1]) if n_classes == 2 else
                        roc_auc_score(yt, proba_t, multi_class="ovr"))
        except Exception:
            auc = float("nan")
        hi = float(np.mean(tag[sel] == "high"))
        row = {"method": m, "n": len(sel), "clean%": round(hi, 3), "acc": round(acc, 4), "auc": round(auc, 4)}
        if PAIRED_RNG:
            row.update({"sel_sha12": sel_sha12(sel), "fit_seed": fit_seed,
                        "train_order_sha12": order_sha12(sel)})
        results.append(row)
        print(f"  {m:16} n={len(sel):5} clean%={hi:.2f} acc={acc:.4f} auc={auc:.4f} ({time.time()-t0:.1f}s)")
    print(f"\n==== TABULAR ({DATASET}, AUC higher better) ====")
    for r in results:
        if "auc" not in r:
            continue  # select-only manifest rows carry no metrics
        print(f"  {r['method']:16} auc={r['auc']:.4f} acc={r['acc']:.4f} clean%={r['clean%']:.2f} n={r['n']}")
    _trial_dump(results, "tabular", DATASET, SEED,
                {"model": MODEL, "noise_frac": NOISE_FRAC, "pool": POOL_N,
                 "budget": BUDGET_FRAC, "paired_rng": int(PAIRED_RNG)})
    if not os.environ.get("RUN_ID"):
        out = os.path.join(_REPO, "outputs", "tabular")
        os.makedirs(out, exist_ok=True)
        import json
        json.dump(results, open(os.path.join(out, f"results_{DATASET}_seed{SEED}.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
