"""Vision arm: data selection for an image model, the SAME framework instantiated on
image signals (NOT text). Demonstrates the unified selector is modality-agnostic.

Pipeline (mirrors the text experiment, but everything is image-native):
  1. quality-variance pool from CIFAR-100: 60% clean (image,label) + 40% controlled
     noise (label-flip / near-duplicate / hard-ambiguous), each tagged;
  2. encode every image once with a frozen vision encoder (CLIP or DINOv2);
  3. three orthogonal channels on the embeddings, reusing the SAME fusion controller
     and budget selector as text:
       authenticity = kNN label agreement (mislabeled/corrupt -> low),
       influence    = -loss of a probe fit on a small CLEAN reference (clean/on-task high),
       redundancy   = embedding novelty (1 - mean cosine sim to neighbors);
  4. each method picks a fixed-budget subset; train a linear probe on the selected
     (embedding,label) pairs; report top-1 accuracy on the clean held-out test set;
  5. multiple seeds, mean +- std.

M2-feasible: frozen-encoder forward pass + sklearn LogisticRegression probe, no GPU
training loop. Env: VIS_ENCODER, POOL_N, TEST_N, NOISE_FRAC, BUDGET_FRAC, KNN, SEED, METHODS.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ENCODER = os.environ.get("VIS_ENCODER", "openai/clip-vit-base-patch32")
POOL_N = int(os.environ.get("POOL_N", "4000"))
TEST_N = int(os.environ.get("TEST_N", "2000"))
NOISE_FRAC = float(os.environ.get("NOISE_FRAC", "0.40"))
BUDGET_FRAC = float(os.environ.get("BUDGET_FRAC", "0.5"))
KNN = int(os.environ.get("KNN", "15"))
VAL_N = int(os.environ.get("VAL_N", "800"))   # independent clean val for the controller's config search
SEED = int(os.environ.get("SEED", "0"))
METHODS = os.environ.get("METHODS", "full,random,coreset,influence_only,auth_only,mmdataselect").split(",")
LAM = float(os.environ.get("LAM", "0.5"))
AUTH_Q = float(os.environ.get("AUTH_Q", "0.25"))
W_INFL = float(os.environ.get("W_INFL", "0.5"))
PAIRED_RNG = os.environ.get("PAIRED_RNG", "0") == "1"
SAVE_DOWNSTREAM_CHECKPOINTS = os.environ.get("SAVE_DOWNSTREAM_CHECKPOINTS", "1") == "1"


def device():
    import torch
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


DATASET = os.environ.get("VIS_DATASET", "uoft-cs/cifar100")  # cifar100 | uoft-cs/cifar10


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


def load_cifar(seed):
    from datasets import load_dataset
    ds = load_dataset(DATASET, split="train")
    te = load_dataset(DATASET, split="test")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds))
    tr_idx = perm[:POOL_N]
    val_idx = perm[POOL_N : POOL_N + VAL_N]     # independent CLEAN val (disjoint from pool), for the controller
    te_idx = rng.permutation(len(te))[:TEST_N]
    _export_split_ids("vision", seed, {
        "arm": "vision", "seed": int(seed), "dataset": DATASET,
        "n_train_source": int(len(ds)), "n_test_source": int(len(te)),
        "pool_ids": [int(i) for i in tr_idx],
        "val_ids": [int(i) for i in val_idx],
        "test_ids": [int(i) for i in te_idx],
        "counts": {"pool": int(len(tr_idx)), "val": int(len(val_idx)), "test": int(len(te_idx))},
        "rng_recipe": ("rng=np.random.default_rng(seed); perm=rng.permutation(len(train_split)); "
                       "pool_ids=perm[:POOL_N]; val_ids=perm[POOL_N:POOL_N+VAL_N]; "
                       "test_ids=rng.permutation(len(test_split))[:TEST_N]; "
                       f"POOL_N={POOL_N}, VAL_N={VAL_N}, TEST_N={TEST_N}; "
                       "ids index the HF datasets train/test splits of " + DATASET),
    })
    img_key = "img" if "img" in ds.column_names else "image"
    lab_key = "fine_label" if "fine_label" in ds.column_names else "label"
    pool_imgs = [ds[int(i)][img_key] for i in tr_idx]
    pool_lab = np.array([ds[int(i)][lab_key] for i in tr_idx])
    val_imgs = [ds[int(i)][img_key] for i in val_idx]
    val_lab = np.array([ds[int(i)][lab_key] for i in val_idx])
    test_imgs = [te[int(i)][img_key] for i in te_idx]
    test_lab = np.array([te[int(i)][lab_key] for i in te_idx])
    vn = float(os.environ.get("VAL_NOISE", "0"))
    if vn > 0:
        # contaminated-validation experiment (Prop 6): flip a fraction of val labels to a
        # RANDOM WRONG class; the controller is not told which ones are corrupted.
        rngv = np.random.default_rng(10_000 + int(vn * 100))
        m = rngv.random(len(val_lab)) < vn
        val_lab = val_lab.copy()
        wrong = (val_lab[m] + rngv.integers(1, 10, m.sum())) % int(max(pool_lab.max(), val_lab.max()) + 1)
        val_lab[m] = wrong
    return pool_imgs, pool_lab, val_imgs, val_lab, test_imgs, test_lab


def encode(imgs, dev):
    import torch
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(ENCODER)
    model = AutoModel.from_pretrained(ENCODER).to(dev).eval()
    def _to_tensor(o):
        if torch.is_tensor(o):
            return o
        for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
            v = getattr(o, attr, None)
            if v is not None:
                return v.mean(1) if attr == "last_hidden_state" else v
        raise TypeError(f"cannot extract features from {type(o)}")

    feats = []
    with torch.no_grad():
        for s in range(0, len(imgs), 128):
            batch = imgs[s : s + 128]
            inp = proc(images=batch, return_tensors="pt").to(dev)
            if hasattr(model, "get_image_features"):
                f = model.get_image_features(pixel_values=inp["pixel_values"])
            else:
                f = model(pixel_values=inp["pixel_values"])
            feats.append(_to_tensor(f).float().cpu().numpy())
    X = np.concatenate(feats, 0)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return X


def inject_noise(labels, seed, n_classes=100):
    """Build a quality-variance label set: 60% clean, 40% noise (flip/dup/hard), tagged."""
    rng = np.random.default_rng(seed + 7)
    n = len(labels)
    obs = labels.copy()
    tag = np.array(["high"] * n, dtype=object)
    n_low = int(round(NOISE_FRAC * n))
    low_idx = rng.permutation(n)[:n_low]
    per = max(1, n_low // 3)
    flip, dup, hard = low_idx[:per], low_idx[per : 2 * per], low_idx[2 * per :]
    for i in flip:                       # label-flip noise -> authenticity should catch
        obs[i] = rng.integers(n_classes)
        tag[i] = "flip"
    for i in dup:                        # near-duplicate marker -> redundancy should catch
        tag[i] = "dup"
    for i in hard:                       # ambiguous/low-value -> influence should down-rank
        tag[i] = "hard"
    return obs, tag, dup




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
    import torch
    from sklearn.linear_model import LogisticRegression

    from mmdataselect.utils.repro_bundle import save_downstream_checkpoint
    from mmdataselect.datatypes import Modality, UnifiedRecord
    from mmdataselect.fusion.console import MultiActorConsole
    from mmdataselect.selectors.budget_select import BudgetSelector
    from mmdataselect.selectors.external_baselines import (
        ccs, d4, density_select, dmf_dynamic, dmf_published_update, dsdm_scores,
        el2n, grand_expected, herding, kcenter_greedy, quadmix,
        quadmix_published_core, semdedup)
    from mmdataselect.signals import InfluenceSignal, RedundancySignal, minmax
    from mmdataselect.utils.pairing import arrays_sha256, order_sha12, reset_rng, sel_sha12, stable_seed

    dev = device()
    cache = os.path.join(_REPO, f"data/processed/vision_{DATASET.split('/')[-1]}_{ENCODER.split('/')[-1]}_p{POOL_N}v{VAL_N}t{TEST_N}_s{SEED}.npz")
    pool_imgs, pool_lab, val_imgs, val_lab, test_imgs, test_lab = load_cifar(SEED)
    if os.path.exists(cache) and "Xval" in np.load(cache, allow_pickle=True):
        z = np.load(cache, allow_pickle=True)
        Xp, Xval, Xt = z["Xp"], z["Xval"], z["Xt"]
        print(f"[encode] cached ({len(Xp)} pool, {len(Xval)} val, {len(Xt)} test)")
    else:
        print(f"[encode] {ENCODER} on {len(pool_imgs)}+{len(val_imgs)}+{len(test_imgs)} imgs (dev={dev}) ...")
        Xp = encode(pool_imgs, dev)
        Xval = encode(val_imgs, dev)
        Xt = encode(test_imgs, dev)
        np.savez(cache, Xp=Xp, Xval=Xval, Xt=Xt)
    n = len(Xp)
    if os.environ.get("VIS_NOISE", "inject") == "real":
        # REAL human label noise (CIFAR-100N, Wei et al. ICLR 2022): replace the injected
        # noise pool with genuine crowd-sourced mislabels (~40.2% noise rate). No synthetic
        # mechanism is injected at all; tags come from noisy-vs-clean comparison.
        import torch as _t
        _dn = _t.load(os.path.join(_REPO, "data/cifar_n/CIFAR-100_human.pt"), weights_only=False)
        _noisy = np.array(_dn["noisy_label"]); _clean = np.array(_dn["clean_label"])
        _rng = np.random.default_rng(SEED)
        _tr_idx = _rng.permutation(len(_clean))[:POOL_N]
        assert (pool_lab == _clean[_tr_idx]).all(), "CIFAR-N alignment failed"
        obs_lab = _noisy[_tr_idx].astype(pool_lab.dtype)
        tag = np.where(obs_lab != pool_lab, "flip", "high").astype(object)
        dup_idx = np.array([], dtype=int)
        print(f"[real-noise] CIFAR-100N human labels: noise rate {float((obs_lab != pool_lab).mean()):.3f}")
    else:
        obs_lab, tag, dup_idx = inject_noise(pool_lab, SEED)
    budget = int(BUDGET_FRAC * n)

    # near-duplicate: copy a few seeds' embeddings onto the 'dup' slots (so redundancy bites)
    if len(dup_idx) > 0:
        seeds = dup_idx[: max(1, len(dup_idx) // 8)]
        for j, i in enumerate(dup_idx):
            Xp[i] = Xp[seeds[j % len(seeds)]] + 0.01 * np.random.default_rng(i).standard_normal(Xp.shape[1])
        Xp /= (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-8)

    # ---- three image-native channels ----
    # Chunked kNN (audit #OOM): a full n x n similarity matrix is 57.6GB at n=120000 (float32),
    # plus another ~115GB for a full argsort -- this OOM-killed a container with a 72GB cgroup
    # limit with no Python traceback (see baselines/deepcore_original/run_original_protocol.py's
    # matching fix). Currently n stays in the low thousands here so this was latent, not yet hit,
    # but the fix is a straight port of the same chunked, argpartition-based rule (mathematically
    # identical to the unchunked S/argsort version -- verified by direct comparison on random data).
    _chunk = 2048
    auth = np.zeros(n, dtype=np.float64)
    redundancy = np.zeros(n, dtype=np.float64)
    for _s0 in range(0, n, _chunk):
        _sims = Xp[_s0:_s0 + _chunk] @ Xp.T
        for _r in range(_sims.shape[0]):
            _sims[_r, _s0 + _r] = -1.0                              # drop self (matches fill_diagonal(-1.0))
        _idx = np.argpartition(-_sims, KNN, axis=1)[:, :KNN]        # k nearest neighbours
        _rows = np.arange(_sims.shape[0])[:, None]
        auth[_s0:_s0 + _chunk] = (obs_lab[_idx] == obs_lab[_s0:_s0 + _chunk, None]).mean(axis=1)
        redundancy[_s0:_s0 + _chunk] = 1.0 - _sims[_rows, _idx].mean(axis=1)
    # influence = -loss of a probe fit on a small CLEAN reference subset (clean/on-task -> high)
    rng = np.random.default_rng(SEED)
    ref = rng.permutation(np.where(tag == "high")[0])[:400]
    ref_clf = LogisticRegression(max_iter=200, C=1.0).fit(Xp[ref], obs_lab[ref])
    proba = ref_clf.predict_proba(Xp)
    classes = ref_clf.classes_
    cls_idx = {c: k for k, c in enumerate(classes)}
    influence = np.array([np.log(proba[i, cls_idx[obs_lab[i]]] + 1e-9) if obs_lab[i] in cls_idx else -20.0
                          for i in range(n)])
    # full-width probe probabilities (columns aligned to label ids) for EL2N / GraNd baselines
    n_classes = int(max(int(obs_lab.max()), int(val_lab.max()), int(test_lab.max()))) + 1
    proba_full = np.zeros((n, n_classes))
    proba_full[:, classes] = proba

    cand = np.arange(n)  # independent clean val (Xval/val_lab) -> the whole pool is selectable
    feats = Xp.astype(float)
    recs = [UnifiedRecord(id=str(i), modality=Modality.IMAGE if hasattr(Modality, "IMAGE") else Modality.TEXT,
                          domain="image", text="") for i in range(n)]
    console = MultiActorConsole(
        [("redundancy", RedundancySignal()), ("influence", InfluenceSignal())],
        weights=np.log(np.array([1 - W_INFL, W_INFL]) + 1e-9),
        conflict_gate=True, anneal=0.6, group_key="domain", trust_region=0.5, ema_beta=0.3, min_weight=0.02,
    )
    imp_dyn = console.importance(recs, scores=np.stack([minmax(redundancy), minmax(influence)], axis=0), progress=0.5)

    # V1/V2 split: ADAPTIVE construction (SH / vote / coordinate ascent / policy search)
    # consumes V1 only; the adjudication gain consumes V2 only (winner's-curse fix).
    _vperm = np.random.default_rng(SEED + 41).permutation(len(val_lab))
    _v1, _v2 = _vperm[: len(_vperm) // 2], _vperm[len(_vperm) // 2 :]
    Xval1, val_lab1 = Xval[_v1], val_lab[_v1]
    Xval2, val_lab2 = Xval[_v2], val_lab[_v2]

    _r2 = np.random.default_rng(SEED + 43).permutation(len(val_lab2))
    _s1, _s2 = _r2[: len(_r2) // 2], _r2[len(_r2) // 2 :]

    def gain(sub):                   # ADJUDICATION gain on V2 only
        clf = LogisticRegression(max_iter=150, C=1.0).fit(Xp[sub], obs_lab[sub])
        pred = clf.predict(Xval2)
        if os.environ.get("ROBUST_VAL", "0") == "1":
            # robust segment-wise adjudication (Prop 8): worst sub-segment gain certifies a
            # lower bound on any test distribution in the segments' convex hull.
            return float(min((pred[_s1] == val_lab2[_s1]).mean(), (pred[_s2] == val_lab2[_s2]).mean()))
        return float((pred == val_lab2).mean())

    def gain1(sub):                  # CONSTRUCTION gain on V1 only
        clf = LogisticRegression(max_iter=150, C=1.0).fit(Xp[sub], obs_lab[sub])
        return float((clf.predict(Xval1) == val_lab1).mean())

    def select(method):
        if PAIRED_RNG:
            reset_rng(SEED, "select", method)
        if method == "full":
            return list(range(n))
        if method == "random":
            rr = np.random.default_rng(stable_seed(SEED, "select", method)) if PAIRED_RNG else rng
            return list(rr.permutation(n)[:budget])
        if method == "coreset":          # k-means coreset (SemDeDup-style diversity baseline)
            from sklearn.cluster import KMeans
            k = min(budget, n)
            km = KMeans(n_clusters=k, n_init=3, random_state=SEED).fit(Xp)
            order = []
            for c in range(k):
                members = np.where(km.labels_ == c)[0]
                if len(members):
                    d = np.linalg.norm(Xp[members] - km.cluster_centers_[c], axis=1)
                    order.append(members[np.argmin(d)])
            return order[:budget]
        if method == "auth_only":
            return list(np.argsort(-auth)[:budget])
        if method == "auth2_only":      # mechanism-matched authenticity v2 (label arm: min(oof, knn))
            from mmdataselect.signals.authenticity_v2 import auth_label
            return list(np.argsort(-auth_label(Xp, obs_lab, knn=KNN, seed=SEED))[:budget])
        if method == "influence_only":
            return list(np.argsort(-influence)[:budget])
        if method == "herding":          # geometric coreset (Welling 2009 / DeepCore)
            return herding(Xp, budget)
        if method == "kcenter":          # k-center greedy coreset (Sener & Savarese 2018 / DeepCore)
            return kcenter_greedy(Xp, budget, seed=SEED)
        if method == "el2n":             # EL2N score-based pruning (Paul et al. 2021 / DeepCore)
            return el2n(proba_full, obs_lab, budget, is_logits=False)
        if method == "grand":            # GraNd faithful = expected gradient norm over early probes (Paul et al. 2021)
            return grand_expected(Xp, obs_lab, budget, seed=SEED)
        if method == "ccs":              # Coverage-centric Coreset Selection (Zheng et al., ICLR 2023)
            return ccs(proba_full, obs_lab, budget, is_logits=False)
        if method == "semdedup":         # semantic deduplication (Abbas et al. 2023)
            return semdedup(Xp, budget, seed=SEED)
        if method == "density":          # Density coverage sampler (Sachdeva et al. 2024, Ask-LLM+Density)
            return density_select(Xp, budget)
        if method == "quadmix":          # QuaDMix-style quality x diversity joint selection (2024)
            return quadmix(auth, Xp, budget, seed=SEED)
        if method == "quadmix_pub":      # QuaDMix Eqs. 1--3, fixed-budget transfer
            return quadmix_published_core(auth, Xp, budget, seed=SEED)
        if method == "d4":               # D4: SemDeDup then prototype diversification (Tirumala et al. 2023)
            return d4(Xp, budget, seed=SEED)
        if method == "dsdm":             # DsDm proxy-scale datamodels (Engstrom et al. 2024); probe metric on V1 only
            w = dsdm_scores(gain1, n, k_runs=int(os.environ.get("DSDM_RUNS", "20")), seed=SEED)
            return [int(i) for i in np.argsort(-w)[:budget]]
        if method == "dmf":              # DMF faithful: dynamic multi-channel reweighting (Yang et al. 2025)
            ch = np.stack([minmax(auth), minmax(influence), minmax(redundancy)], axis=0)
            return dmf_dynamic(ch, budget, val_reward=gain1, seed=SEED)
        if method == "dmf_pub":          # Multi-Actor Eqs. 6--8, published-update transfer
            ch = np.stack([minmax(auth), minmax(influence), minmax(redundancy)], axis=0)
            return dmf_published_update(ch, budget, val_reward=gain1, rounds=6, seed=SEED)
        if method == "mmds_noauth":
            return BudgetSelector(lam=LAM).select(recs, imp_dyn, budget, features=feats)
        if method == "mmdataselect":     # authenticity prefilter -> influence x diversity
            thr = float(np.quantile(auth, AUTH_Q))
            imp = imp_dyn.copy()
            imp[auth < thr] = -1e9
            return BudgetSelector(lam=LAM).select(recs, imp, budget, features=feats)
        if method == "mmds_adapt":       # ADAPTIVE controller (framework core): picks channel
            from mmdataselect.fusion.adaptive import AdaptiveController   # weights+diversity by held-out gain

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
            ctrl = AdaptiveController(lam_grid=(0.0, 0.25, 0.6), prefilter_grid=(0.0, AUTH_Q), seed=SEED)
            # portfolio includes the exact baselines too -> controller is >= each by construction
            extras = [(b, (lambda bb: (lambda k: select(bb)))(b)) for b in
                      ("coreset", "auth_only", "auth2_only", "influence_only", "mmdataselect", "herding",
                       "kcenter", "el2n", "grand", "ccs", "semdedup", "density",
                       "quadmix_pub", "dmf", "dmf_pub",  # style-proxy quadmix removed from portfolio (PROTOCOL_INVALID, docs/quadmix_styleproxy_invalidation.md)
                       "d4", "dsdm")]
            extras.append(("random", lambda k: select("random")))
            # per-channel REVERSE candidate: realizability for the identification argument --
            # in boundary-regime environments the anti-authenticity direction is the good one
            extras.append(("auth_bottom", lambda k: [int(i) for i in np.argsort(auth)[:k]]))
            cheap = None
            if os.environ.get("ADAPT_SH", "0") == "1":
                # low-fidelity gain for successive halving (Theorem B): probe trained on a
                # 40% subsample of the selection with fewer iterations, same independent val.
                def cheap(sub, _r=np.random.default_rng(SEED + 7)):
                    sub = np.asarray(list(sub))
                    m2 = max(50, int(0.4 * len(sub)))
                    idx = sub[_r.permutation(len(sub))[:m2]]
                    c2 = LogisticRegression(max_iter=60, C=1.0).fit(Xp[idx], obs_lab[idx])
                    return float((c2.predict(Xval1) == val_lab1).mean())
            sel = ctrl.select(recs, np.stack([auth, influence, redundancy], axis=0), budget,
                              features=feats, held_out_gain=gain, extra_strategies=extras,
                              cheap_gain=cheap, construct_gain=gain1,
                              policy_search=(os.environ.get("ADAPT_GRPO", "0") == "1"))
            if ctrl.sh_stats_:
                print(f"    [sh] fusions {ctrl.sh_stats_['fusions_total']} -> {ctrl.sh_stats_['finalists']} "
                      f"(cheap evals {ctrl.sh_stats_['cheap_evals']})")
            print(f"    [adapt] picked '{ctrl.chosen_['strategy']}' (val={ctrl.chosen_['val_gain']:.3f})")
            globals()["_ADAPT_MANIFEST"] = {"leaderboard": list(getattr(ctrl, "leaderboard_", []) or []), "chosen": dict(getattr(ctrl, "chosen_", {}) or {}), "sel_sha12": __import__("hashlib").sha256(str(sorted(int(i) for i in sel)).encode()).hexdigest()[:12] if "sel" in dir() else None}
            return sel
        raise ValueError(method)

    print(f"vision | encoder={ENCODER} pool={n} test={len(Xt)} budget={budget} seed={SEED}")
    print(f"  pool tags: {dict(zip(*np.unique(tag, return_counts=True)))}")
    globals()["_PAIRING_MANIFEST"] = {
        "pool_sha256": arrays_sha256(Xp, obs_lab, tag.astype(str)),
        "validation_sha256": arrays_sha256(Xval, val_lab),
        "test_sha256": arrays_sha256(Xt, test_lab),
        "shared_initialization_rule": "stable_seed(paper_seed, final-fit)",
        "training_input_order": "sorted selected integer ids",
    }
    globals()["_REPRO_SELECTIONS"] = {}
    globals()["_REPRO_PREDICTIONS"] = {}
    globals()["_REPRO_SELECTION_SOURCE"] = {
        "features": Xp,
        "observed_labels": obs_lab,
        "clean_labels": pool_lab,
        "quality_tags": tag.astype(str),
    }
    globals()["_REPRO_EVALUATION_DATA"] = {
        "validation_features": Xval,
        "validation_labels": val_lab,
        "test_features": Xt,
        "test_labels": test_lab,
    }
    run_config = {
        "noise_frac": NOISE_FRAC,
        "vis_noise": os.environ.get("VIS_NOISE", "inject"),
        "drop": os.environ.get("DROP_CHANNEL", ""),
        "pool": POOL_N,
        "budget": BUDGET_FRAC,
        "val_n": VAL_N,
        "paired_rng": int(PAIRED_RNG),
    }
    trial_dir = _trial_dir("vision", DATASET, SEED, run_config)
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
        clf = LogisticRegression(max_iter=300, C=1.0, random_state=fit_seed).fit(Xp[sel], obs_lab[sel])
        pred = clf.predict(Xt)
        acc = float((pred == test_lab).mean())
        globals()["_REPRO_PREDICTIONS"][m] = {
            "y_true": test_lab,
            "y_pred": pred,
        }
        checkpoint_path = None
        if SAVE_DOWNSTREAM_CHECKPOINTS:
            checkpoint_path = save_downstream_checkpoint(
                clf,
                os.path.join(trial_dir, "checkpoints", str(m).replace("/", "_")),
                metadata={
                    "arm": "vision",
                    "dataset": str(DATASET),
                    "seed": int(SEED),
                    "method": str(m),
                    "fit_seed": int(fit_seed),
                    "selection_sha12": sel_sha12(sel),
                    "config": run_config,
                },
            )
            globals()["_REPRO_CHECKPOINT_PATHS"][m] = checkpoint_path
        hi = float(np.mean(tag[sel] == "high"))
        row = {"method": m, "n": len(sel), "clean%": round(hi, 3), "acc": round(acc, 4)}
        if checkpoint_path is not None:
            row["checkpoint_path"] = str(checkpoint_path)
        if PAIRED_RNG:
            row.update({"sel_sha12": sel_sha12(sel), "fit_seed": fit_seed,
                        "train_order_sha12": order_sha12(sel)})
        results.append(row)
        print(f"  {m:16} n={len(sel):5} clean%={hi:.2f} test_acc={acc:.4f} ({time.time()-t0:.1f}s)")
    print("\n==== VISION (CIFAR-100 top-1 acc, higher better) ====")
    for r in results:
        if "acc" not in r:
            continue  # select-only manifest rows carry no metrics
        print(f"  {r['method']:16} acc={r['acc']:.4f} clean%={r['clean%']:.2f} n={r['n']}")
    _trial_dump(results, "vision", DATASET, SEED, run_config)
    out = os.path.join(_REPO, "outputs", "vision")
    os.makedirs(out, exist_ok=True)
    import json
    if not os.environ.get("RUN_ID"):
        json.dump(results, open(os.path.join(out, f"results_seed{SEED}.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
