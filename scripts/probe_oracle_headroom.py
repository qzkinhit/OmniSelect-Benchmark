"""Diagnostic probe: where is the real headroom of the selection problem?

For each modality (vision CIFAR-100 from cached CLIP embeddings, TEP from local .dat), measure

  oracle        select only true-clean samples (uses the hidden tags -- upper bound)
  auth (r1)     current single-round kNN-label-agreement authenticity, top-budget
  auth (r2/r3)  ITERATED authenticity: fit a probe on the provisionally clean set, rescore by
                combining kNN agreement with the probe's per-sample agreement, re-rank, repeat
  random        lower reference

and report: selected clean fraction, downstream metric, and the noise-detection AUROC of each
scorer. This tells us (a) how much headroom is left between the current method and the oracle,
(b) whether the binding constraint is noise-detection quality, and (c) whether cheap iterative
refinement closes the gap. Pure diagnosis -- nothing here ships until verified on the bench.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SEED = int(os.environ.get("SEED", "0"))
NOISE_FRAC = 0.40
KNN = 15


def _auroc(score, is_noise):
    """AUROC of (-score) as a noise detector (low score should mean noise)."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(is_noise.astype(int), -score))


def _chunked_knn(X, k, chunk=2048):
    """Each row's top-k nearest-neighbour indices by cosine/dot similarity,
    without ever materializing the full n x n similarity matrix.

    A naive `S = X @ X.T` followed by `np.argsort(-S, axis=1)[:, :k]` is a
    latent OOM landmine: at pool scale n it needs an (n, n) float32 matrix
    (n^2 * 4 bytes) plus another full argsort index array, e.g. ~57.6GB +
    ~115GB at n=120000. This processes rows in blocks, holding only a
    (chunk, n) slab at a time, and uses argpartition (only needs the top-k
    unordered, not a full sort) to pick each row's neighbours. Numerically
    identical (same top-k *set* per row, order not guaranteed -- which is
    fine since every consumer here only reduces over the set, e.g. a mean).
    """
    n = X.shape[0]
    nbr = np.empty((n, k), dtype=np.int64)
    for s0 in range(0, n, chunk):
        s1 = min(s0 + chunk, n)
        sims = X[s0:s1] @ X.T
        for r in range(s1 - s0):
            sims[r, s0 + r] = -1.0  # self-exclusion sentinel (below any real cosine sim)
        nbr[s0:s1] = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    return nbr


def strong_auth(X, obs, knn_nbr, seed, folds=5):
    """Mechanism-matched authenticity estimator (the candidate upgrade).

    label flips   -> out-of-fold probe agreement: K-fold cross-validated predicted probability
                     of the OBSERVED label (confident-learning principle; out-of-fold so the
                     noise never scores itself).
    corruption    -> feature-space kNN-distance outlier score (corrupted rows sit far away).
    (near-dups are the coverage channel's job, not authenticity's.)
    Combine by rank-mean so no scale dominates.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    n = len(obs)
    oof = np.zeros(n)
    y_safe = obs.copy()
    # stratified folds need >=2 per class; fall back to plain KFold on failure
    try:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, y_safe)
        splits = list(splitter)
    except Exception:
        from sklearn.model_selection import KFold
        splits = list(KFold(n_splits=folds, shuffle=True, random_state=seed).split(X))
    for tr, va in splits:
        clf = LogisticRegression(max_iter=200, C=1.0).fit(X[tr], obs[tr])
        proba = clf.predict_proba(X[va])
        cls = {c: k for k, c in enumerate(clf.classes_)}
        oof[va] = np.array([proba[j, cls[obs[i]]] if obs[i] in cls else 0.0
                            for j, i in enumerate(va)])
    # kNN-distance outlier score (higher = more inlier)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    knn_sim = np.sort(S, axis=1)[:, -KNN:].mean(axis=1)      # mean sim to k nearest = inlierness
    knn_agree = np.array([np.mean(obs[knn_nbr[i]] == obs[i]) for i in range(n)])

    def rank01(v):
        r = np.argsort(np.argsort(v))
        return r / (len(v) - 1 + 1e-9)

    comps = {"oof_agree": oof, "knn_agree": knn_agree, "knn_inlier": knn_sim}
    combined = (rank01(oof) + rank01(knn_agree) + rank01(knn_sim)) / 3.0
    return combined, comps, rank01


def _report(tag, sel, tag_arr, acc, extra=""):
    hi = float(np.mean(tag_arr[sel] == "high"))
    comp = {t: int(np.sum(tag_arr[sel] == t)) for t in np.unique(tag_arr) if t != "high"}
    print(f"  {tag:26} clean%={hi:.3f} metric={acc:.4f} in-sel:{comp} {extra}")


def _mech_matrix(scores: dict, tag_arr):
    """AUROC of each detector against EACH noise mechanism separately (vs clean)."""
    from sklearn.metrics import roc_auc_score
    mechs = [t for t in np.unique(tag_arr) if t != "high"]
    clean_mask = (tag_arr == "high")
    print(f"  {'detector':16} " + " ".join(f"{m:>8}" for m in mechs))
    for name, s in scores.items():
        row = []
        for m in mechs:
            mask = clean_mask | (tag_arr == m)
            y = (tag_arr[mask] == m).astype(int)
            row.append(roc_auc_score(y, -s[mask]))
        print(f"  {name:16} " + " ".join(f"{v:8.3f}" for v in row))


# ---------------- vision (CIFAR-100, cached CLIP embeddings) ----------------
def probe_vision(seed):
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression

    cache = os.path.join(_REPO, f"data/processed/vision_cifar100_clip-vit-base-patch32_p4000v2000t2000_s{seed}.npz")
    z = np.load(cache, allow_pickle=True)
    Xp, Xval, Xt = z["Xp"], z["Xval"], z["Xt"]
    ds = load_dataset("uoft-cs/cifar100", split="train")
    te = load_dataset("uoft-cs/cifar100", split="test")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds))
    tr_idx = perm[:4000]
    te_idx = rng.permutation(len(te))[:2000]
    pool_lab = np.array([ds[int(i)]["fine_label"] for i in tr_idx])
    test_lab = np.array([te[int(i)]["fine_label"] for i in te_idx])

    # noise injection: EXACT copy of run_vision_experiment.inject_noise
    nrng = np.random.default_rng(seed + 7)
    n = len(pool_lab)
    obs = pool_lab.copy()
    tag = np.array(["high"] * n, dtype=object)
    n_low = int(round(NOISE_FRAC * n))
    low_idx = nrng.permutation(n)[:n_low]
    per = max(1, n_low // 3)
    flip, dup, hard = low_idx[:per], low_idx[per:2 * per], low_idx[2 * per:]
    for i in flip:
        obs[i] = nrng.integers(100)
        tag[i] = "flip"
    for i in dup:
        tag[i] = "dup"
    for i in hard:
        tag[i] = "hard"
    if len(dup) > 0:
        seeds_ = dup[: max(1, len(dup) // 8)]
        for j, i in enumerate(dup):
            Xp[i] = Xp[seeds_[j % len(seeds_)]] + 0.01 * np.random.default_rng(i).standard_normal(Xp.shape[1])
        Xp /= (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-8)

    budget = int(0.5 * n)
    is_noise = (tag != "high")

    def train_eval(sel):
        clf = LogisticRegression(max_iter=300, C=1.0).fit(Xp[sel], obs[sel])
        return float((clf.predict(Xt) == test_lab).mean())

    print(f"\n==== VISION CIFAR-100 seed{seed} (pool {n}, budget {budget}, 40% noise) ====")
    # random
    sel_r = list(rng.permutation(n)[:budget])
    _report("random", np.array(sel_r), tag, train_eval(sel_r))
    # oracle: only true-clean (there are 2400 clean, budget 2000 -> all-clean possible)
    clean_idx = np.where(~is_noise)[0]
    sel_o = list(rng.permutation(clean_idx)[:budget]) if len(clean_idx) >= budget else list(clean_idx)
    _report("oracle(true-clean)", np.array(sel_o), tag, train_eval(sel_o))

    # round-1 auth: kNN label agreement (current signal)
    nbr = _chunked_knn(Xp, KNN)
    auth1 = np.array([np.mean(obs[nbr[i]] == obs[i]) for i in range(n)])
    sel1 = list(np.argsort(-auth1)[:budget])
    _report("auth r1 (kNN)", np.array(sel1), tag, train_eval(sel1), f"det-AUROC={_auroc(auth1, is_noise):.3f}")

    # iterated auth: fit probe on provisional top-clean, rescore = probe agreement + kNN
    score = auth1.copy()
    for r in (2, 3):
        top = np.argsort(-score)[: int(0.5 * n)]          # provisional clean set
        probe = LogisticRegression(max_iter=200, C=1.0).fit(Xp[top], obs[top])
        proba = probe.predict_proba(Xp)
        cls = {c: k for k, c in enumerate(probe.classes_)}
        agree = np.array([proba[i, cls[obs[i]]] if obs[i] in cls else 0.0 for i in range(n)])
        score = 0.5 * auth1 + 0.5 * (agree / (agree.max() + 1e-9))
        sel_it = list(np.argsort(-score)[:budget])
        _report(f"auth r{r} (iterated)", np.array(sel_it), tag,
                train_eval(sel_it), f"det-AUROC={_auroc(score, is_noise):.3f}")

    sa, comps, rank01 = strong_auth(Xp, obs, nbr, seed)
    sel_sa = list(np.argsort(-sa)[:budget])
    _report("auth STRONG (rank-mean)", np.array(sel_sa), tag,
            train_eval(sel_sa), f"det-AUROC={_auroc(sa, is_noise):.3f}")
    # mechanism-matched: flips are the harmful ones -> conjunctive min(oof, knn_agree);
    # dups/hard have CORRECT labels, leave them to coverage/margin, do NOT let inlierness in.
    mm = np.minimum(rank01(comps["oof_agree"]), rank01(comps["knn_agree"]))
    sel_mm = list(np.argsort(-mm)[:budget])
    _report("auth MECH (min oof,knn)", np.array(sel_mm), tag,
            train_eval(sel_mm), f"det-AUROC={_auroc(mm, is_noise):.3f}")
    print("  --- detector x mechanism AUROC ---")
    _mech_matrix(comps, tag)
    return None


# ---------------- TEP (local .dat, MLP) ----------------
def probe_tep(seed):
    from sklearn.metrics import f1_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    def _read(p):
        a = np.loadtxt(p)
        return a.T if a.shape[0] == 52 else a

    Xs, ys = [], []
    for f in range(22):
        tr = _read(os.path.join(_REPO, f"data/tep/d{f:02d}.dat"))
        Xs.append(tr); ys.append(np.full(len(tr), f))
    X = np.vstack(Xs); y = np.concatenate(ys)
    Xte_l, yte_l = [], []
    for f in range(22):
        tep = _read(os.path.join(_REPO, f"data/tep/d{f:02d}_te.dat"))
        tep = tep[160:] if f > 0 else tep
        Xte_l.append(tep); yte_l.append(np.full(len(tep), f))
    Xte = np.vstack(Xte_l); yte = np.concatenate(yte_l)

    rng = np.random.default_rng(seed)
    pi = rng.permutation(len(X))[:3000]
    ti = rng.permutation(len(Xte))[:2000]
    sc = StandardScaler().fit(X[pi])
    Xp, yp = sc.transform(X[pi]), y[pi]
    Xt, yt = sc.transform(Xte[ti]), yte[ti]
    n = len(Xp)

    nrng = np.random.default_rng(seed + 7)
    obs = yp.copy()
    tag = np.array(["high"] * n, dtype=object)
    n_low = int(round(NOISE_FRAC * n))
    low = nrng.permutation(n)[:n_low]
    per = max(1, n_low // 3)
    flip, corr, dup = low[:per], low[per:2 * per], low[2 * per:]
    for i in flip:
        obs[i] = nrng.integers(22); tag[i] = "flip"
    for i in corr:
        Xp[i] += nrng.normal(0, 3.0, Xp.shape[1]); tag[i] = "corrupt"
    for i in dup:
        tag[i] = "dup"
    is_noise = (tag != "high")
    budget = int(0.3 * n)

    def train_eval(sel):
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=120, random_state=seed).fit(Xp[sel], obs[sel])
        return float(f1_score(yt, clf.predict(Xt), average="macro"))

    print(f"\n==== TEP seed{seed} (pool {n}, budget {budget}, 40% noise, MLP macro-F1) ====")
    sel_r = list(rng.permutation(n)[:budget])
    _report("random", np.array(sel_r), tag, train_eval(sel_r))
    clean_idx = np.where(~is_noise)[0]
    sel_o = list(rng.permutation(clean_idx)[:budget])
    _report("oracle(true-clean)", np.array(sel_o), tag, train_eval(sel_o))

    Xn = Xp / (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-8)
    nbr = _chunked_knn(Xn, KNN)
    auth1 = np.array([np.mean(obs[nbr[i]] == obs[i]) for i in range(n)])
    sel1 = list(np.argsort(-auth1)[:budget])
    _report("auth r1 (kNN)", np.array(sel1), tag, train_eval(sel1), f"det-AUROC={_auroc(auth1, is_noise):.3f}")

    from sklearn.linear_model import LogisticRegression
    score = auth1.copy()
    for r in (2, 3):
        top = np.argsort(-score)[: int(0.5 * n)]
        probe = LogisticRegression(max_iter=200).fit(Xp[top], obs[top])
        proba = probe.predict_proba(Xp)
        cls = {c: k for k, c in enumerate(probe.classes_)}
        agree = np.array([proba[i, cls[obs[i]]] if obs[i] in cls else 0.0 for i in range(n)])
        score = 0.5 * auth1 + 0.5 * (agree / (agree.max() + 1e-9))
        sel_it = list(np.argsort(-score)[:budget])
        _report(f"auth r{r} (iterated)", np.array(sel_it), tag,
                train_eval(sel_it), f"det-AUROC={_auroc(score, is_noise):.3f}")

    sa, comps, rank01 = strong_auth(Xn, obs, nbr, seed)
    sel_sa = list(np.argsort(-sa)[:budget])
    _report("auth STRONG (rank-mean)", np.array(sel_sa), tag,
            train_eval(sel_sa), f"det-AUROC={_auroc(sa, is_noise):.3f}")
    mm = np.minimum(rank01(comps["oof_agree"]), rank01(comps["knn_agree"]))
    sel_mm = list(np.argsort(-mm)[:budget])
    _report("auth MECH (min oof,knn)", np.array(sel_mm), tag,
            train_eval(sel_mm), f"det-AUROC={_auroc(mm, is_noise):.3f}")
    # TEP has real feature corruption -> conjunctive WITH the outlier detector too
    mm3 = np.minimum(mm, rank01(comps["knn_inlier"]))
    sel_mm3 = list(np.argsort(-mm3)[:budget])
    _report("auth MECH3 (+inlier)", np.array(sel_mm3), tag,
            train_eval(sel_mm3), f"det-AUROC={_auroc(mm3, is_noise):.3f}")
    print("  --- detector x mechanism AUROC ---")
    _mech_matrix(comps, tag)


if __name__ == "__main__":
    probe_vision(SEED)
    probe_tep(SEED)
