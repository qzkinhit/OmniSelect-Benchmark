"""Controlled local implementations of recognized data-selection baselines.

These are the community-standard baselines a reviewer expects to see, kept as pure functions
(features / per-sample signals in, selected indices out) so each modality runner can call
them and ALSO fold them into the AdaptiveController's portfolio (extra_strategies) -- the
controller is then >= these recognized baselines by construction too, not only >= our own
symmetric arms.

Each function documents whether it preserves the published core or is a reduced proxy.
All functions receive the same features, budget, and clean reference in the unified
testbed. Exact original-system reproduction is tracked separately in the fidelity ledger.

Implemented
-----------
geometric coresets (DeepCore family), operate on an embedding matrix:
  - herding            (Welling, 2009; DeepCore): greedily match the set mean in feature space
  - kcenter_greedy     (Sener & Savarese, ICLR 2018): farthest-point / k-center coreset
score-based pruning (DeepCore family), operate on downstream per-sample signals:
  - el2n               (Paul et al., NeurIPS 2021): ||softmax(logits) - onehot(y)||_2, keep top-k
  - grand              (Paul et al., NeurIPS 2021): last-layer gradient-norm proxy = EL2N * ||phi||
distribution matching (text):
  - dsir               (Xie et al., NeurIPS 2023): hashed n-gram importance resampling toward a
                       clean target distribution (Gumbel-top-k on importance log-weights)
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = ["herding", "kcenter_greedy", "el2n", "grand", "grand_expected", "ccs", "dmf_dynamic",
           "dmf_published_update", "quadmix", "quadmix_expected_counts",
           "quadmix_published_core", "dsir_select", "semdedup", "density_select"]


def _as2d(features: np.ndarray) -> np.ndarray:
    f = np.asarray(features, dtype=np.float64)
    return f.reshape(f.shape[0], -1)


def herding(features: np.ndarray, k: int) -> list:
    """Herding coreset (Welling 2009). Greedily pick the point that pulls the running
    selected-mean closest to the full-set mean in feature space. Standard DeepCore baseline."""
    X = _as2d(features)
    n = X.shape[0]
    k = min(k, n)
    target = X.mean(axis=0)
    chosen: list = []
    running = np.zeros_like(target)
    mask = np.ones(n, dtype=bool)
    for t in range(k):
        # next selected mean would be (running + X[i]) / (t+1); minimise its distance to target
        cand = (running[None, :] + X) / (t + 1)
        d = np.linalg.norm(cand - target[None, :], axis=1)
        d[~mask] = np.inf
        i = int(np.argmin(d))
        chosen.append(i)
        running = running + X[i]
        mask[i] = False
    return chosen


def kcenter_greedy(features: np.ndarray, k: int, seed: int = 0) -> list:
    """k-center greedy / coreset (Sener & Savarese 2018). Farthest-point traversal: each step
    add the point farthest from the current selected set. Standard DeepCore baseline."""
    X = _as2d(features)
    n = X.shape[0]
    k = min(k, n)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(n))
    chosen = [start]
    dist = np.linalg.norm(X - X[start][None, :], axis=1)
    for _ in range(1, k):
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(X - X[i][None, :], axis=1))
    return chosen


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def el2n(probs_or_logits: np.ndarray, labels: np.ndarray, k: int, *, is_logits: bool = True) -> list:
    """EL2N (Paul et al. 2021): error L2 norm ||p - onehot(y)||_2 from an early/probe model.
    Keep the top-k hardest (largest error). Standard DeepCore score-based baseline."""
    P = _softmax(np.asarray(probs_or_logits, float)) if is_logits else np.asarray(probs_or_logits, float)
    y = np.asarray(labels).astype(int)
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    score = np.linalg.norm(P - onehot, axis=1)
    return [int(i) for i in np.argsort(-score)[:min(k, len(score))]]


def grand(probs_or_logits: np.ndarray, labels: np.ndarray, features: np.ndarray, k: int,
          *, is_logits: bool = True) -> list:
    """GraNd (Paul et al. 2021): expected gradient norm. For a linear head the last-layer
    gradient is (p - onehot(y)) outer phi, whose norm is ||p - onehot(y)|| * ||phi||. We use
    that exact last-layer proxy (EL2N weighted by feature norm). Keep top-k."""
    P = _softmax(np.asarray(probs_or_logits, float)) if is_logits else np.asarray(probs_or_logits, float)
    y = np.asarray(labels).astype(int)
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    err = np.linalg.norm(P - onehot, axis=1)
    phi = np.linalg.norm(_as2d(features), axis=1)
    score = err * phi
    return [int(i) for i in np.argsort(-score)[:min(k, len(score))]]


def grand_expected(features: np.ndarray, labels: np.ndarray, k: int, *,
                   K: int = 5, steps: int = 8, lr: float = 0.5, seed: int = 0) -> list:
    """GraNd faithful (Paul et al. 2021): the EXPECTED gradient norm over several early-training
    checkpoints. We fit K linear heads from random init with a few full-batch GD steps (early
    training, non-saturated softmax), and for each take the exact last-layer gradient norm
    ||p - onehot(y)|| * sqrt(||phi||^2 + 1) (weights + bias), then average over the K models.
    This is genuinely distinct from EL2N: EL2N is the error norm of ONE converged probe (which on
    a noise-injected pool is bimodal so ties collapse), while GraNd averages non-saturated gradient
    norms over the training distribution, so its ranking differs. Keep top-k (largest = hardest)."""
    X = _as2d(np.asarray(features, float))
    n, d = X.shape
    y = np.asarray(labels).astype(int)
    C = int(y.max()) + 1
    onehot = np.zeros((n, C)); onehot[np.arange(n), y] = 1.0
    phinorm = np.sqrt((X ** 2).sum(1) + 1.0)          # last-layer param grad includes the bias term
    score = np.zeros(n)
    for r in range(K):
        rng = np.random.default_rng(seed * 1000 + r)
        W = rng.standard_normal((d, C)) * 0.01
        b = np.zeros(C)
        for _ in range(steps):                         # a few GD steps -> early-training checkpoint
            P = _softmax(X @ W + b)
            G = P - onehot
            W -= lr * (X.T @ G) / n
            b -= lr * G.mean(0)
        P = _softmax(X @ W + b)
        score += np.linalg.norm(P - onehot, axis=1) * phinorm
    score /= K
    return [int(i) for i in np.argsort(-score)[:min(k, n)]]


def ccs(probs_or_logits: np.ndarray, labels: np.ndarray, k: int, *, is_logits: bool = False,
        bins: int = 50, cutoff: float = 0.1) -> list:
    """CCS - Coverage-centric Coreset Selection (Zheng et al., ICLR 2023). Prune a fraction of the
    hardest (highest-EL2N) examples first (they are often mislabeled/outliers), then sample the
    remaining budget uniformly across difficulty strata so the selection covers the full
    difficulty spectrum rather than only easy or only hard points."""
    P = _softmax(np.asarray(probs_or_logits, float)) if is_logits else np.asarray(probs_or_logits, float)
    y = np.asarray(labels).astype(int)
    onehot = np.zeros_like(P); onehot[np.arange(len(y)), y] = 1.0
    diff = np.linalg.norm(P - onehot, axis=1)          # EL2N difficulty
    n = len(diff); k = min(k, n)
    order = np.argsort(diff)                            # easy -> hard
    keep = order[: int(n * (1.0 - cutoff))]            # drop the hardest cutoff fraction
    d = diff[keep]
    edges = np.quantile(d, np.linspace(0, 1, bins + 1))
    strata = [keep[(d >= edges[b]) & (d <= edges[b + 1])] for b in range(bins)]
    rng = np.random.default_rng(0)
    out, per = [], max(1, k // bins)
    for s in strata:
        if len(s):
            out.extend(int(i) for i in rng.permutation(s)[:per])
    if len(out) < k:                                   # top up from the kept pool
        rest = [int(i) for i in keep if int(i) not in set(out)]
        out.extend(rest[: k - len(out)])
    return out[:k]


def dmf_dynamic(channel_scores: np.ndarray, k: int, val_reward, *, rounds: int = 6, eta: float = 0.5,
                seed: int = 0) -> list:
    """Dynamic-fusion proxy used by the frozen historical batches.

    The published Multi-Actor system uses actor-memory EMA and an additive collaboration
    update around the mean actor reward. This reduced cross-modal proxy instead applies
    multiplicative channel weights to local validation rewards. It must not be described
    as an original-system reproduction. A published-update implementation is tracked as a
    separate fidelity task.
    """
    S = np.asarray(channel_scores, float)              # (m, n)
    m, n = S.shape
    k = min(k, n)
    w = np.ones(m) / m
    best_sel, best_r = None, -np.inf
    for _ in range(rounds):
        fused = w @ S
        sel = list(np.argsort(-fused)[:k])
        r = float(val_reward(sel))
        if r > best_r:
            best_r, best_sel = r, sel
        # per-channel reward = validation gain of that channel alone -> reweight (multiplicative)
        rew = np.array([float(val_reward(list(np.argsort(-S[j])[:k]))) for j in range(m)])
        rew = (rew - rew.min()) / (rew.max() - rew.min() + 1e-9)
        w = w * np.exp(eta * rew)
        w /= w.sum()
    return best_sel if best_sel is not None else list(np.argsort(-(w @ S))[:k])


def dmf_published_update(channel_scores: np.ndarray, k: int, val_reward, *, rounds: int = 6,
                         eta: float = 0.5, seed: int = 0, return_trace: bool = False):
    """Published-update transfer of the Multi-Actor collaboration rule.

    Equation 6 forms a weighted actor score. Equations 7 and 8 update every
    collaboration weight additively by ``eta * (actor_reward - mean_reward)``.
    The original actor memories and influence-function rewards are replaced by
    the caller's cross-modal channel scores and validation callback. A projection
    to the probability simplex is applied after Eq. 8 so the transferred weights
    remain valid mixture weights. This is an equation-level transfer, not a
    reproduction of the original text-pretraining system.
    """
    del seed
    S = np.asarray(channel_scores, dtype=float)
    if S.ndim != 2:
        raise ValueError("channel_scores must have shape (actors, samples)")
    m, n = S.shape
    k = min(int(k), n)
    theta = np.ones(m, dtype=float) / m
    best_sel, best_reward = None, -np.inf
    trace = []
    for step in range(int(rounds)):
        fused = theta @ S
        sel = [int(i) for i in np.argsort(-fused)[:k]]
        fused_reward = float(val_reward(sel))
        if fused_reward > best_reward:
            best_reward, best_sel = fused_reward, sel
        actor_rewards = np.array([
            float(val_reward([int(i) for i in np.argsort(-S[j])[:k]]))
            for j in range(m)
        ])
        mean_reward = float(actor_rewards.mean())
        raw_next = theta + float(eta) * (actor_rewards - mean_reward)
        projected = np.maximum(raw_next, 0.0)
        if projected.sum() <= 1e-12:
            projected = np.ones(m, dtype=float)
        projected /= projected.sum()
        trace.append({"round": step, "theta": theta.copy(),
                      "actor_rewards": actor_rewards.copy(),
                      "mean_reward": mean_reward, "raw_next": raw_next.copy(),
                      "projected_next": projected.copy(),
                      "fused_reward": fused_reward})
        theta = projected
    result = best_sel if best_sel is not None else [int(i) for i in np.argsort(-(theta @ S))[:k]]
    return (result, trace) if return_trace else result


def quadmix_expected_counts(quality: np.ndarray, domains: np.ndarray, *,
                            alpha: Optional[np.ndarray] = None,
                            lambdas: float | np.ndarray = 100.0,
                            omegas: float | np.ndarray = 0.05,
                            etas: float | np.ndarray = 1.0,
                            epsilons: float | np.ndarray = 0.001,
                            token_weights: Optional[np.ndarray] = None,
                            quality_higher_is_better: bool = True) -> np.ndarray:
    """QuaDMix equations 1 to 3, returning expected sampling counts.

    Quality criteria are min-max normalized. They are flipped when the caller's
    convention is higher-is-better because the paper defines smaller scores as
    higher quality. Equation 1 merges criteria with domain-specific weights.
    Equation 2 computes the token-weighted within-domain percentile rank.
    Equation 3 applies the domain-specific sigmoid expected-repeat sampler.
    """
    Q = np.asarray(quality, dtype=float)
    if Q.ndim == 1:
        Q = Q[:, None]
    d = np.asarray(domains)
    if len(Q) != len(d):
        raise ValueError("quality and domains length mismatch")
    n, n_criteria = Q.shape
    tw = np.ones(n, dtype=float) if token_weights is None else np.asarray(token_weights, dtype=float)
    if len(tw) != n or np.any(tw <= 0):
        raise ValueError("token_weights must be positive and match the pool")
    unique = np.unique(d)
    m = len(unique)

    def per_domain(value, name):
        a = np.asarray(value, dtype=float)
        if a.ndim == 0:
            return np.repeat(float(a), m)
        if a.shape != (m,):
            raise ValueError(f"{name} must be scalar or have one value per domain")
        return a

    lam = per_domain(lambdas, "lambdas")
    omg = per_domain(omegas, "omegas")
    eta = per_domain(etas, "etas")
    eps = per_domain(epsilons, "epsilons")
    if alpha is None:
        A = np.ones((m, n_criteria), dtype=float) / n_criteria
    else:
        A = np.asarray(alpha, dtype=float)
        if A.ndim == 1:
            A = np.repeat(A[None, :], m, axis=0)
        if A.shape != (m, n_criteria):
            raise ValueError("alpha must have shape (domains, criteria)")
        A = A / (A.sum(axis=1, keepdims=True) + 1e-12)

    qnorm = np.empty_like(Q)
    for j in range(n_criteria):
        lo, hi = float(Q[:, j].min()), float(Q[:, j].max())
        qnorm[:, j] = (Q[:, j] - lo) / (hi - lo + 1e-12)
    if quality_higher_is_better:
        qnorm = 1.0 - qnorm

    counts = np.zeros(n, dtype=float)
    for mi, label in enumerate(unique):
        idx = np.where(d == label)[0]
        merged = qnorm[idx] @ A[mi]
        order = np.argsort(merged, kind="mergesort")
        ordered_idx = idx[order]
        cumulative = np.cumsum(tw[ordered_idx])
        rank = cumulative / cumulative[-1]
        z = np.clip(-lam[mi] * (omg[mi] - rank), -60.0, 60.0)
        expected = np.where(rank <= omg[mi], eta[mi] / (1.0 + np.exp(z)) + eps[mi], eps[mi])
        counts[ordered_idx] = expected
    return counts


def quadmix_published_core(quality: np.ndarray, features: np.ndarray, k: int, *,
                           domains: Optional[np.ndarray] = None,
                           token_weights: Optional[np.ndarray] = None,
                           seed: int = 0, n_domains: int = 8,
                           lambdas: float | np.ndarray = 100.0,
                           omegas: float | np.ndarray = 0.05,
                           etas: float | np.ndarray = 1.0,
                           epsilons: float | np.ndarray = 0.001) -> list:
    """Fixed-budget adaptation of the published QuaDMix sampling equations.

    If native domain labels are unavailable, k-means labels over the shared
    representation substitute for the paper's DeBERTa domain classifier. The
    expected-repeat values from Eq. 3 are converted to a no-replacement fixed-size
    subset with Gumbel top-k. The original LightGBM search over 3,000 proxy models
    is replaced by the explicit frozen parameters supplied here.
    """
    X = _as2d(features)
    n = len(X)
    k = min(int(k), n)
    if domains is None:
        from sklearn.cluster import KMeans
        nc = min(max(1, int(n_domains)), n)
        domains = KMeans(n_clusters=nc, n_init=3, random_state=seed).fit_predict(X)
    counts = quadmix_expected_counts(
        quality, np.asarray(domains), lambdas=lambdas, omegas=omegas,
        etas=etas, epsilons=epsilons, token_weights=token_weights,
        quality_higher_is_better=True,
    )
    rng = np.random.default_rng(seed)
    keys = np.log(counts + 1e-30) + rng.gumbel(size=n)
    return [int(i) for i in np.argsort(-keys)[:k]]


def quadmix(quality: np.ndarray, features: np.ndarray, k: int, *, alpha: float = 0.5,
            bins: int = 20, seed: int = 0) -> list:
    """QuaDMix-style proxy: jointly optimize quality and diversity under a fixed record budget.
    that trades quality density against coverage. We bucket by quality, allocate the budget across
    buckets by a quality-tilted weight (alpha), and within each bucket pick the most diverse
    (farthest-first) points, so high-quality and well-spread samples are kept together.
    This is not the published domain-specific rank and sigmoid expected-repeat sampler,
    nor its 3,000-proxy-model LightGBM parameter search.
    """
    q = np.asarray(quality, float)
    X = _as2d(np.asarray(features, float))
    n = len(q); k = min(k, n)
    if k <= 0:
        return []
    order = np.argsort(q, kind="mergesort")             # low -> high quality
    edges = np.quantile(q, np.linspace(0, 1, bins + 1))
    # Assign every record to exactly one half-open quantile bucket.  The old
    # pair of inclusive comparisons admitted boundary ties to adjacent buckets,
    # which silently turned a fixed-record subset into repeated observations.
    bucket = np.searchsorted(edges[1:-1], q, side="right")
    weights = np.array([(0.5 * (edges[b] + edges[b + 1])) for b in range(bins)])
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
    weights = (1 - alpha) + alpha * weights            # quality-tilted budget allocation
    weights /= weights.sum()
    out = []
    for b in range(bins):
        mem = order[bucket[order] == b]
        kb = int(round(k * weights[b]))
        if len(mem) and kb:
            # farthest-first within the bucket for diversity
            sub = X[mem]
            rng = np.random.default_rng(seed + b)
            picked = [int(rng.integers(len(mem)))]
            available = np.ones(len(mem), dtype=bool)
            available[picked[0]] = False
            dist = np.linalg.norm(sub - sub[picked[0]][None], axis=1)
            while len(picked) < min(kb, len(mem)):
                # Mask already selected local indices.  This also prevents
                # duplicates when all feature distances are exactly zero.
                i = int(np.argmax(np.where(available, dist, -np.inf)))
                picked.append(i)
                available[i] = False
                dist = np.minimum(dist, np.linalg.norm(sub - sub[i][None], axis=1))
            out.extend(int(mem[p]) for p in picked)
    if len(out) < k:
        selected = set(out)
        rest = [int(i) for i in order[::-1] if int(i) not in selected]
        out.extend(rest[: k - len(out)])
    out = out[:k]
    if len(out) != k or len(set(out)) != k:
        raise RuntimeError("quadmix must return exactly k unique records")
    return out


def semdedup(features: np.ndarray, k: int, *, seed: int = 0, n_clusters: Optional[int] = None,
             eps: float = 0.05) -> list:
    """SemDeDup (Abbas et al., 2023), faithful rule: k-means cluster the (unit-normalized)
    embeddings, and WITHIN each cluster mark pairs with cosine similarity above 1 - eps as
    semantic duplicates; each duplicate group keeps exactly one representative (the paper keeps
    the example with the LOWEST similarity to the cluster centroid). Survivors are returned
    first; if the budget exceeds the number of survivors, fill from the removed duplicates."""
    from sklearn.cluster import KMeans
    X = _as2d(features)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    n = X.shape[0]; k = min(k, n)
    K = n_clusters or max(2, min(64, n // 20))
    km = KMeans(n_clusters=K, n_init=3, random_state=seed).fit(X)
    thr = 1.0 - eps
    keep, dropped = [], []
    for c in range(K):
        mem = np.where(km.labels_ == c)[0]
        if not len(mem):
            continue
        cen = km.cluster_centers_[c]
        cen = cen / (np.linalg.norm(cen) + 1e-12)
        # iterate low->high centroid similarity: the paper's kept representative is the
        # low-centroid-similarity member of each duplicate group
        order = mem[np.argsort(X[mem] @ cen)]
        kept_c = []
        for i in order:
            sims = X[kept_c] @ X[i] if kept_c else np.array([])
            if len(sims) and sims.max() > thr:
                dropped.append(int(i))         # duplicate of an already-kept example
            else:
                kept_c.append(int(i))
        keep.extend(kept_c)
    rng = np.random.default_rng(seed)
    keep = list(rng.permutation(np.array(keep, dtype=int)))
    if len(keep) < k:
        keep.extend(dropped[: k - len(keep)])
    return [int(i) for i in keep[:k]]


def density_select(features: np.ndarray, k: int, *, knn: int = 10, seed: int = 0) -> list:
    """Density sampler (Sachdeva et al., "How to Train Data-Efficient LLMs", 2024), faithful
    rule: estimate local density with a kernel-style kNN estimate and SAMPLE examples with
    probability inversely proportional to density (inverse-propensity diversified sampling),
    so dense regions are thinned and the selection covers the space. Gumbel top-k makes the
    inverse-density sampling reproducible without replacement."""
    X = _as2d(features)
    n = X.shape[0]; k = min(k, n)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(knn + 1, n)).fit(X)
    dist, _ = nn.kneighbors(X)
    sparsity = dist[:, 1:].mean(axis=1)                 # larger = lower density
    logits = np.log(sparsity + 1e-12)                   # sampling prob proportional to 1/density
    rng = np.random.default_rng(seed)
    g = rng.gumbel(size=n)
    return [int(i) for i in np.argsort(-(logits + g))[:k]]


def dsir_select(counts: np.ndarray, target_counts: np.ndarray, k: int, *,
                seed: int = 0, smoothing: float = 1.0) -> list:
    """DSIR (Xie et al. 2023): importance resampling on hashed n-gram features toward a clean
    target distribution. We fit a bag-of-features unigram model for the raw pool and for the
    clean target, score each candidate by its log importance weight log p_target / p_raw, and
    take the budget via Gumbel-top-k (resampling without replacement, faithful to DSIR).

    counts        : (n, V) candidate hashed n-gram count matrix (raw pool).
    target_counts : (V,) or (m, V) clean target feature counts (the same clean reference all
                    methods see -- not the answer key).
    """
    C = np.asarray(counts, float)
    n, V = C.shape
    k = min(k, n)
    raw_total = C.sum(axis=0) + smoothing
    p_raw = raw_total / raw_total.sum()
    tc = np.asarray(target_counts, float)
    tc = tc.sum(axis=0) if tc.ndim == 2 else tc
    t_total = tc + smoothing
    p_tgt = t_total / t_total.sum()
    logw_feat = np.log(p_tgt) - np.log(p_raw)            # per-feature log importance
    # per-document log importance = sum over its features (bag-of-words log-likelihood ratio)
    row_len = C.sum(axis=1) + 1e-9
    logw = (C @ logw_feat) / row_len                     # length-normalised, as in DSIR practice
    rng = np.random.default_rng(seed)
    gumbel = -np.log(-np.log(rng.uniform(size=n) + 1e-12) + 1e-12)
    keys = logw + gumbel                                 # Gumbel-top-k == sampling w/o replacement
    return [int(i) for i in np.argsort(-keys)[:k]]


# ---------------------------------------------------------------------------
# 2024-2025 closest competitors (server round): D4, DsDm, RegMix, PerpCorr.
# Same discipline: pure functions, faithful to the original algorithm at the
# scale we can run, scope reductions stated explicitly in the paper.
# ---------------------------------------------------------------------------

def d4(features: np.ndarray, k: int, *, seed: int = 0, dedup_frac: float = 0.25,
       n_clusters: Optional[int] = None) -> list:
    """D4 (Tirumala et al., NeurIPS 2023): SemDeDup then SSL-prototype diversification.

    Stage 1 removes the most semantically duplicated fraction (same within-cluster
    cosine criterion as semdedup). Stage 2 clusters the survivors and drops the most
    PROTOTYPICAL points (closest to their centroid), keeping the diverse ones, i.e.
    rank by distance to own centroid descending and take the top-k.
    """
    from sklearn.cluster import KMeans
    f = _as2d(features)
    n = len(f)
    keep = semdedup(f, max(k, int(round((1.0 - dedup_frac) * n))), seed=seed)
    keep = np.asarray(keep)
    if len(keep) <= k:
        return [int(i) for i in keep[:k]]
    g = f[keep]
    nc = n_clusters or max(2, int(np.sqrt(len(g))))
    km = KMeans(n_clusters=nc, n_init=3, random_state=seed).fit(g)
    d_cent = np.linalg.norm(g - km.cluster_centers_[km.labels_], axis=1)
    order = np.argsort(-d_cent)                     # farthest-from-prototype first
    return [int(keep[i]) for i in order[:k]]


def dsdm_scores(train_eval_fn, n: int, *, k_runs: int = 24, subset_frac: float = 0.5,
                seed: int = 0, ridge: float = 1.0) -> np.ndarray:
    """DsDm (Engstrom et al., 2024): datamodel-based selection, proxy-scale reduction.

    Linear datamodel fitted by subset regression: draw k_runs random subsets, obtain the
    downstream metric of a cheap probe trained on each subset via train_eval_fn(indices),
    then ridge-regress the metric on centered inclusion indicators. The coefficient of a
    sample is its estimated marginal contribution; selection takes the top-k. The probe
    metric MUST come from the construction split (V1) so the adjudication split stays
    untouched (Theorem 4 discipline).
    """
    rng = np.random.default_rng(seed)
    m = int(round(subset_frac * n))
    Xind = np.zeros((k_runs, n))
    y = np.zeros(k_runs)
    for r in range(k_runs):
        idx = rng.permutation(n)[:m]
        Xind[r, idx] = 1.0
        y[r] = float(train_eval_fn([int(i) for i in idx]))
    Xc = Xind - Xind.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    A = Xc.T @ Xc + ridge * np.eye(n)
    w = np.linalg.solve(A, Xc.T @ yc)
    return w


def regmix_mixture(domain_ids: Sequence, train_eval_fn, *, n_probe: int = 12,
                   seed: int = 0) -> dict:
    """RegMix (Liu et al., ICLR 2025): mixture-proportion regression, tiny-proxy reduction.

    Draw n_probe Dirichlet mixtures over domains, evaluate each mixture with a cheap
    probe via train_eval_fn(domain_weights dict) -> metric (V1 only), fit linear
    regression metric ~ weights, and return the predicted-best mixture on the simplex
    (argmax over a dense random candidate set under the fitted model).
    """
    doms = sorted(set(domain_ids))
    D = len(doms)
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(D), size=n_probe)
    y = np.array([float(train_eval_fn({d: float(w[j]) for j, d in enumerate(doms)})) for w in W])
    Xc = W - W.mean(axis=0, keepdims=True)
    coef, *_ = np.linalg.lstsq(Xc, y - y.mean(), rcond=None)
    cand = rng.dirichlet(np.ones(D), size=4096)
    best = cand[np.argmax(cand @ coef)]
    return {d: float(best[j]) for j, d in enumerate(doms)}


def perpcorr_select(ppl: np.ndarray, domain_ids: Sequence, domain_gain: dict, k: int) -> list:
    """Perplexity-Correlations (Thrush et al., ICLR 2025), single-model reduction.

    The original estimates, across many public models, the correlation between domain
    perplexity and benchmark performance, then selects data from positively correlated
    domains. With one reference model we reduce faithfully: domain weights are the
    measured construction-split gains (higher-gain domains get more budget,
    softmax-allocated), and within a domain the lowest-perplexity samples are taken
    first. The reduction is stated in the paper.
    """
    doms = sorted(set(domain_ids))
    g = np.array([float(domain_gain.get(d, 0.0)) for d in doms])
    gz = (g - g.mean()) / (g.std() + 1e-9)
    alloc = np.exp(gz) / np.exp(gz).sum()
    dom_arr = np.asarray(list(domain_ids))
    out = []
    for j, d in enumerate(doms):
        idx = np.where(dom_arr == d)[0]
        take = min(len(idx), int(round(alloc[j] * k)))
        order = idx[np.argsort(ppl[idx])]           # low reference PPL first
        out.extend(int(i) for i in order[:take])
    rest = [i for i in np.argsort(ppl) if i not in set(out)]
    out.extend(int(i) for i in rest[: max(0, k - len(out))])
    return out[:k]
