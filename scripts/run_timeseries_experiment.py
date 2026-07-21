"""Time-series arm: data selection for a forecasting model, the SAME unified framework
on time-series signals. Selection = choosing which training WINDOWS to keep under a
fixed budget; the model is a small DLinear trained from scratch (faithful to the TSFM
data-curation line: Chronos KernelSynth/TSMixup, Moirai-LOTSA filtering).

Pipeline mirrors vision/tabular:
  1. windows from ETTh1 (OT channel, standard LSF benchmark); 60% clean + 40% controlled
     noise (corrupt / flat / shuffle / near-duplicate), each tagged;
  2. three channels on each input window (same fusion controller):
       authenticity = temporal-structure score (lag-1 autocorrelation; flat/shuffled/noisy
         windows lose it), influence = -error of a clean-reference DLinear (forecastable
         on-distribution windows score high), redundancy = window-shape novelty;
  3. each method picks a budgeted window set; train DLinear; report MASE on clean held-out
     test windows (lower is better) -> we report NEG-MASE so "higher = better" like others;
  4. multiple seeds. M2-feasible (DLinear is two linear layers). Env: TS_DATASET, POOL_N,
     TEST_N, L, H, NOISE_FRAC, BUDGET_FRAC, SEED, METHODS.
"""
from __future__ import annotations

import os
import sys
import time
import hashlib

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

POOL_N = int(os.environ.get("POOL_N", "3000"))
TEST_N = int(os.environ.get("TEST_N", "1500"))
VAL_N = int(os.environ.get("VAL_N", "1000"))
L = int(os.environ.get("L", "96"))        # input length
H = int(os.environ.get("H", "24"))        # forecast horizon
NOISE_FRAC = float(os.environ.get("NOISE_FRAC", "0.40"))
BUDGET_FRAC = float(os.environ.get("BUDGET_FRAC", "0.3"))
KNN = int(os.environ.get("KNN", "15"))
SEED = int(os.environ.get("SEED", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "60"))
METHODS = os.environ.get("METHODS", "full,random,coreset,auth_only,influence_only,mmdataselect,mmds_adapt").split(",")
LAM = float(os.environ.get("LAM", "0.5"))
AUTH_Q = float(os.environ.get("AUTH_Q", "0.25"))
W_INFL = float(os.environ.get("W_INFL", "0.5"))
PAIRED_RNG = os.environ.get("PAIRED_RNG", "0") == "1"


TS_DATASET = os.environ.get("TS_DATASET", "ETTh1")   # ETTh1 | ETTm1 | ETTh2 | ETTm2 (standard LSF benchmarks)
TS_MODEL = os.environ.get("TS_MODEL", "dlinear")     # dlinear (from scratch) | chronos (fine-tuned TS foundation model)


def load_series():
    import pandas as pd
    if TS_DATASET.startswith("daisy_"):
        # DaISy (KU Leuven SISTA identification database), process-industry systems.
        # daisy_cstr = [98-002] continuous stirred tank reactor: cols time, coolant flow q,
        # concentration Ca, temperature T (0.1 min sampling, 7500 steps). We forecast the
        # concentration channel. daisy_steamgen = [98-003] Abbott steam generator (col 5 =
        # drum pressure). Files: data/daisy/<name>.dat (from ftp.esat.kuleuven.be/pub/SISTA/data).
        name = TS_DATASET.split("_", 1)[1]
        path = os.path.join(_REPO, f"data/daisy/{name}.dat")
        if not os.path.exists(path) and os.path.exists(path + ".gz"):
            import gzip, shutil
            with gzip.open(path + ".gz", "rb") as f_in, open(path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        col = {"cstr": 2, "steamgen": 5}.get(name, 2)
        arr = np.loadtxt(path)
        return arr[:, col].astype(float)
    url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{TS_DATASET}.csv"
    cache = os.path.join(_REPO, f"data/processed/{TS_DATASET.lower()}.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
    else:
        df = pd.read_csv(url)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.to_csv(cache, index=False)
    return df["OT"].to_numpy(dtype=float)


def _export_split_ids(arm, seed, payload):
    """Env-gated split-id manifest dump (POST_DATA_LOCK audit item 四.3): when
    SPLIT_EXPORT_DIR is set, persist the exact seeded split indices + rng recipe
    used by this run, then continue normally. No effect when the env var is unset.
    For this arm the seeded split lives in main() (load_series returns the raw
    series only), so the hook is called right after the window starts are drawn."""
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


def make_windows(series, idxs):
    X = np.stack([series[i : i + L] for i in idxs])
    Y = np.stack([series[i + L : i + L + H] for i in idxs])
    return X, Y


def inject_noise(X, Y, seed):
    rng = np.random.default_rng(seed + 7)
    n = len(X)
    Xn, Yn = X.copy(), Y.copy()
    tag = np.array(["high"] * n, dtype=object)
    n_low = int(round(NOISE_FRAC * n))
    low = rng.permutation(n)[:n_low]
    per = max(1, n_low // 4)
    corrupt, flat, shuf, dup = low[:per], low[per : 2 * per], low[2 * per : 3 * per], low[3 * per :]
    sd = X.std() + 1e-8
    for i in corrupt:                       # heavy noise -> authenticity/influence
        Xn[i] = Xn[i] + rng.standard_normal(L) * sd * 1.5
        tag[i] = "corrupt"
    for i in flat:                          # flat/degenerate -> authenticity (no structure)
        Xn[i] = np.full(L, Xn[i].mean())
        tag[i] = "flat"
    for i in shuf:                          # shuffled -> destroys temporal structure
        Xn[i] = rng.permutation(Xn[i])
        tag[i] = "shuffle"
    if len(dup):                            # near-duplicate -> redundancy
        seeds = dup[: max(1, len(dup) // 8)]
        for j, i in enumerate(dup):
            Xn[i] = Xn[seeds[j % len(seeds)]] + 0.01 * rng.standard_normal(L) * sd
            Yn[i] = Yn[seeds[j % len(seeds)]]
            tag[i] = "dup"
    return Xn, Yn, tag


def _autocorr1(x):
    x = x - x.mean()
    d = (x[:-1] ** 2).sum()
    return float((x[:-1] * x[1:]).sum() / (d + 1e-9)) if d > 0 else 0.0


_CHRONOS_CACHE = {}


class _ChronosWrap:
    """Adapter: fine-tuned chronos-bolt behaves like the DLinear callable in mase()."""

    def __init__(self, model, H):
        self.model, self.H = model, H

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, x):                            # x: (B, L) float tensor
        import torch
        with torch.no_grad():
            out = self.model(context=x)
        q = out.quantile_preds                        # (B, Q, H)
        mid = q.shape[1] // 2
        return q[:, mid, : self.H]                    # median forecast


def train_chronos(Xtr, Ytr, dev, seed, epochs=None):
    """Few-shot fine-tune of the chronos-bolt-tiny time-series foundation model on the
    selected windows. Same signature and return interface as train_dlinear, so every
    call site (reference model, V1 construction, V2 adjudication, final fit) and the
    shared mase() evaluation stay bit-for-bit on the same protocol."""
    import copy
    import torch
    from chronos import BaseChronosPipeline
    torch.manual_seed(seed)
    if "pipe" not in _CHRONOS_CACHE:
        _CHRONOS_CACHE["pipe"] = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-tiny", device_map=dev, torch_dtype=torch.float32)
    base = _CHRONOS_CACHE["pipe"].model
    m = copy.deepcopy(base).to(dev).train()
    ep = int(os.environ.get("CHRONOS_EPOCHS", epochs or 3))
    ep = min(ep, int(os.environ.get("CHRONOS_MAX_EPOCHS", "4")))
    opt = torch.optim.AdamW(m.parameters(), lr=float(os.environ.get("CHRONOS_LR", "1e-4")))
    xb = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    yb = torch.tensor(Ytr, dtype=torch.float32, device=dev)
    n = len(xb)
    for _ in range(ep):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, 64):
            idx = perm[s: s + 64]
            opt.zero_grad()
            loss = m(context=xb[idx], target=yb[idx]).loss
            loss.backward()
            opt.step()
    return _ChronosWrap(m.eval(), H=yb.shape[1])


def train_model(Xtr, Ytr, dev, seed, epochs=None):
    if TS_MODEL == "chronos":
        return train_chronos(Xtr, Ytr, dev, seed, epochs)
    return train_dlinear(Xtr, Ytr, dev, seed, epochs)


def train_dlinear(Xtr, Ytr, dev, seed, epochs=None):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    epochs = epochs or EPOCHS

    class DLinear(nn.Module):
        def __init__(self, L, H, k=25):
            super().__init__()
            self.k = k
            self.lt = nn.Linear(L, H)
            self.ls = nn.Linear(L, H)

        def forward(self, x):                          # x: (B, L)
            pad = torch.nn.functional.pad(x, (self.k // 2, self.k // 2), mode="replicate")
            trend = torch.nn.functional.avg_pool1d(pad.unsqueeze(1), self.k, 1).squeeze(1)
            trend = trend[:, : x.shape[1]]
            return self.lt(trend) + self.ls(x - trend)

    m = DLinear(L, H).to(dev).train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    xb = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    yb = torch.tensor(Ytr, dtype=torch.float32, device=dev)
    n = len(xb)
    order_hash = hashlib.sha256()
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        if PAIRED_RNG:
            order_hash.update(perm.detach().cpu().numpy().tobytes())
        for s in range(0, n, 256):
            idx = perm[s : s + 256]
            opt.zero_grad()
            loss = ((m(xb[idx]) - yb[idx]) ** 2).mean()
            loss.backward()
            opt.step()
    if PAIRED_RNG:
        globals()["_LAST_TRAIN_ORDER_SHA12"] = order_hash.hexdigest()[:12]
    return m


def mase(model, Xte, Yte, dev):
    import torch
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte, dtype=torch.float32, device=dev)).cpu().numpy()
    mae = np.abs(pred - Yte).mean()
    naive = np.abs(Yte - Xte[:, -1:]).mean()      # naive: last value carried forward
    return float(mae / (naive + 1e-9))




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
    import torch

    from mmdataselect.datatypes import Modality, UnifiedRecord
    from mmdataselect.fusion.console import MultiActorConsole
    from mmdataselect.fusion.adaptive import AdaptiveController
    from mmdataselect.selectors.budget_select import BudgetSelector
    from mmdataselect.selectors.external_baselines import (
        d4, density_select, dmf_dynamic, dmf_published_update, dsdm_scores,
        herding, kcenter_greedy, quadmix, quadmix_published_core, semdedup)
    from mmdataselect.signals import InfluenceSignal, RedundancySignal, minmax
    from mmdataselect.utils.pairing import arrays_sha256, reset_rng, sel_sha12, stable_seed

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    series = load_series()
    mu, sd = series[: int(0.7 * len(series))].mean(), series[: int(0.7 * len(series))].std()
    series = (series - mu) / (sd + 1e-9)
    rng = np.random.default_rng(SEED)
    # three mutually exclusive time segments (pool | val | test) with an L+H gap between
    # them, so no window of one segment shares any time step with another segment. Windows
    # WITHIN a segment still overlap (correlated); the paper reports this honestly and the
    # generalization remark uses the effective (not nominal) validation size.
    max_start = len(series) - L - H
    split = int(0.7 * max_start)
    gap = L + H
    pool_starts = rng.choice(np.arange(max(1, split - gap)), size=min(POOL_N, max(1, split - gap)), replace=False)
    rest = np.arange(split, max_start)
    half = len(rest) // 2
    val_zone = rest[: max(1, half - gap)]
    test_zone = rest[half:]
    if os.environ.get("TS_VAL_MODE", "full") == "recent":
        # rolling validation (Theorem C minimal version): keep only the most recent third of
        # the validation zone, i.e. the segment temporally closest to test. Aligning the
        # measurement segment with the deployment segment shrinks the drift term in the
        # tracking bound; the paper reports full vs recent side by side.
        keep = max(1, len(val_zone) // 3)
        val_zone = val_zone[-keep:]
    val_starts = rng.choice(val_zone, size=min(VAL_N, len(val_zone)), replace=False)
    test_starts = rng.choice(test_zone, size=min(TEST_N, len(test_zone)), replace=False)
    _export_split_ids("timeseries", SEED, {
        "arm": "timeseries", "seed": int(SEED), "dataset": TS_DATASET,
        "series_len": int(len(series)), "L": int(L), "H": int(H),
        "ts_val_mode": os.environ.get("TS_VAL_MODE", "full"),
        "pool_ids": [int(i) for i in pool_starts],
        "val_ids": [int(i) for i in val_starts],
        "test_ids": [int(i) for i in test_starts],
        "counts": {"pool": int(len(pool_starts)), "val": int(len(val_starts)),
                   "test": int(len(test_starts))},
        "rng_recipe": ("rng=np.random.default_rng(SEED); max_start=len(series)-L-H; "
                       "split=int(0.7*max_start); gap=L+H; "
                       "pool_ids=rng.choice(arange(max(1,split-gap)), size=min(POOL_N,max(1,split-gap)), replace=False); "
                       "rest=arange(split,max_start); half=len(rest)//2; "
                       "val_zone=rest[:max(1,half-gap)] (TS_VAL_MODE=recent keeps last third); "
                       "test_zone=rest[half:]; "
                       "val_ids=rng.choice(val_zone, size=min(VAL_N,len(val_zone)), replace=False); "
                       "test_ids=rng.choice(test_zone, size=min(TEST_N,len(test_zone)), replace=False); "
                       f"POOL_N={POOL_N}, VAL_N={VAL_N}, TEST_N={TEST_N}, L={L}, H={H}; "
                       "ids are window START positions in the z-normalized OT series "
                       "(normalization uses first 70% mean/std); the seeded draws follow ONE shared "
                       "rng stream in this exact call order"),
    })

    Xp, Yp = make_windows(series, pool_starts)
    Xval, Yval = make_windows(series, val_starts)
    Xt, Yt = make_windows(series, test_starts)
    Xp, Yp, tag = inject_noise(Xp, Yp, SEED)
    n = len(Xp)
    budget = int(BUDGET_FRAC * n)

    # ---- three time-series channels ----
    auth = np.array([abs(_autocorr1(Xp[i])) for i in range(n)])              # temporal structure
    Xn = Xp / (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-8)
    # Chunked kNN (audit #OOM): a full n x n similarity matrix is 57.6GB at n=120000 (float32),
    # plus another ~115GB for a full argsort -- this OOM-killed a container with a 72GB cgroup
    # limit with no Python traceback (see baselines/deepcore_original/run_original_protocol.py's
    # matching fix). Currently n stays in the low thousands here so this was latent, not yet hit,
    # but the fix is a straight port of the same chunked, argpartition-based rule (mathematically
    # identical to the unchunked S/argsort version -- verified by direct comparison on random data).
    _chunk = 2048
    redundancy = np.zeros(n, dtype=np.float64)
    for _s0 in range(0, n, _chunk):
        _sims = Xn[_s0:_s0 + _chunk] @ Xn.T
        for _r in range(_sims.shape[0]):
            _sims[_r, _s0 + _r] = -1.0                              # drop self (matches fill_diagonal(-1.0))
        _idx = np.argpartition(-_sims, KNN, axis=1)[:, :KNN]        # k nearest neighbours
        _rows = np.arange(_sims.shape[0])[:, None]
        redundancy[_s0:_s0 + _chunk] = 1.0 - _sims[_rows, _idx].mean(axis=1)     # window novelty
    ref = rng.permutation(np.where(tag == "high")[0])[:400]                  # clean reference forecaster
    ref_m = train_model(Xp[ref], Yp[ref], dev, SEED, epochs=40)
    with torch.no_grad():
        pred_all = ref_m(torch.tensor(Xp, dtype=torch.float32, device=dev)).cpu().numpy()
    influence = -np.abs(pred_all - Yp).mean(axis=1)                          # -error = forecastable/on-dist

    feats = Xn.astype(float)
    recs = [UnifiedRecord(id=str(i), modality=Modality.TEXT, domain="timeseries", text="") for i in range(n)]
    console = MultiActorConsole(
        [("redundancy", RedundancySignal()), ("influence", InfluenceSignal())],
        weights=np.log(np.array([1 - W_INFL, W_INFL]) + 1e-9),
        conflict_gate=True, anneal=0.6, group_key="domain", trust_region=0.5, ema_beta=0.3, min_weight=0.02,
    )
    imp_dyn = console.importance(recs, scores=np.stack([minmax(redundancy), minmax(influence)], axis=0), progress=0.5)

    _vperm = np.random.default_rng(SEED + 41).permutation(len(Xval))
    _v1, _v2 = _vperm[: len(_vperm) // 2], _vperm[len(_vperm) // 2 :]
    Xval1, Yval1, Xval2, Yval2 = Xval[_v1], Yval[_v1], Xval[_v2], Yval[_v2]

    # temporal sub-segments of V2 (sorted by window start), with an L+H gap between them so
    # no window of one sub-segment shares time steps with the other; both sub-segments are
    # scored against ONE global naive denominator computed on all of V2 (a per-segment
    # denominator would make the worst segment a scale artifact).
    _v2starts = val_starts[_v2]
    _ord2 = np.argsort(_v2starts)
    _half2 = len(_ord2) // 2
    _bnd = _v2starts[_ord2[_half2]]
    _t1 = np.array([i for i in _ord2[:_half2] if _v2starts[i] + L + H <= _bnd], dtype=int)
    _t2 = _ord2[_half2:]
    _naive2 = float(np.abs(Yval2 - Xval2[:, -1:]).mean())

    def _err_num(m, X, Y):
        import torch as _th
        m.eval()
        with _th.no_grad():
            pred = m(_th.tensor(X, dtype=_th.float32, device=dev)).cpu().numpy()
        return float(np.abs(Y - pred).mean())

    def _paired_subset(sub):
        return sorted(int(i) for i in sub) if PAIRED_RNG else [int(i) for i in sub]

    def _fit_seed(sub, stage):
        # Fair paired comparison: every method in one paper seed receives the same
        # initialization. The selected subset must not leak into the initialization.
        return reset_rng(SEED, stage) if PAIRED_RNG else SEED

    def neg_mase_val(sub):           # ADJUDICATION on V2
        sub = _paired_subset(sub)
        m = train_model(Xp[sub], Yp[sub], dev, _fit_seed(sub, "v2-fit"), epochs=40)
        if os.environ.get("ROBUST_VAL", "0") == "1" and len(_t1) > 8:
            # robust adjudication: worst TEMPORAL sub-segment, shared global denominator.
            return -max(_err_num(m, Xval2[_t1], Yval2[_t1]), _err_num(m, Xval2[_t2], Yval2[_t2])) / _naive2
        return -mase(m, Xval2, Yval2, dev)

    def neg_mase_val1(sub):          # CONSTRUCTION on V1
        sub = _paired_subset(sub)
        return -mase(train_model(Xp[sub], Yp[sub], dev, _fit_seed(sub, "v1-fit"), epochs=40), Xval1, Yval1, dev)

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
            out = []
            for c in range(k):
                mem = np.where(km.labels_ == c)[0]
                if len(mem):
                    out.append(int(mem[np.argmin(np.linalg.norm(Xn[mem] - km.cluster_centers_[c], axis=1))]))
            return out[:budget]
        if method == "auth_only":
            return list(np.argsort(-auth)[:budget])
        if method == "auth2_only":       # mechanism-matched v2: temporal structure AND inlierness
            def _r01(v):                 # (corrupt windows are feature-space outliers; flat/shuffled
                r = np.argsort(np.argsort(v))    # have no temporal structure -> conjunction)
                return r / (len(v) - 1 + 1e-9)
            inlier = 1.0 - redundancy    # mean sim to k nearest neighbours
            a2 = np.minimum(_r01(auth), _r01(inlier))
            return list(np.argsort(-a2)[:budget])
        if method == "influence_only":
            return list(np.argsort(-influence)[:budget])
        if method == "herding":          # geometric coreset (Welling 2009 / DeepCore)
            return herding(Xn, budget)
        if method == "kcenter":          # k-center greedy coreset (Sener & Savarese 2018 / DeepCore)
            return kcenter_greedy(Xn, budget, seed=SEED)
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
            w = dsdm_scores(neg_mase_val1, n, k_runs=int(os.environ.get("DSDM_RUNS", "12")), seed=SEED)
            return [int(i) for i in np.argsort(-w)[:budget]]
        if method == "dmf":              # DMF faithful: dynamic multi-channel reweighting (Yang et al. 2025)
            ch = np.stack([minmax(auth), minmax(influence), minmax(redundancy)], axis=0)
            return dmf_dynamic(ch, budget, val_reward=neg_mase_val1, rounds=3, seed=SEED)
        if method == "dmf_pub":          # Multi-Actor Eqs. 6--8, published-update transfer
            ch = np.stack([minmax(auth), minmax(influence), minmax(redundancy)], axis=0)
            return dmf_published_update(ch, budget, val_reward=neg_mase_val1, rounds=6, seed=SEED)
        if method == "mmdataselect":
            thr = float(np.quantile(auth, AUTH_Q))
            imp = imp_dyn.copy()
            imp[auth < thr] = -1e9
            return BudgetSelector(lam=LAM).select(recs, imp, budget, features=feats)
        if method == "mmds_adapt":
            ctrl = AdaptiveController(lam_grid=(0.0, 0.25, 0.6), prefilter_grid=(0.0, AUTH_Q),
                                     prefilter_channel=0, seed=SEED)
            sel = ctrl.select(recs, np.stack([auth, influence, redundancy], axis=0), budget,
                              features=feats, held_out_gain=neg_mase_val,
                              construct_gain=neg_mase_val1,
                              policy_search=(os.environ.get("ADAPT_GRPO", "0") == "1"),
                              extra_strategies=[(b, (lambda bb: (lambda k: select(bb)))(b)) for b in
                                                ("mmdataselect", "auth2_only", "herding", "kcenter",
                                                 "semdedup", "density", "quadmix_pub",  # style-proxy quadmix removed from portfolio (PROTOCOL_INVALID)
                                                 "dmf", "dmf_pub", "d4", "dsdm")]
                                                + [("random", lambda k: select("random")),
                                                   ("auth_bottom", lambda k: [int(i) for i in np.argsort(auth)[:k]])])
            print(f"    [adapt] picked '{ctrl.chosen_['strategy']}' (val_negMASE={ctrl.chosen_['val_gain']:.3f})")
            globals()["_ADAPT_MANIFEST"] = {"leaderboard": list(getattr(ctrl, "leaderboard_", []) or []), "chosen": dict(getattr(ctrl, "chosen_", {}) or {}), "sel_sha12": __import__("hashlib").sha256(str(sorted(int(i) for i in sel)).encode()).hexdigest()[:12] if "sel" in dir() else None}
            return sel
        raise ValueError(method)

    print(f"timeseries={TS_DATASET} OT | L={L} H={H} pool={n} test={len(Xt)} budget={budget} dev={dev} seed={SEED}")
    print(f"  tags: {dict(zip(*np.unique(tag, return_counts=True)))}")
    globals()["_PAIRING_MANIFEST"] = {
        "pool_sha256": arrays_sha256(Xp, Yp, tag.astype(str)),
        "validation_sha256": arrays_sha256(Xval, Yval),
        "test_sha256": arrays_sha256(Xt, Yt),
        "shared_initialization_rule": "stable_seed(paper_seed, fit-stage)",
        "training_input_order": "sorted selected integer ids then seeded epoch permutations",
    }
    results = []
    _sel_only = os.environ.get("SELECT_ONLY", "0") == "1"
    for m in METHODS:
        t0 = time.time()
        sel = _paired_subset(select(m))
        selection_sha = sel_sha12(sel)
        if _sel_only:  # selection-manifest replay: NO training. train_order_sha12 is
            # produced INSIDE train_model for this arm, so replay can verify only the
            # selection hash here (train order stays HASH_ONLY_NOT_REPLAYED).
            results.append({"method": m, "n_selected": len(sel),
                            "selected_ids": [int(i) for i in sel],
                            "training_order": None, "sel_sha12": selection_sha,
                            "train_order_sha12": None})
            print(f"  [select-only] {m:16} n={len(sel)} sel={selection_sha}")
            continue
        fit_seed = _fit_seed(sel, "final-fit")
        mdl = train_model(Xp[sel], Yp[sel], dev, fit_seed)
        sc = mase(mdl, Xt, Yt, dev)
        hi = float(np.mean(tag[sel] == "high"))
        row = {"method": m, "n": len(sel), "clean%": round(hi, 3), "mase": round(sc, 4)}
        if PAIRED_RNG:
            row.update({"sel_sha12": selection_sha,
                        "fit_seed": fit_seed,
                        "train_order_sha12": globals().get("_LAST_TRAIN_ORDER_SHA12")})
        results.append(row)
        print(f"  {m:16} n={len(sel):5} clean%={hi:.2f} MASE={sc:.4f} ({time.time()-t0:.1f}s)")
    print(f"\n==== TIMESERIES ({TS_DATASET}, model={TS_MODEL}, MASE lower=better) ====")
    for r in sorted(results, key=lambda r: r.get("mase", 1e9)):
        if "mase" not in r:
            continue  # select-only manifest rows carry no metrics
        print(f"  {r['method']:16} MASE={r['mase']:.4f} clean%={r['clean%']:.2f} n={r['n']}")
    p = _trial_dump(results, "timeseries", TS_DATASET, SEED,
                    {"model": TS_MODEL, "pool": POOL_N, "budget": BUDGET_FRAC,
                     "noise": NOISE_FRAC, "L": L, "H": H,
                     "paired_rng": int(PAIRED_RNG)})
    print(f"saved -> {p}")
    if not os.environ.get("RUN_ID"):
        out = os.path.join(_REPO, "outputs", "timeseries")
        os.makedirs(out, exist_ok=True)
        import json
        json.dump(results, open(os.path.join(out, f"results_seed{SEED}.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
