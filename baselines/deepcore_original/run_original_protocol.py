"""EL2N / GraNd / CCS / Random on the ORIGINAL protocol: ResNet-18 trained from scratch
on CIFAR-10, scores from an early checkpoint (Paul et al. 2021 / Zheng et al. 2023).

This is the "did we reproduce it right" evidence: it reproduces the *published qualitative
claim* in the *original data + original model*, not the frozen-feature proxy used in the
cross-modal bench.
  - EL2N (Paul 2021): ||softmax(logit)-onehot(y)|| at epoch ~10, keep hardest.
  - GraNd (Paul 2021): expected loss-gradient norm at epoch ~10, keep hardest.
  - CCS   (Zheng 2023): prune hardest cutoff, then stratified sample by difficulty.
  - Random.
Report: test acc at a given keep fraction. Published finding to reproduce:
  at HIGH pruning (keep 30%), CCS >> EL2N (EL2N collapses because it keeps only hard/noisy),
  and both < random is FALSE for EL2N at moderate pruning on clean CIFAR-10.
Env: KEEP (fraction, default 0.3), SCORE_EPOCH (default 10), TRAIN_EPOCH (default 40),
     SEED, POOL (train subset size, default full 50000).
"""
import os, sys, time, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv, torchvision.transforms as T
from torchvision.models import resnet18
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("SEED","0")); torch.manual_seed(SEED); np.random.seed(SEED)
KEEP = float(os.environ.get("KEEP","0.3"))
SCORE_EPOCH = int(os.environ.get("SCORE_EPOCH","10"))
TRAIN_EPOCH = int(os.environ.get("TRAIN_EPOCH","40"))
POOL = int(os.environ.get("POOL","50000"))
DATA = os.path.join(os.path.dirname(__file__), "..","..","data","cifar10")
DATASET = os.environ.get("DATASET", "cifar10")          # cifar10 | imagenet100
IN_RES = int(os.environ.get("IN_RES", "112"))            # ImageNet-100 train/eval resolution
VAL_N = int(os.environ.get("VAL_N", "5000"))             # adjudication split carved from train

import numpy as _np2
class _NpzCIFAR(torch.utils.data.Dataset):
    def __init__(self, X, y, transform):
        self.X=X; self.y=y; self.tf=transform
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        from PIL import Image
        img=Image.fromarray(self.X[i]); return self.tf(img), int(self.y[i])

def cifar():
    z=_np2.load(os.path.join(os.path.dirname(__file__),"..","..","data","cifar10_np","cifar10.npz"))
    tf_tr = T.Compose([T.RandomCrop(32,padding=4),T.RandomHorizontalFlip(),T.ToTensor(),
                       T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    tf_te = T.Compose([T.ToTensor(),T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    tr=_NpzCIFAR(z["Xtr"],z["ytr"],tf_tr); te=_NpzCIFAR(z["Xte"],z["yte"],tf_te)
    tr._raw=(z["Xtr"],z["ytr"]); return tr,te

class _HFImage(torch.utils.data.Dataset):
    def __init__(self, hfds, transform):
        self.ds = hfds; self.tf = transform
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        r = self.ds[int(i)]
        img = r["image"].convert("RGB")
        return self.tf(img), int(r["label"])


def imagenet100():
    from datasets import load_dataset
    import glob
    # prefer the fully-downloaded local parquet shards (no hub resolution: the link is
    # flaky and offline hub mode requires a prior successful load)
    hf_home = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
    # 0) processed Arrow cache (fully offline, fastest)
    arrow_tr = sorted(glob.glob(hf_home + "/datasets/clane9___imagenet-100/**/imagenet-100-train*.arrow", recursive=True))
    arrow_te = sorted(glob.glob(hf_home + "/datasets/clane9___imagenet-100/**/imagenet-100-validation*.arrow", recursive=True))
    if arrow_tr and arrow_te:
        from datasets import Dataset, concatenate_datasets
        dtr = concatenate_datasets([Dataset.from_file(a) for a in arrow_tr])
        dte = concatenate_datasets([Dataset.from_file(a) for a in arrow_te])
        print(f"[data] arrow cache: train={len(dtr)} val={len(dte)}")
    else:
        snap = glob.glob(hf_home + "/hub/datasets--clane9--imagenet-100/snapshots/*/data")
        if not snap:
            snap = []
    if arrow_tr and arrow_te:
        pass
    elif snap:
        tr_files = sorted(glob.glob(snap[0] + "/train-*.parquet"))
        te_files = sorted(glob.glob(snap[0] + "/validation-*.parquet"))
        dtr = load_dataset("parquet", data_files=tr_files, split="train")
        dte = load_dataset("parquet", data_files=te_files, split="train")
    else:
        dtr = load_dataset("clane9/imagenet-100", split="train")
        dte = load_dataset("clane9/imagenet-100", split="validation")
    norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    tf_tr = T.Compose([T.RandomResizedCrop(IN_RES, scale=(0.35, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    tf_te = T.Compose([T.Resize(int(IN_RES * 1.15)), T.CenterCrop(IN_RES), T.ToTensor(), norm])
    tr = _HFImage(dtr, tf_tr); te = _HFImage(dte, tf_te)
    tr._eval_ds = _HFImage(dtr, tf_te)
    tr._labels = np.array(dtr["label"])
    return tr, te


def get_data():
    return imagenet100() if DATASET == "imagenet100" else cifar()


def net():
    if DATASET == "imagenet100":
        return resnet18(num_classes=100).to(DEV)      # standard ImageNet stem at IN_RES
    m = resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool = nn.Identity()  # CIFAR stem
    return m.to(DEV)

def train(model, ds, idx, epochs, bs=256, lr=0.1, capture_scores_at=None, n=50000):
    sub = torch.utils.data.Subset(ds, idx)
    g = torch.Generator(); g.manual_seed(SEED)          # fixed data order per seed (audit #5)
    dl = torch.utils.data.DataLoader(sub,batch_size=bs,shuffle=True,
                                     num_workers=int(os.environ.get("NW","4")),
                                     generator=g,drop_last=False)
    opt = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    el2n = np.zeros(n); grand = np.zeros(n)
    feats = np.zeros((n, 512), dtype=np.float32); labs = np.zeros(n, dtype=int)
    for ep in range(epochs):
        model.train()
        for xb,yb in dl:
            xb,yb=xb.to(DEV),yb.to(DEV); opt.zero_grad()
            out=model(xb); loss=F.cross_entropy(out,yb); loss.backward(); opt.step()
        sch.step()
        if capture_scores_at is not None and ep+1==capture_scores_at:
            el2n,grand,feats,labs = score_all(model, ds, idx, n)
    return el2n, grand, feats, labs

def _eval_view(ds):
    if hasattr(ds, "_eval_ds"):
        return ds._eval_ds
    tf_te=T.Compose([T.ToTensor(),T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    X,y=ds._raw; return _NpzCIFAR(X,y,tf_te)


def score_all(model, ds, idx, n):
    """EL2N + GraNd(last-layer grad-norm proxy) + penultimate features + labels."""
    model.eval()
    ncls = 100 if DATASET == "imagenet100" else 10
    raw=_eval_view(ds)
    dl=torch.utils.data.DataLoader(torch.utils.data.Subset(raw,list(idx)),batch_size=256,
                                   num_workers=int(os.environ.get("NW","0")))
    el2n=_np2.zeros(n); grand=_np2.zeros(n); pos=0
    feats=_np2.zeros((n,512),dtype=_np2.float32); labs=_np2.zeros(n,dtype=int)
    # penultimate hook
    buf={}
    h=model.avgpool.register_forward_hook(lambda m,i,o: buf.__setitem__("z", o.flatten(1).detach()))
    with torch.no_grad():
        for xb,yb in dl:
            xb=xb.to(DEV); logit=model(xb); p=F.softmax(logit,1).cpu().numpy()
            oh=_np2.eye(ncls)[yb.numpy()]
            err=_np2.linalg.norm(p-oh,axis=1)
            z=buf["z"].cpu().numpy()
            featnorm=_np2.linalg.norm(z,axis=1)          # ||penultimate phi|| (audit #1:
            # last-layer grad of CE = (p-onehot) (x) phi, so norm = err * ||phi||)
            for j,g in enumerate(idx[pos:pos+len(yb)]):
                el2n[g]=err[j]; grand[g]=err[j]*max(featnorm[j],1e-6)
                feats[g]=z[j]; labs[g]=int(yb[j])
            pos+=len(yb)
    h.remove()
    return el2n,grand,feats,labs

def ccs_select(diff, keep_n, cutoff=0.1, bins=50, seed=0):
    """CCS (Zheng et al., ICLR 2023), official form: prune the hardest `cutoff`
    fraction, split the remaining difficulty RANGE into `bins` EQUAL-WIDTH strata,
    allocate the budget evenly, and REDISTRIBUTE the unused allocation of sparse
    strata to the others (ascending-size order), sampling uniformly inside a stratum."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    order = np.argsort(diff)
    kept = order[: int(n * (1 - cutoff))]
    dd = diff[kept]
    lo, hi = float(dd.min()), float(dd.max())
    edges = np.linspace(lo, hi, bins + 1)
    strata = []
    for b in range(bins):
        top = edges[b + 1] if b == bins - 1 else edges[b + 1]
        m = kept[(dd >= edges[b]) & ((dd <= top) if b == bins - 1 else (dd < top))]
        strata.append(m)
    out = []
    budget = keep_n
    for i, sset in enumerate(sorted(strata, key=len)):
        alloc = budget // (len(strata) - i) if (len(strata) - i) else 0
        take = min(len(sset), alloc)
        if take > 0:
            out += [int(x) for x in rng.permutation(sset)[:take]]
        budget -= take
    return np.array(out[:keep_n])


def _ccs_toy_test():
    """equal-width bins + dynamic reallocation sanity (audit #4)."""
    diff = np.concatenate([np.zeros(2), np.full(98, 0.5)])   # sparse stratum: 2 items
    sel = ccs_select(diff, keep_n=50, cutoff=0.0, bins=2, seed=0)
    assert len(sel) == 50, len(sel)
    n_easy = int((diff[sel] == 0).sum())
    assert n_easy == 2, f"sparse stratum should contribute all 2, got {n_easy}"
    assert int((diff[sel] == 0.5).sum()) == 48, "reallocation to dense stratum failed"
    print("[ccs-toy] equal-width + dynamic reallocation OK")


def _knn_label_agree(feats, labs, k=15, chunk=2048):
    """Exact top-k kNN label agreement without materializing the NxN matrix
    (audit #OOM: 120k^2 float32 = 57.6GB + argsort 115GB). Peak extra memory =
    chunk x N float32 (~1GB at chunk=2048, N=120k)."""
    fp = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    n = len(fp)
    agree = np.zeros(n, dtype=np.float32)
    for s0 in range(0, n, chunk):
        sims = fp[s0:s0 + chunk] @ fp.T
        for r in range(sims.shape[0]):
            sims[r, s0 + r] = -2.0                       # drop self
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        agree[s0:s0 + chunk] = (labs[idx] == labs[s0:s0 + chunk, None]).mean(axis=1)
    return agree


def _acc(m, dl):
    m.eval(); cor=tot=0
    with torch.no_grad():
        for xb,yb in dl:
            pred=m(xb.to(DEV)).argmax(1).cpu(); cor+=(pred==yb).sum().item(); tot+=len(yb)
    return cor/max(1,tot)


def main():
    tr,te=get_data()
    rng=np.random.default_rng(SEED)
    perm=rng.permutation(len(tr))
    val_idx=perm[:VAL_N]                                   # adjudication split, never in pool
    pool_idx=perm[VAL_N:VAL_N+POOL]
    full=np.asarray(pool_idx)
    import hashlib as _hh
    SCORE_RUNS = int(os.environ.get("SCORE_RUNS", "3"))   # Data-Diet: average over inits
    _code12 = _hh.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]
    _ck = os.path.join(os.path.dirname(__file__), "..", "..", "outputs",
                       f"score_ckpt_{DATASET}_s{SEED}_e{SCORE_EPOCH}_r{SCORE_RUNS}_{_code12}.npz")
    if os.path.exists(_ck):
        z = np.load(_ck)
        el2n, grand, feats, labs = z["el2n"], z["grand"], z["feats"], z["labs"]
        print(f"[score] checkpoint loaded ({_ck})")
    else:
        print(f"[score] ResNet-18 x{SCORE_RUNS} inits to epoch {SCORE_EPOCH} (dev={DEV}, data={DATASET}, pool={len(full)})")
        t0=time.time()
        el2n = np.zeros(len(tr)); grand = np.zeros(len(tr))
        for r_ in range(SCORE_RUNS):
            torch.manual_seed(SEED * 1000 + r_)
            e_, g_, feats, labs = train(net(), tr, full, SCORE_EPOCH,
                                        capture_scores_at=SCORE_EPOCH, n=len(tr))
            el2n += e_; grand += g_
        el2n /= SCORE_RUNS; grand /= SCORE_RUNS
        os.makedirs(os.path.dirname(_ck), exist_ok=True)
        np.savez(_ck, el2n=el2n, grand=grand, feats=feats, labs=labs)
        print(f"[score] done in {time.time()-t0:.0f}s -> {_ck}")
    keep_n=int(len(full)*KEEP)
    _only = [s for s in os.environ.get("SUBSETS", "").split(",") if s]  # computed early: gates
    # which of the expensive O(n*k*d) geometric subsets below actually need to run.
    # authenticity candidate: kNN label agreement in penultimate feature space (pool only)
    lp=labs[full]
    auth=_knn_label_agree(feats[full], lp, k=15)
    subsets={
      "random": rng.permutation(full)[:keep_n],
      "el2n":   full[np.argsort(-el2n[full])[:keep_n]],
      "grand":  full[np.argsort(-grand[full])[:keep_n]],
      "ccs":    full[np.asarray(ccs_select(el2n[full], keep_n, seed=SEED), dtype=int)],  # pool-local -> global ids
      "auth":   full[np.argsort(-auth)[:keep_n]],
    }
    if os.environ.get("GEOM_CORESETS", "1") == "1":
        # DeepCore geometric coresets on the penultimate features (original protocol,
        # same CIFAR-10 testbed): herding (Welling 2009) + k-center greedy (Sener 2018).
        from mmdataselect.selectors.external_baselines import herding, kcenter_greedy
        fp = feats[full]
        if not _only or "herding" in _only:
            subsets["herding"] = full[np.asarray(herding(fp, keep_n), dtype=int)]
            print(f"[progress] herding done @ {time.time():.0f}", flush=True)
        if not _only or "kcenter" in _only:
            subsets["kcenter"] = full[np.asarray(kcenter_greedy(fp, keep_n), dtype=int)]
            print(f"[progress] kcenter done @ {time.time():.0f}", flush=True)

    # ---- 8 additional baselines, ported (not copy-pasted) from run_vision_experiment.py's
    # select() onto this file's local/global-id convention: fp/auth/influence/redundancy are
    # indexed pool-locally (aligned with `full`); `full[local_idx]` maps back to global ids.
    # Pure additions below this line -- nothing above (random/el2n/grand/ccs/auth) is touched.
    fp = feats[full]                                          # (pool, 512) penultimate features

    # Coverage-only/Coreset: k-means, nearest-to-centroid representative kept per cluster
    # (run_vision_experiment.py's `method == "coreset"` block, same k-means-medoid rule).
    # MiniBatchKMeans, not KMeans: this protocol's keep_n runs into the thousands (13500 for
    # CIFAR-10, 15000 for ImageNet-100), and full-batch Lloyd's iterations at that many
    # clusters over n~50000 points do not finish in practical time. MiniBatchKMeans keeps the
    # same medoid-selection rule below with a tractable per-step cost.
    from sklearn.cluster import MiniBatchKMeans
    if not _only or "coreset" in _only:
        _k = min(keep_n, len(full))
        _km = MiniBatchKMeans(n_clusters=_k, n_init=3, batch_size=max(1024, _k), random_state=SEED).fit(fp)
        _coreset_local = []
        for _c in range(_k):
            _members = np.where(_km.labels_ == _c)[0]
            if len(_members):
                _d = np.linalg.norm(fp[_members] - _km.cluster_centers_[_c], axis=1)
                _coreset_local.append(int(_members[np.argmin(_d)]))
        subsets["coreset"] = full[np.asarray(_coreset_local[:keep_n], dtype=int)]
        print(f"[progress] coreset done @ {time.time():.0f}", flush=True)

    from mmdataselect.selectors.external_baselines import (
        semdedup, density_select, quadmix_published_core, dmf_published_update)
    subsets["semdedup"] = full[np.asarray(semdedup(fp, keep_n, seed=SEED), dtype=int)]
    print(f"[progress] semdedup done @ {time.time():.0f}", flush=True)
    subsets["density"] = full[np.asarray(density_select(fp, keep_n, knn=10, seed=SEED), dtype=int)]
    print(f"[progress] density done @ {time.time():.0f}", flush=True)
    # auth is already pool-local (length == len(full), see `auth=_knn_label_agree(...)` above),
    # so it is passed directly as the `quality` channel (NOT `auth[full]`, which would index a
    # pool-local array with global ids and be out of range).
    subsets["quadmix_pub"] = full[np.asarray(quadmix_published_core(auth, fp, keep_n, seed=SEED), dtype=int)]
    print(f"[progress] quadmix_pub done @ {time.time():.0f}", flush=True)

    # influence: log-prob under a probe (LogisticRegression) fit on a small reference subset of
    # `full`. This protocol has no injected label noise / no "high"-quality tag to filter the
    # reference by (unlike run_vision_experiment.py's CIFAR-100 arm), so the reference is a
    # uniform random draw from `full`. Uses an INDEPENDENT rng (SEED+91), never the shared `rng`
    # that subsets["random"] already consumed above -> random/el2n/grand/ccs/auth stay
    # byte-identical to before this change.
    from sklearn.linear_model import LogisticRegression
    _rng_infl = np.random.default_rng(SEED + 91)
    _ref_n = min(2000, len(full) // 4)
    _ref_local = _rng_infl.permutation(len(full))[:_ref_n]
    _clf_infl = LogisticRegression(max_iter=200, C=1.0).fit(fp[_ref_local], lp[_ref_local])
    _proba = _clf_infl.predict_proba(fp)
    _cls_idx = {c: j for j, c in enumerate(_clf_infl.classes_)}
    influence = np.array([np.log(_proba[i, _cls_idx[lp[i]]] + 1e-9) if lp[i] in _cls_idx else -20.0
                          for i in range(len(full))])
    subsets["influence_only"] = full[np.argsort(-influence)[:keep_n]]
    print(f"[progress] influence_only done @ {time.time():.0f}", flush=True)

    # redundancy: 1 - mean cosine similarity to the 15 nearest neighbours (novelty); the exact
    # `S = Xp @ Xp.T` / neighbour-mean-similarity rule from run_vision_experiment.py, ported to
    # L2-normalized fp. Chunked exactly like `_knn_label_agree` above (audit #OOM: a full NxN
    # similarity matrix at ImageNet-100 pool scale is 120k^2 float32 = 57.6GB, plus the argsort
    # index array at ~115GB -- this blew past this container's 72GB cgroup memory.max and got
    # silently OOM-killed with no Python traceback). Peak extra memory = chunk x N float32 (~1GB).
    _fpn = fp / (np.linalg.norm(fp, axis=1, keepdims=True) + 1e-8)
    _n_r = len(_fpn)
    _chunk_r = 2048
    redundancy = np.zeros(_n_r, dtype=np.float32)
    for _s0 in range(0, _n_r, _chunk_r):
        _sims = _fpn[_s0:_s0 + _chunk_r] @ _fpn.T
        for _r in range(_sims.shape[0]):
            _sims[_r, _s0 + _r] = -2.0                    # drop self
        _idx = np.argpartition(-_sims, 15, axis=1)[:, :15]
        _rows = np.arange(_sims.shape[0])[:, None]
        redundancy[_s0:_s0 + _chunk_r] = 1.0 - _sims[_rows, _idx].mean(axis=1)
    redundancy = redundancy.astype(float)

    # Fixed-weight-fusion (mmdataselect): authenticity prefilter -> influence x diversity console
    # fusion -> diminishing-returns budget selection. Exact port of run_vision_experiment.py's
    # `method == "mmdataselect"` block (same console/selector params), re-targeted at this
    # file's pool-local recs/features.
    from mmdataselect.datatypes import Modality, UnifiedRecord
    from mmdataselect.fusion.console import MultiActorConsole
    from mmdataselect.selectors.budget_select import BudgetSelector
    from mmdataselect.signals import InfluenceSignal, RedundancySignal, minmax
    _AUTH_Q = float(os.environ.get("AUTH_Q", "0.25"))
    _W_INFL = float(os.environ.get("W_INFL", "0.5"))
    _LAM = float(os.environ.get("LAM", "0.5"))
    _recs = [UnifiedRecord(id=str(i), modality=Modality.IMAGE if hasattr(Modality, "IMAGE") else Modality.TEXT,
                           domain="image", text="") for i in range(len(full))]
    _console = MultiActorConsole(
        [("redundancy", RedundancySignal()), ("influence", InfluenceSignal())],
        weights=np.log(np.array([1 - _W_INFL, _W_INFL]) + 1e-9),
        conflict_gate=True, anneal=0.6, group_key="domain", trust_region=0.5, ema_beta=0.3, min_weight=0.02,
    )
    _imp_dyn = _console.importance(_recs, scores=np.stack([minmax(redundancy), minmax(influence)], axis=0),
                                   progress=0.5)
    _thr = float(np.quantile(auth, _AUTH_Q))
    _imp = _imp_dyn.copy()
    _imp[auth < _thr] = -1e9
    subsets["mmdataselect"] = full[np.asarray(
        BudgetSelector(lam=_LAM).select(_recs, _imp, keep_n, features=fp.astype(float)), dtype=int)]
    print(f"[progress] mmdataselect done @ {time.time():.0f}", flush=True)

    # DMF-pub: published-update multi-actor reweighting over [auth, influence, redundancy].
    # Design decision for val_reward (no `val_idx` feature extraction, see task note): `feats`
    # is only ever populated by score_all() on idx=full, so `val_idx`'s features do not exist
    # without an extra forward pass through a trained backbone. Simpler and equally valid: carve
    # a held-out LOCAL slice off the tail of `full` (last 10%, `_hold_local`) and EXCLUDE it from
    # DMF's own candidate universe (`_sel_local` = the other 90%), so DMF can never select into
    # its own validation slice and val_reward is always measured on data disjoint from whatever
    # probe it just fit -- reuses already-computed feats/labs at zero extra extraction cost
    # instead of the val_idx route.
    _n_hold = max(1, int(0.1 * len(full)))
    _hold_local = np.arange(len(full) - _n_hold, len(full))
    _sel_local = np.arange(len(full) - _n_hold)
    channel_scores = np.stack([minmax(auth[_sel_local]), minmax(influence[_sel_local]),
                               minmax(redundancy[_sel_local])], axis=0)

    def _dmf_val_reward(sub):
        sub_local = _sel_local[np.asarray(list(sub), dtype=int)]
        _clf_dmf = LogisticRegression(max_iter=150, C=1.0).fit(fp[sub_local], lp[sub_local])
        _pred = _clf_dmf.predict(fp[_hold_local])
        return float((_pred == lp[_hold_local]).mean())

    _dmf_sel = dmf_published_update(channel_scores, keep_n, val_reward=_dmf_val_reward, rounds=6, seed=SEED)
    subsets["dmf_pub"] = full[_sel_local[np.asarray(_dmf_sel, dtype=int)]]
    print(f"[progress] dmf_pub done @ {time.time():.0f}", flush=True)

    dl_te=torch.utils.data.DataLoader(te,batch_size=256,num_workers=int(os.environ.get("NW","0")))
    dl_val=torch.utils.data.DataLoader(torch.utils.data.Subset(_eval_view(tr),list(val_idx)),
                                       batch_size=256,num_workers=int(os.environ.get("NW","0")))
    print(f"\n==== ORIGINAL PROTOCOL: ResNet-18 from scratch, {DATASET}, keep {KEEP:.0%} ({keep_n}) ====")
    rows=[]
    torch.manual_seed(SEED)
    _base_sd = {k: v.clone() for k, v in net().state_dict().items()}  # same init for every method (audit #5)
    _only = [s for s in os.environ.get("SUBSETS", "").split(",") if s]
    if _only:   # fill-in mode: fit only the named subsets (selection/scores unchanged)
        subsets = {k: v for k, v in subsets.items() if k in _only}
        print(f"[subset-filter] fitting only: {list(subsets)}")
    from mmdataselect.utils.pairing import sel_sha12
    sel_ids = {}
    for name,idx in subsets.items():
        ids_sorted = sorted(int(i) for i in idx)
        sel_ids[name] = ids_sorted
        m=net(); m.load_state_dict(_base_sd); train(m,tr,np.asarray(idx),TRAIN_EPOCH)
        va=_acc(m,dl_val); ta=_acc(m,dl_te)
        rows.append((name,va,ta,len(idx)))
        print(f"  {name:8} val={va:.4f} test={ta:.4f}  n={len(idx)}  sel_sha12={sel_sha12(ids_sorted)}")
    # controller row: full-fidelity adjudication over the fixed candidate set (Prop 2).
    # Suppressed in fill-in mode - a ctrl row over a filtered subset would be misleading.
    if not _only:
        win=max(rows,key=lambda r:r[1])
        print(f"  {'ctrl':8} val={win[1]:.4f} test={win[2]:.4f}  n={win[3]}  picked={win[0]}")

    # ---- persist results.json (merge with any existing file so a SUBSETS-filtered
    # fill-in run adds to, never clobbers, methods already saved by a prior run) ----
    _ds_dir = {"cifar10": "cifar10_full"}.get(DATASET, DATASET)
    _out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results_canonical", "vision",
                            _ds_dir, "run_id=original-protocol-resnet18", f"seed_{SEED}")
    _out_path = os.path.join(_out_dir, "results.json")
    _existing = {}
    if os.path.exists(_out_path):
        _prev = json.load(open(_out_path))
        _existing = {r["method"]: r for r in _prev.get("results", []) if r.get("method") != "ctrl"}
    for name, va, ta, n in rows:
        _existing[name] = {"method": name, "val": round(float(va), 4), "test": round(float(ta), 4),
                            "n": int(n), "selected_ids": sel_ids[name], "sel_sha12": sel_sha12(sel_ids[name])}
    _best = max(_existing.values(), key=lambda r: r["val"])
    _out = {
        "arm": "vision", "dataset": DATASET, "seed": SEED,
        "config": {"in_res": IN_RES, "score_epoch": SCORE_EPOCH, "train_epoch": TRAIN_EPOCH,
                   "score_runs": SCORE_RUNS, "pool": POOL, "val_n": VAL_N, "keep": KEEP},
        "results": list(_existing.values()) + [{"method": "ctrl", "val": _best["val"], "test": _best["test"],
                                                 "n": _best["n"], "picked": _best["method"],
                                                 "selected_ids": _best["selected_ids"], "sel_sha12": _best["sel_sha12"]}],
    }
    os.makedirs(_out_dir, exist_ok=True)
    json.dump(_out, open(_out_path, "w"), indent=2)
    print(f"[saved] -> {_out_path}", flush=True)
if __name__=="__main__": main()
