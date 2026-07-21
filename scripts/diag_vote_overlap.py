"""Empirical check of Prop 5's complementarity conditions on the vision arm (seed 0).

Rebuilds the exact seed-0 vision setting (cached CLIP embeddings + same noise injection),
forms the candidate selections that the controller's vote-ensemble aggregates, and reports:
  - each candidate's harmful fraction |S_i ∩ H| / k;
  - pairwise overlap of harmful mis-selections (condition (i): near-disjoint);
  - fraction of harmful records selected by >= 2 candidates;
  - size of the common core C (condition (ii): |C| >= k);
  - weight balance w_(s-1)+w_(s) vs w_(1) (condition (iii));
  - the vote selection's harmful fraction vs the best member.
"""
import os, sys, numpy as np
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src")); sys.path.insert(0, _REPO)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
SEED=0; POOL_N=4000; VAL_N=800; TEST_N=2000; NOISE_FRAC=0.40; KNN=15
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression

ds = load_dataset("uoft-cs/cifar100", split="train")
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(ds))
tr_idx = perm[:POOL_N]; val_idx = perm[POOL_N:POOL_N+VAL_N]
lab_key = "fine_label"
pool_lab = np.array([ds[int(i)][lab_key] for i in tr_idx])
val_lab  = np.array([ds[int(i)][lab_key] for i in val_idx])
z = np.load(os.path.join(_REPO, f"data/processed/vision_cifar100_clip-vit-base-patch32_p{POOL_N}v{VAL_N}t{TEST_N}_s{SEED}.npz"))
Xp, Xval = z["Xp"], z["Xval"]

# same noise injection as runner
rng2 = np.random.default_rng(SEED + 7)
n = len(pool_lab); obs = pool_lab.copy(); tag = np.array(["high"]*n, dtype=object)
n_low = int(round(NOISE_FRAC*n)); low = rng2.permutation(n)[:n_low]
per = max(1, n_low//3); flip, dup, hard = low[:per], low[per:2*per], low[2*per:]
for i in flip: obs[i] = rng2.integers(100); tag[i] = "flip"
for i in dup: tag[i] = "dup"
for i in hard: tag[i] = "hard"
seeds = dup[:max(1, len(dup)//8)]
for j,i in enumerate(dup):
    Xp[i] = Xp[seeds[j % len(seeds)]] + 0.01*np.random.default_rng(i).standard_normal(Xp.shape[1])

# channels (same as runner)
# Chunked kNN (audit #OOM, see run_vision_experiment.py / run_original_protocol.py's matching fix):
# avoids materializing a full n x n similarity matrix + full argsort index array (57.6GB+115GB at
# n=120000, OOM-killed a 72GB-cgroup container with no traceback). n is small here but this keeps
# the pattern consistent -- mathematically identical to the unchunked S/argsort version since auth
# and red are both order-insensitive (mean over the top-KNN set).
_chunk = 2048
auth = np.zeros(n, dtype=np.float64)
red = np.zeros(n, dtype=np.float64)
for _s0 in range(0, n, _chunk):
    _sims = Xp[_s0:_s0 + _chunk] @ Xp.T
    for _r in range(_sims.shape[0]):
        _sims[_r, _s0 + _r] = -1.0                              # drop self (matches fill_diagonal(-1.0))
    _idx = np.argpartition(-_sims, KNN, axis=1)[:, :KNN]        # k nearest neighbours (unordered)
    _rows = np.arange(_sims.shape[0])[:, None]
    auth[_s0:_s0 + _chunk] = (obs[_idx] == obs[_s0:_s0 + _chunk, None]).mean(axis=1)
    red[_s0:_s0 + _chunk] = 1.0 - _sims[_rows, _idx].mean(axis=1)
rng3 = np.random.default_rng(SEED)
ref = rng3.permutation(np.where(tag == "high")[0])[:400]
ref_clf = LogisticRegression(max_iter=200, C=1.0).fit(Xp[ref], obs[ref])
proba = ref_clf.predict_proba(Xp); cls = {c:k for k,c in enumerate(ref_clf.classes_)}
infl = np.array([np.log(proba[i, cls[obs[i]]] + 1e-9) if obs[i] in cls else -20.0 for i in range(n)])
k = n // 2
H = set(np.where(tag != "high")[0])          # 有害 = 全部注入噪声(flip/dup/hard)
Hf = set(np.where((tag=="flip")|(tag=="dup"))[0])   # 严格有害 = 翻转+近重复

def gain(sel):
    c = LogisticRegression(max_iter=150, C=1.0).fit(Xp[sel], obs[sel])
    return float((c.predict(Xval) == val_lab).mean())

from mmdataselect.selectors.external_baselines import dmf_dynamic
from mmdataselect.signals import minmax
cands = {
  "auth_only": list(np.argsort(-auth)[:k]),
  "dmf": dmf_dynamic(np.stack([minmax(auth),minmax(infl),minmax(red)]), k, val_reward=gain, seed=SEED),
  "influence_only": list(np.argsort(-infl)[:k]),
}
g_rand = gain(list(np.random.default_rng(SEED+1).permutation(n)[:k]))
gs = {m: gain(s) for m,s in cands.items()}
w = {m: max(gs[m]-g_rand, 0)+1e-9 for m in cands}
print(f"验证增益: rand={g_rand:.3f} " + " ".join(f"{m}={gs[m]:.3f}(w={w[m]:.3f})" for m in cands))

# 条件(iii) 权重均衡
ws = sorted(w.values())
print(f"条件(iii) w小两和 {ws[0]+ws[1]:.3f} vs w最大 {ws[-1]:.3f} -> {'满足' if ws[0]+ws[1]>ws[-1] else '不满足'}")

# 有害误选集及其重叠(条件 i)
Hsel = {m: set(s) & Hf for m,s in cands.items()}
ms = list(cands)
print("\n各候选严格有害误选率:", {m: round(len(Hsel[m])/k,4) for m in ms})
for a in range(len(ms)):
    for b in range(a+1,len(ms)):
        i,j = ms[a],ms[b]
        inter = len(Hsel[i]&Hsel[j]); denom = max(1,min(len(Hsel[i]),len(Hsel[j])))
        print(f"  有害重叠 {i}∩{j}: {inter} (占较小集 {inter/denom:.2%})")
all_h = Hsel[ms[0]]|Hsel[ms[1]]|Hsel[ms[2]]
ge2_h = {x for x in all_h if sum(x in Hsel[m] for m in ms) >= 2}
print(f"被>=2候选共同误选的有害记录: {len(ge2_h)}/{len(all_h)} ({(len(ge2_h)/max(1,len(all_h))):.2%}) -> 条件(i)近似度 γ")

# 共同核(条件 ii)与投票选集
votes = np.zeros(n)
for m,s in cands.items(): votes[list(s)] += w[m]
C = {x for x in range(n) if sum(x in set(cands[m]) for m in ms) >= 2}
print(f"共同核 |C|={len(C)} vs k={k} -> 条件(ii) {'满足' if len(C)>=k else '不满足'}")
V = list(np.argsort(-(votes + 1e-9*minmax(auth)))[:k])
hv = len(set(V)&Hf)/k
print(f"\n投票合成有害率 {hv:.4f} vs 最优单一 {min(len(Hsel[m])/k for m in ms):.4f} "
      f"-> {'投票更干净✓' if hv <= min(len(Hsel[m])/k for m in ms) else '投票未更干净'}")
print(f"投票合成验证增益 {gain(V):.3f} vs 成员最优 {max(gs.values()):.3f}")
