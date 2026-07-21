"""Tab-AICL FAITHFUL protocol: iterative cold-start active in-context learning for TabPFN,
reporting AULC (area under the accuracy-vs-context-size curve), as in Ma et al. 2026.

This is the "did we reproduce it right" check on the ORIGINAL protocol and ORIGINAL datasets
(OpenML: ionosphere is the paper's headline). Published finding to recover: the active rules
(margin / hybrid) beat random-context on most datasets *when re-conditioning each round* (the
single-shot version in the cross-modal bench does NOT re-condition, which is why it can lose).

Protocol: start 1 labelled example/class; each round fit TabPFN on the current context,
score the pool, acquire B by the rule, add to context; record test accuracy at each context
size; AULC = mean over the trajectory up to N_max labels. Env: DS (ionosphere), NMAX (100),
B (10), SEEDS (0 1 2 3 4).
"""
import os, numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

DS = os.environ.get("DS", "ionosphere")
NMAX = int(os.environ.get("NMAX", "100"))
B = int(os.environ.get("B", "10"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").split()]

def load():
    d = fetch_openml(DS, version=1, as_frame=True)
    X = d.data.select_dtypes(include=[np.number]).fillna(0.0).to_numpy(float)
    y = LabelEncoder().fit_transform(d.target.astype(str).to_numpy())
    return StandardScaler().fit_transform(X), y


def kcenter_init(X, pool, k, seed):
    rng = np.random.default_rng(seed); chosen=[int(rng.choice(pool))]
    dist=np.linalg.norm(X[pool]-X[chosen[0]][None],axis=1)
    while len(chosen)<k and len(chosen)<len(pool):
        i=int(np.argmax(dist)); c=pool[i]; chosen.append(int(c))
        dist=np.minimum(dist,np.linalg.norm(X[pool]-X[c][None],axis=1))
    return chosen[:k]

def acquire(rule, clf, X, ctx, pool, b, seed):
    if rule=="random":
        return list(np.random.default_rng(seed).permutation(pool)[:b])
    if rule=="coreset":
        return kcenter_init(X, np.array(pool), b, seed)
    p = clf.predict_proba(X[pool]); s=np.sort(p,axis=1)
    margin = s[:,-1]-(s[:,-2] if p.shape[1]>1 else 0.0)
    if rule=="margin":
        return [int(pool[i]) for i in np.argsort(margin)[:b]]
    if rule=="hybrid":
        mm=[int(pool[i]) for i in np.argsort(margin)[:b//2]]
        rest=[c for c in pool if c not in set(mm)]
        cc=kcenter_init(X, np.array(rest), b-len(mm), seed)
        return mm+cc

def run_rule(rule, X, y, tr, te, seed):
    from tabpfn import TabPFNClassifier
    def _mk():
        return TabPFNClassifier.create_default_for_version("v2", device="cpu", ignore_pretraining_limits=True)
    rng=np.random.default_rng(seed)
    classes=np.unique(y[tr]); ctx=[]
    for c in classes:                       # 1 per class start
        ctx.append(int(rng.choice(tr[y[tr]==c])))
    pool=[int(i) for i in tr if i not in set(ctx)]
    curve=[]
    while True:
        clf=_mk()
        clf.fit(X[ctx], y[ctx])
        acc=float((clf.predict(X[te])==y[te]).mean()); curve.append((len(ctx),acc))
        if len(ctx)>=NMAX or not pool: break
        add=acquire(rule, clf, X, ctx, pool, min(B,len(pool),NMAX-len(ctx)), seed+len(ctx))
        add=[a for a in add if a in set(pool)][:min(B,len(pool))]
        if not add: add=[pool[0]]
        ctx+=add; pool=[p for p in pool if p not in set(add)]
    accs=[a for _,a in curve]
    return float(np.mean(accs)), curve

def main():
    X,y=load()
    print(f"[{DS}] n={len(y)} d={X.shape[1]} classes={len(np.unique(y))} | AULC over context 1/class..{NMAX}, B={B}")
    rules=["random","margin","coreset","hybrid"]
    agg={r:[] for r in rules}
    for seed in SEEDS:
        tr,te=train_test_split(np.arange(len(y)),test_size=0.4,random_state=seed,stratify=y)
        for r in rules:
            aulc,_=run_rule(r,X,y,tr,te,seed); agg[r].append(aulc)
    print(f"\n==== Tab-AICL ITERATIVE AULC ({DS}, {len(SEEDS)} seeds) ====")
    base=np.mean(agg["random"])
    for r in rules:
        m=np.mean(agg[r]); print(f"  {r:8} AULC={m:.4f} ± {np.std(agg[r]):.4f}   ΔvsRandom={m-base:+.4f}")
if __name__=="__main__": main()
