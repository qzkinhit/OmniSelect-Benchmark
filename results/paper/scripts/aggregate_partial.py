"""Recompute cross-task summaries, preserving independent-run comparison limits.

Historical keys ending in mixed10/five9 are retained only as compatibility
aliases. The explicit scope records are authoritative for task counts.
"""
import json, math, sys
import numpy as np
from scipy import stats
r = json.load(open(sys.argv[1]))
T = r["tasks"]
MAIN = [t for t in ["C-100","C-100N","C-10","ETTh1","ETTm1","ETTh2","CSTR","Steam","TEP21","Elec","Text"] if t in T and "below_random_seeds" in T[t]]
print("aggregating tasks:", MAIN)
FIVE = [t for t in MAIN if T[t]["n_seeds"]>=2]
DISP = ["random","influence_only","coreset","mmdataselect","auth_only","herding","el2n","grand","ccs","density","quadmix_pub","dmf_pub"]
out = {"scope": {"tasks": MAIN, "n_tasks": len(MAIN), "seeds_per_task": {t:T[t]["n_seeds"] for t in MAIN}, "deprecated_key_suffixes": "mixed10/five9/five_seed denote legacy names, not the current counts"}}
# verdict counts over five-seed tasks
cnt = {"W":0,"T":0,"L":0,"N/A":0}; cells=0
per_base = {}
for t in FIVE:
    for b,v in T[t]["verdicts"].items():
        if b=="random" or True:
            cnt[v["verdict"]] += 1; cells += 1
            per_base.setdefault(b, {"W":0,"T":0,"L":0,"N/A":0}); per_base[b][v["verdict"]] += 1
out["verdicts_all_cells_five_seed"] = {"cells": cells, **cnt, "per_baseline": per_base}
sv = {t: (T[t]["strongest_displayed"], T[t]["verdicts"][T[t]["strongest_displayed"]]["verdict"]) for t in FIVE}
out["vs_strongest"] = sv
out["vs_strongest_counts"] = {k: sum(1 for t in FIVE if sv[t][1]==k) for k in ["W","T","L","N/A"]}
# below random
out["below_random_pairs"] = {"pairs": sum(T[t]["n_seeds"] for t in MAIN), "below": sum(len(T[t]["below_random_seeds"]) for t in MAIN), "equal": sum(len(T[t]["equal_random_seeds"]) for t in MAIN)}
# gains
g = {}
for b in DISP + ["mmds_adapt","full"]:
    ts = [t for t in MAIN if b in T[t]["gain_over_random"]]
    g[b] = {"tasks": ts, "mean_gain": float(np.mean([T[t]["gain_over_random"][b] for t in ts])),
            "omni_mean_gain_same_tasks": float(np.mean([T[t]["gain_over_random"]["mmds_adapt"] for t in ts]))}
out["gains_mixed10"] = g
g5 = {}
for b in DISP + ["mmds_adapt","full"]:
    ts = [t for t in FIVE if b in T[t]["gain_over_random"]]
    g5[b] = {"tasks": ts, "mean_gain": float(np.mean([T[t]["gain_over_random"][b] for t in ts])),
            "omni_mean_gain_same_tasks": float(np.mean([T[t]["gain_over_random"]["mmds_adapt"] for t in ts]))}
out["gains_five9"] = g5
# strongest displayed baseline by mean gain (over the tasks where it is defined on all 10)
alltask = [b for b in DISP if len(g[b]["tasks"])==len(MAIN) and b!="random"]
out["strongest_by_mean_gain_all10"] = max(alltask, key=lambda b: g[b]["mean_gain"])
# displayed oracle mean gain (best displayed per task by test mean)
out["displayed_oracle_gain_mixed10"] = float(np.mean([T[t]["gain_over_random"][T[t]["strongest_displayed"]] for t in MAIN]))
out["displayed_oracle_gain_five9"] = float(np.mean([T[t]["gain_over_random"][T[t]["strongest_displayed"]] for t in FIVE]))
# each displayed strategy below random on at least one task (seed means, MAIN)
below = {b: [t for t in MAIN if b in T[t]["rows"] and b in T[t]["displayed_below_random"]] for b in DISP if b!="random"}
out["displayed_below_random_tasks"] = below
# Friedman over MAIN with strategies instantiated on all
K = [b for b in DISP if all(b in T[t]["rows"] and not T[t]["rows"][b].get("adaptation", False) for t in MAIN)] + ["mmds_adapt"]
M = []
for t in MAIN:
    sign = -1 if T[t]["utility"] in ("neg_mase","neg_gmean_ppl") else 1
    vals = [sign*T[t]["rows"][b]["mean"] for b in K]
    M.append(vals)
M = np.array(M)
ranks = np.array([stats.rankdata(-row) for row in M])
avg = ranks.mean(0)
chi, p = stats.friedmanchisquare(*M.T)
k=len(K); N=len(MAIN)
qalpha = {10: 3.164}  # Nemenyi q_0.05 for k=10 (Demsar 2006 table)
cd = (stats.studentized_range.ppf(0.95,k,np.inf)/math.sqrt(2))*math.sqrt(k*(k+1)/(6*N))
out["friedman_mixed10"] = {"strategies": K, "avg_ranks": dict(zip(K, avg.round(3).tolist())), "chi2": float(chi), "p": float(p), "k": k, "N": N, "CD": cd}
out["friedman_mixed10"]["scope_note"] = "Original fully covered method set only. Adaptation rows are excluded; all-method ranking is descriptive in table_method_ranks.json."
# Wilcoxon one-sided (omni > baseline) on gains, Holm over baselines
wil = {}
for b in DISP:
    if b=="random":
        ts=[t for t in MAIN]; d=[T[t]["gain_over_random"]["mmds_adapt"] for t in ts]
    else:
        ts=[t for t in MAIN if b in T[t]["gain_over_random"] and T[t]["rows"][b].get("paired_inference_vs_original_omni_eligible", True)]
        d=[T[t]["gain_over_random"]["mmds_adapt"]-T[t]["gain_over_random"][b] for t in ts]
    d=[x for x in d]
    nz=[x for x in d if x!=0]
    res = stats.wilcoxon(nz, alternative="greater", method=stats.PermutationMethod(n_resamples=np.inf)) if len(nz)>0 else None
    wil[b] = {"tasks": ts, "n": len(d), "n_nonzero": len(nz), "pos": sum(x>0 for x in d), "neg": sum(x<0 for x in d), "zero": sum(x==0 for x in d), "p_one": float(res.pvalue) if res else None}
ps = sorted([(v["p_one"], b) for b,v in wil.items() if v["p_one"] is not None])
m=len(ps); holm={}; run=0
for i,(p,b) in enumerate(ps):
    adj=min(1.0,(m-i)*p); run=max(run,adj); holm[b]=run
for b in wil: wil[b]["holm"]=holm.get(b)
out["wilcoxon_mixed10"] = wil
# coverage table rows (seed 0) with r1
cov={}
for t in MAIN+["C-10-CLIP"]:
    if T[t]["n_seeds"]==0: continue
    s0 = T[t]["per_seed"][str(T[t]["seeds"][0])] if str(T[t]["seeds"][0]) in T[t]["per_seed"] else T[t]["per_seed"][T[t]["seeds"][0]]
    n = s0["counts"]["rank"]
    row={"n_rank": n, "n_con": s0["counts"]["con"], "n_conf": s0["counts"]["conf"], "m_nominal": s0["m_nominal"], "m_distinct": s0["m_distinct"],
         "grid_total_distinct": s0["grid_total"], "finalists": s0["finalists"]}
    ms = [T[t]["per_seed"][k]["m_nominal"] for k in T[t]["per_seed"]]
    md = [T[t]["per_seed"][k]["m_distinct"] for k in T[t]["per_seed"]]
    row["m_nominal_range"]=[min(ms),max(ms)]; row["m_distinct_range"]=[min(md),max(md)]
    neff = 542 if t=="Elec" else n
    row["n_eff"]=neff
    row["2r1_distinct"] = 2*math.sqrt(math.log(2*s0["m_distinct"]/0.05)/(2*neff))
    row["2r1_nominal"] = 2*math.sqrt(math.log(2*s0["m_nominal"]/0.05)/(2*neff))
    row["adoptions"] = T[t]["adoptions"]; row["seeds"]=T[t]["n_seeds"]
    cov[t]=row
out["coverage"]=cov
# FDE
fde={}
for t in MAIN+["C-10-CLIP"]:
    if T[t]["n_seeds"]==0: continue
    vals=[v["fde_total"] for v in T[t]["per_seed"].values()]
    lay={k: [v["fde_layers"][k] for v in T[t]["per_seed"].values()] for k in ("construction","ranking")}
    mem=[v["fde_by_stage"].get("member",0) for v in T[t]["per_seed"].values()]
    con=[v["fde_by_stage"].get("con",0) for v in T[t]["per_seed"].values()]
    fits=[v["fits"] for v in T[t]["per_seed"].values()]
    secs=[v["secs_total"] for v in T[t]["per_seed"].values()]
    ctrl=[v["secs_controller"] for v in T[t]["per_seed"].values()]
    full=[v["full_fit_secs"] for v in T[t]["per_seed"].values() if v["full_fit_secs"] is not None]
    fde[t]={"total":[min(vals),max(vals)], "member":[min(mem),max(mem)], "con":[min(con),max(con)], "ranking":[min(lay["ranking"]),max(lay["ranking"])], "fits":[min(fits),max(fits)], "secs_total":[min(secs),max(secs)], "secs_controller":[min(ctrl),max(ctrl)], "full_fit_secs":[min(full),max(full)] if full else None}
out["fde"]=fde
for old, new in {"verdicts_all_cells_five_seed":"verdicts_all_cells", "gains_mixed10":"gains", "strongest_by_mean_gain_all10":"strongest_by_mean_gain", "displayed_oracle_gain_mixed10":"displayed_oracle_gain", "friedman_mixed10":"friedman_original_methods", "wilcoxon_mixed10":"wilcoxon"}.items():
    out[new] = out[old]
out["inference_limits"] = "Per-task t intervals are descriptive unadjusted paired intervals. Cross-task Wilcoxon uses one-sided exact permutation with Holm over available baselines. Independent text adaptations are excluded from paired inference against historical OmniSelect."
json.dump(out, open(sys.argv[2],"w"), indent=1, default=str, allow_nan=False)
print(json.dumps({k:out[k] for k in ["verdicts_all_cells_five_seed","vs_strongest","vs_strongest_counts","below_random_pairs","strongest_by_mean_gain_all10","displayed_oracle_gain_mixed10","displayed_oracle_gain_five9","friedman_mixed10"]}, indent=0, default=str))
for b,v in out["gains_mixed10"].items(): print(b, len(v["tasks"]), round(v["mean_gain"]*100,2), round(v["omni_mean_gain_same_tasks"]*100,2), '| five9', round(out["gains_five9"][b]["mean_gain"]*100,2), round(out["gains_five9"][b]["omni_mean_gain_same_tasks"]*100,2))
for b,v in out["wilcoxon_mixed10"].items(): print('wil', b, v)
print(json.dumps(out["displayed_below_random_tasks"]))
for t,v in out["coverage"].items(): print(t, v)
for t,v in out["fde"].items(): print(t, v)
