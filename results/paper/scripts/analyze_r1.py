"""Recompute the main_nc v2 main-table statistics from the copied cell records (decision.json,
leaderboard.json, metrics.json, timings.json). Output: r1_stats.json with provenance."""
import json, glob, math, os, sys
from collections import defaultdict
from scipy import stats
import numpy as np

L = sys.argv[1]
OUT = sys.argv[2]
TASKS = [("vision","cifar100","clip_vitb32","C-100"),("vision","cifar100n","clip_vitb32","C-100N"),
         ("native","cifar10","resnet18_scratch","C-10"),("native","imagenet100","resnet18_scratch","IN-100"),
         ("timeseries","ETTh1","dlinear","ETTh1"),("timeseries","ETTm1","dlinear","ETTm1"),("timeseries","ETTh2","dlinear","ETTh2"),
         ("timeseries","daisy_cstr","dlinear","CSTR"),("timeseries","daisy_steamgen","dlinear","Steam"),
         ("process","tep21","mlp","TEP21"),("tabular","electricity","tabpfn","Elec"),("text","five_domain","smollm2_135m","Text"),
         ("vision","cifar10_clip","clip_vitb32","C-10-CLIP")]
DISPLAYED = ["random","influence_only","coreset","mmdataselect","auth_only","herding","el2n","grand","ccs","density","quadmix_pub","dmf_pub"]
LOWER = {"neg_mase","neg_gmean_ppl"}

def nat(u, v): return -v if u in LOWER else v

def tint(d):
    n=len(d); m=float(np.mean(d))
    if n<2: return m, float('nan'), float('nan')
    sd=float(np.std(d,ddof=1)); h=stats.t.ppf(0.975,n-1)*sd/math.sqrt(n)
    return m, m-h, m+h

res = {"source_dir": L, "tasks": {}}
for track, ds, learner, label in TASKS:
    cells = sorted(glob.glob(f"{L}/{track}/{ds}/{learner}/seed_*/decision.json"))
    if not cells:
        res["tasks"][label] = {"n_seeds": 0}; continue
    per = defaultdict(dict)     # method -> seed -> stored utility (higher better)
    lbtest = defaultdict(dict)  # every leaderboard candidate -> seed -> u_test
    info = {}
    utility = None
    for c in cells:
        d = os.path.dirname(c)
        seed = int(d.rsplit("_",1)[1])
        dec = json.load(open(c)); met = json.load(open(d+"/metrics.json")); lb = json.load(open(d+"/leaderboard.json"))
        tim = json.load(open(d+"/timings.json")); cfg = json.load(open(d+"/config.json")); spl = json.load(open(d+"/splits.json"))
        utility = met["utility"]
        ALIAS = {"coverage_text":"coreset","fixed_fusion":"mmdataselect","herding_text":"herding","density_text":"density"} if track=="text" else {}
        for name,row in met["rows"].items():
            if row.get("u_test") is not None: per[ALIAS.get(name,name)][seed] = row["u_test"]
        for r in lb["rows"]:
            nm = ALIAS.get(r["name"], r["name"])
            if nm in DISPLAYED and nm not in per.get(nm, {}) and r.get("u_test") is not None and seed not in per[nm]:
                per[nm][seed] = r["u_test"]
        ranked = [r for r in lb["rows"] if r["role"]!="screened_out"]
        for r in lb["rows"]:
            if r.get("u_test") is not None: lbtest[r["name"]][seed] = r["u_test"]
        info[seed] = {
            "elected": dec["elected"], "reference": dec["reference"], "adopted": dec["adopted"],
            "elected_test": dec.get("elected_test_utility"),
            "gate": dec["gate"]["kind"], "k_tested": dec["gate"].get("k_tested"),
            "m_nominal": len(ranked), "m_distinct": len(set(r["sel_sha12"] for r in ranked)),
            "n_ref_eligible": sum(1 for r in ranked if r["role"]=="reference"),
            "n_challenger": sum(1 for r in ranked if r["role"]=="challenger"),
            "grid_total": (dec.get("screening") or {}).get("total"), "finalists": (dec.get("screening") or {}).get("finalists"),
            "screen_evals": (dec.get("screening") or {}).get("evaluations"),
            "counts": spl["counts"],
            "fde_total": tim["fde"]["total"], "fde_layers": tim["fde"]["layers"], "fde_by_stage": tim["fde"]["by_stage"],
            "fits": tim["cache"]["fits"], "cache_hits": tim["cache"]["cache_hits"], "distinct_subsets": tim["cache"]["distinct_subsets"],
            "secs_total": tim["stages"]["total"], "secs_controller": tim["stages"].get("controller"), "secs_member": tim["stages"].get("member_selection"),
            "full_fit_secs": (tim.get("candidates",{}).get("full") or {}).get("fit_secs"),
            "git": cfg["git"]["sha"],
            "precheck": dec.get("precheck"),
        }
    seeds = sorted(info)
    T = {"n_seeds": len(seeds), "seeds": seeds, "utility": utility, "per_seed": {s: info[s] for s in seeds}}
    # table rows
    rows = {}
    for name in DISPLAYED + ["mmds_adapt","full","kcenter","semdedup"]:
        if name not in per: continue
        vals = [nat(utility, per[name][s]) for s in seeds if s in per[name]]
        rows[name] = {"n": len(vals), "mean": float(np.mean(vals)), "sd": float(np.std(vals,ddof=1)) if len(vals)>1 else None,
                      "per_seed": {s: nat(utility, per[name][s]) for s in seeds if s in per[name]}}
    T["rows"] = rows
    sign = -1 if utility in LOWER else 1
    disp = [n for n in DISPLAYED if n in rows]
    best = max(disp, key=lambda n: sign*rows[n]["mean"])
    T["strongest_displayed"] = best
    # paired verdicts vs every displayed baseline
    verd = {}
    for n in disp:
        common = [s for s in seeds if s in per[n] and s in per["mmds_adapt"]]
        diffs = [per["mmds_adapt"][s]-per[n][s] for s in common]   # stored utility, higher better
        m,lo,hi = tint(diffs)
        v = "n<2" if len(common)<2 else ("W" if lo>0 else ("L" if hi<0 else "T"))
        verd[n] = {"n": len(common), "mean": m, "lo": lo, "hi": hi, "verdict": v}
    T["verdicts"] = verd
    # below random pairs
    T["below_random_seeds"] = [s for s in seeds if per["mmds_adapt"][s] < per["random"][s]]
    T["equal_random_seeds"] = [s for s in seeds if per["mmds_adapt"][s] == per["random"][s]]
    # gains (direction-normalized, seed means)
    rmean = rows["random"]["mean"]
    def gain(n):
        mm = rows[n]["mean"]; return (sign*(mm - rmean))/abs(rmean)
    T["gain_over_random"] = {n: gain(n) for n in rows}
    # per-seed gain for omni
    T["per_seed_gain_omni"] = {s: sign*(nat(utility,per['mmds_adapt'][s]) - nat(utility,per['random'][s]))/abs(nat(utility,per['random'][s])) for s in seeds}
    # displayed strategies below random on this task (seed means)
    T["displayed_below_random"] = [n for n in disp if n!="random" and sign*(rows[n]["mean"]-rmean) < 0]
    # portfolio oracle: best leaderboard candidate by test mean over seeds where present in all seeds
    allc = {n: v for n, v in lbtest.items() if all(s in v for s in seeds)}
    port_best = max(allc, key=lambda n: np.mean([allc[n][s] for s in seeds]))
    T["portfolio_best_mean"] = {"name": port_best, "mean": float(nat(utility, np.mean([allc[port_best][s] for s in seeds])))}
    # per-seed oracle over ranked candidates (max u_test per seed)
    T["per_seed_oracle_mean"] = float(nat(utility, np.mean([max(v[s] for v in lbtest.values() if s in v) for s in seeds])))
    # undisplayed members: best by test mean
    und = {n: v for n, v in allc.items() if n not in DISPLAYED and n not in ("mmds_adapt","full") and not n.startswith("fuse") and not n.startswith("vote") }
    T["undisplayed_members"] = {n: float(nat(utility, np.mean([und[n][s] for s in seeds]))) for n in und}
    # full vs omni
    if "full" in per:
        common=[s for s in seeds if s in per["full"]]
        diffs=[per["mmds_adapt"][s]-per["full"][s] for s in common]
        m,lo,hi=tint(diffs); T["vs_full"]={"n":len(common),"mean":m,"lo":lo,"hi":hi}
    T["n_distinct_elected"] = len(set(info[s]["elected"] for s in seeds))
    T["adoptions"] = sum(1 for s in seeds if info[s]["adopted"])
    res["tasks"][label] = T

json.dump(res, open(OUT,"w"), indent=1, default=str)
print("wrote", OUT)
