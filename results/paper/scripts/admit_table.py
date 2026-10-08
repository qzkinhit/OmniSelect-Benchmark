"""Share of each corruption kind that a strategy admits into its selection, averaged over seeds.

For every corrupted task and seed: count, per pool tag, the records of that tag inside the selection of a
strategy and divide by the number of pool records with the tag. Strategies: Auth-only (auth_only), Coverage
(coreset), k-center (kcenter), Infl-only (influence_only), Herding (herding), the elected strategy of the
batch, and, for batches that have them, the best cooperative candidate by construction-split utility
(the screening winner, never chosen by test). Writes JSON to summary/admit_shares_<batch>.json.
"""
import glob, json, os, sys, statistics as st, collections
import numpy as np
R = sys.argv[2] if len(sys.argv) > 2 else "results_and_logs"; batch = sys.argv[1] if len(sys.argv) > 1 else "main"
STRATS = {"auth_only": "Auth-only", "coreset": "Coverage", "kcenter": "k-center", "influence_only": "Infl-only", "herding": "Herding"}
out = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
for f in glob.glob(f"{R}/{batch}/*/*/*/seed_*/decision.json"):
    c = os.path.dirname(f); task = c.split("/")[-3]
    sp = json.load(open(c + "/splits.json")); tags = np.asarray(sp["pool_tags"]); kinds = sorted(set(tags.tolist()))
    if kinds == ["high"]: continue
    d = json.load(open(f)); lb = json.load(open(c + "/leaderboard.json")); rows = {r["name"]: r for r in (lb if isinstance(lb, list) else lb["rows"])}
    picks = dict(STRATS); picks[d["elected"]] = "elected"
    coop = [r for n, r in rows.items() if (n.startswith("coop") or n == "clean_top") and r.get("u_con") is not None]
    coop_best = max(coop, key=lambda r: r["u_con"])["name"] if coop else None
    if coop_best: picks[coop_best] = "coop (best on V_con)"
    for cdir in glob.glob(c + "/candidates/*"):
        sc = json.load(open(cdir + "/scores.json")); name = sc["name"]
        labels = [lab for key, lab in picks.items() if key == name]
        if not labels: continue
        idx = np.load(cdir + "/selection.npz")["idx"]; sel_tags = tags[idx]
        for lab in labels:
            for k in kinds:
                out[task][lab][k].append(float((sel_tags == k).sum()) / max(int((tags == k).sum()), 1))
res = {t: {lab: {k: round(st.mean(v), 4) for k, v in kd.items()} for lab, kd in s.items()} for t, s in out.items()}
json.dump(res, open(f"admission_shares_{batch}.json", "w"), indent=1)
for t in sorted(res):
    print("==", t)
    for lab, kd in res[t].items():
        print("   %-22s %s" % (lab, "  ".join("%s %.3f" % (k, v) for k, v in sorted(kd.items()))))
