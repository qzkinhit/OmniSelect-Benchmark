import json
import re
from collections import Counter

CP = "/root/autodl-tmp/OmniSelect/experiments/controller_reval_v5/checkpoint.json"
REP = "/root/autodl-tmp/OmniSelect/experiments/controller_reval_report_v5.json"
LOG = "/root/controller_reval_v5_full.log"

cp = json.load(open(CP))
rep = json.load(open(REP))
reps = rep["replays"]
inval = rep.get("invalid_replays") or []

cp_keys = set(cp["entries"].keys())
rep_keys = set(f"{r['config']}::{r['seed']}" for r in reps)
# checkpoint key format probe
sample_cp = list(cp["entries"].keys())[0]
print("checkpoint key sample:", sample_cp)
print("report key sample:", list(rep_keys)[0])
print(f"checkpoint entries={len(cp_keys)} report replays={len(reps)} invalid_replays={len(inval)}")
print("invalid_replays detail:", json.dumps(inval, indent=1)[:600])

# normalize checkpoint keys to config::seed if needed
def norm(k):
    # try to extract config and seed
    return k
cpn = set(norm(k) for k in cp_keys)
only_cp = cpn - rep_keys
only_rep = rep_keys - cpn
print("in checkpoint not report:", list(only_cp)[:6])
print("in report not checkpoint:", list(only_rep)[:6])

# log [trial] lines: which (config,seed)
txt = open(LOG, errors="ignore").read()
trials = re.findall(r"^\[trial\]\s+(\S+)\s+seed(\d+)", txt, re.M)
tset = set(f"{c}::{s}" for c, s in trials)
print(f"\nlog [trial] lines={len(trials)} unique={len(tset)}")
print("in report not in log [trial]:", list(rep_keys - tset)[:6])
resumes = re.findall(r"^\[resume\]\s+(\S+)\s+seed(\d+)", txt, re.M)
print("resume markers:", resumes)
