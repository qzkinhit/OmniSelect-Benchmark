import json

REP = "/root/autodl-tmp/OmniSelect/experiments/controller_reval_report_v5.json"
rep = json.load(open(REP))
reps = rep["replays"]
changed = [r for r in reps if not r.get("unchanged")]
print(f"total replays={len(reps)} changed={len(changed)}\n")
hdr = f"{'config':22} {'seed':>4} {'old_picked':30} {'new_picked':30} {'oV':>7} {'nV':>7} {'oT':>7} {'nT':>7} {'dT':>7} pm"
print(hdr)
for r in sorted(changed, key=lambda x: (x["config"], x["seed"])):
    def s(x, n=30):
        x = "" if x is None else str(x)
        return x[:n]
    print(f"{s(r['config'],22):22} {r['seed']:>4} {s(r['old_picked']):30} {s(r['new_picked']):30} "
          f"{str(r.get('old_val'))[:7]:>7} {str(r.get('new_val'))[:7]:>7} "
          f"{str(r.get('old_test'))[:7]:>7} {str(r.get('new_test'))[:7]:>7} "
          f"{str(r.get('delta_test'))[:7]:>7} {r.get('picked_match')}")

# 分类信号
print("\n--- 分类信号 ---")
cats = {"picked_diff_metric_same": [], "picked_same_metric_diff": [], "both_diff": [], "old_parse_none": []}
for r in changed:
    om, nm = r["old_picked"], r["new_picked"]
    dt = r.get("delta_test")
    if om is None or r.get("old_test") is None:
        cats["old_parse_none"].append((r["config"], r["seed"]))
    elif om != nm and (dt is not None and dt <= 0.005):
        cats["picked_diff_metric_same"].append((r["config"], r["seed"]))
    elif om == nm and (dt is not None and dt > 0.005):
        cats["picked_same_metric_diff"].append((r["config"], r["seed"]))
    else:
        cats["both_diff"].append((r["config"], r["seed"]))
for k, v in cats.items():
    print(f"  {k}: {len(v)} {v}")
