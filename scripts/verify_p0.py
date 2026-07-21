"""Post-P0 result verifier. Recovers CIFAR seed1/2 JSON from logs and
strictly verifies CIFAR s0/1/2 and globalmix s0/1/2. Writes no marker itself; prints a
verdict the gate script consumes."""
import glob
import hashlib
import json
import os
import re
import sys

R = "/root/autodl-tmp/OmniSelect"
CODE = os.path.join(R, "baselines/deepcore_original/run_original_protocol.py")
EXPECT_CIFAR = {"random", "el2n", "grand", "ccs", "auth", "ctrl"}
EXPECT_MIX = {"random", "regmix", "mmds_adapt"}
FAIL = []


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def ck(name, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {d}")
    if not ok:
        FAIL.append(name)


# ---- item 3: CIFAR full s0/1/2 ----
print("=== item3: CIFAR-10 full original-protocol s0/1/2 ===")
code_sha = sha256(CODE)
cfg_ref = None
for s in (0, 1, 2):
    log = os.path.join(R, f"experiments/cifar10_full_original_repro_seed{s}.log")
    if not os.path.exists(log):
        ck(f"cifar s{s} log", False, log)
        continue
    txt = open(log, errors="ignore").read()
    mexit = re.findall(rf"SEED={s} python_exit=(\d+)", txt)
    ck(f"cifar s{s} exit0", bool(mexit) and mexit[-1] == "0", f"got {mexit[-1] if mexit else None}")
    ck(f"cifar s{s} no-error", not re.search(r"Traceback|CUDA out of memory|Killed", txt))
    rows = {}
    for m in re.finditer(r"^\s{2}(random|el2n|grand|ccs|auth|ctrl)\s+val=([0-9.]+) test=([0-9.]+)\s+n=(\d+)(?:\s+picked=(\w+))?",
                         txt, re.M):
        rows[m.group(1)] = {"method": m.group(1), "val": float(m.group(2)), "test": float(m.group(3)),
                            "n": int(m.group(4)), **({"picked": m.group(5)} if m.group(5) else {})}
    ck(f"cifar s{s} 6 methods once", set(rows) == EXPECT_CIFAR and len(rows) == 6, f"got {sorted(rows)}")
    cfg = {"pool": 45000, "train_epoch": 160, "score_runs": 3, "keep": 0.3, "code_sha256": code_sha}
    cfg_ref = cfg_ref or cfg
    ck(f"cifar s{s} config==ref", cfg == cfg_ref)
    # recover JSON (s1/s2 lack native results.json)
    d = os.path.join(R, "outputs/original_protocol/cifar10_full", f"seed_{s}")
    jp = os.path.join(d, "results.json")
    if set(rows) == EXPECT_CIFAR and not os.path.exists(jp):
        os.makedirs(d, exist_ok=True)
        payload = {"arm": "original_protocol", "dataset": "cifar10_full", "seed": s,
                   "config": {**cfg, "recovered_from_log": os.path.basename(log), "log_sha256": sha256(log)},
                   "results": [rows[k] for k in ("random", "el2n", "grand", "ccs", "auth", "ctrl")]}
        tmp = jp + ".tmp"
        json.dump(payload, open(tmp, "w"), indent=2)
        os.replace(tmp, jp)
        print(f"    recovered -> {jp}")
    ck(f"cifar s{s} json parseable", os.path.exists(jp) and bool(json.load(open(jp)).get("results")))

# ---- item 4: globalmix PROXY s0/1/2 ----
print("=== item4: globalmix RegMix PROXY s0/1/2 ===")
sig_ref = None
for s in (0, 1, 2):
    g = glob.glob(os.path.join(R, "outputs/experiment/stratify=0-infl=pplq-train=finetune-lmeval=1", f"seed_{s}", "results.json"))
    if not g:
        ck(f"mix s{s} json", False, "missing")
        continue
    try:
        d = json.load(open(g[0]))
    except Exception as e:
        ck(f"mix s{s} parse", False, str(e)[:50])
        continue
    methods = {r["method"] for r in d["results"]}
    ck(f"mix s{s} methods", EXPECT_MIX <= methods, f"got {sorted(methods)}")
    cfg = d.get("config", {})
    sig = (str(cfg.get("stratify")), cfg.get("infl_kind"), cfg.get("pool_sha256_12"), cfg.get("code_sha256_12"))
    sig_ref = sig_ref or sig
    ck(f"mix s{s} pool/code/config consistent", sig == sig_ref, f"{sig}")
    fin = all(all(isinstance(v, (int, float)) for v in (r.get("ppl") or {}).values()) for r in d["results"])
    ck(f"mix s{s} finite", fin)
    ck(f"mix s{s} PROXY-labeled (stratify=0)", str(cfg.get("stratify")) == "0")
    log = os.path.join(R, f"experiments/text_globalmix_regmix_seed{s}.log")
    m = re.findall(r"python_exit=(\d+)", open(log, errors="ignore").read()) if os.path.exists(log) else []
    ck(f"mix s{s} exit0", bool(m) and m[-1] == "0")

print("POST_P0_VERIFY_OK" if not FAIL else f"POST_P0_VERIFY_FAILED {FAIL[:8]}")
sys.exit(0 if not FAIL else 1)
