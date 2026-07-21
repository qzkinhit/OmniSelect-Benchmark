"""E-lane post-processor + validator (audit round). Waits for nothing itself - run it
after the three sub-runs finished. For each trial (IN100 seed0/1, CIFAR10-full seed0):
parse the log, build a structured JSON with exactly six rows (random/el2n/grand/ccs/
auth/ctrl), verify python_exit=0, values in [0,1], ctrl row consistent with its picked
candidate's numbers, and record full config + code/log sha256. Writes E_VALIDATED_OK
only when every trial passes."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

_REPO = "/root/autodl-tmp/OmniSelect"
EXPECT = ["random", "el2n", "grand", "ccs", "auth", "ctrl"]
FAIL = []

TRIALS = [
    ("experiments/imagenet100_protocol_seed0.log", "imagenet100", 0, r"SEED=0 python_exit=(\d+)",
     {"in_res": 112, "score_epoch": 10, "train_epoch": 40, "score_runs": 3, "pool": 120000,
      "val_n": 5000, "keep": 0.3, "scale_note": "reduced-scale qualitative validation"}),
    ("experiments/imagenet100_protocol_seed1.log", "imagenet100", 1, r"SEED=1 python_exit=(\d+)",
     {"in_res": 112, "score_epoch": 10, "train_epoch": 40, "score_runs": 3, "pool": 120000,
      "val_n": 5000, "keep": 0.3, "scale_note": "reduced-scale qualitative validation"}),
    ("experiments/cifar10_full_original_repro_seed0.log", "cifar10_full", 0, r"SEED=0 python_exit=(\d+)",
     {"score_epoch": 10, "train_epoch": 160, "score_runs": 3, "pool": 45000, "val_n": 5000,
      "keep": 0.3, "scale_note": "original-protocol reproduction (single seed)"}),
]


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    code = os.path.join(_REPO, "baselines/deepcore_original/run_original_protocol.py")
    for log, ds, seed, exit_pat, cfg in TRIALS:
        fp = os.path.join(_REPO, log)
        name = f"{ds} seed{seed}"
        if not os.path.exists(fp):
            check(f"{name} log exists", False, fp)
            continue
        txt = open(fp, errors="ignore").read()
        m = re.findall(exit_pat, txt)
        check(f"{name} exit0", bool(m) and m[-1] == "0", f"got {m[-1] if m else 'none'}")
        check(f"{name} no-error", not re.findall(r"Traceback|CUDA out of memory|Killed", txt))
        rows = {}
        for mm in re.finditer(r"^\s{2}(random|el2n|grand|ccs|auth|ctrl)\s+val=([0-9.]+) test=([0-9.]+)\s+n=(\d+)(?:\s+picked=(\w+))?",
                              txt, re.M):
            rows[mm.group(1)] = {"method": mm.group(1), "val": float(mm.group(2)),
                                 "test": float(mm.group(3)), "n": int(mm.group(4)),
                                 **({"picked": mm.group(5)} if mm.group(5) else {})}
        check(f"{name} exactly 6 rows", set(rows) == set(EXPECT),
              f"got {sorted(rows)}")
        if set(rows) == set(EXPECT):
            ok_range = all(0.0 <= r["val"] <= 1.0 and 0.0 <= r["test"] <= 1.0 for r in rows.values())
            check(f"{name} values in [0,1]", ok_range)
            picked = rows["ctrl"].get("picked")
            check(f"{name} ctrl picked recorded", bool(picked), str(picked))
            if picked in rows:
                same = (abs(rows["ctrl"]["val"] - rows[picked]["val"]) < 1e-9 and
                        abs(rows["ctrl"]["test"] - rows[picked]["test"]) < 1e-9)
                check(f"{name} ctrl==picked({picked}) numbers", same)
            d = os.path.join(_REPO, "outputs", "original_protocol", ds, f"seed_{seed}")
            os.makedirs(d, exist_ok=True)
            payload = {"arm": "original_protocol", "dataset": ds, "seed": seed,
                       "config": {**cfg, "recovered_from_log": log, "log_sha256": sha256(fp),
                                  "code_sha256": sha256(code)},
                       "results": [rows[k] for k in EXPECT]}
            tmp = os.path.join(d, ".tmp")
            json.dump(payload, open(tmp, "w"), indent=2)
            os.replace(tmp, os.path.join(d, "results.json"))
            print(f"  [info] artifact -> {d}/results.json")
    if FAIL:
        print(f"E_VALIDATION_FAILED ({len(FAIL)}): {FAIL[:8]}")
        return 1
    with open("/root/E_VALIDATED_OK", "w") as fh:
        fh.write("in100 seed0/1 + cifar10_full seed0")
    print("E_VALIDATION_OK -> /root/E_VALIDATED_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
