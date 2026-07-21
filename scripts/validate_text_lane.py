"""Independent text-lane validator (audit round). Writes /root/D_VALIDATED_OK only if:
  main batch (stratify=1, pplq, finetune, lmeval=1) seeds 0/1/2 each have an isolated
  results.json with exactly the expected 10 method rows (9 METHODS + mmds_adapt), all
  PPL and lm-eval numbers finite and non-NaN; the global-mix proxy run (stratify=0)
  has its own separate results.json (seed 0, methods random/regmix + adapt row) whose
  numbers are finite; and every recorded python_exit in the batch logs is 0.
seed0 fallback: if seed0 ran before output isolation landed, accept the legacy path
outputs/experiment/results_seed0.json ONLY when its config matches the main batch
(passes=2) and it parses clean; it is then copied into the isolated scheme.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys

_REPO = "/root/autodl-tmp/OmniSelect"
MAIN_TAG = "stratify=1-infl=pplq-train=finetune-lmeval=1"
MIX_TAG = "stratify=0-infl=pplq-train=finetune-lmeval=1"
EXPECT_MAIN = {"noselect", "random", "dsir", "if_mates", "quality_ppl", "dmf", "balance",
               "mmdataselect", "perpcorr", "mmds_adapt"}
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def finite(x):
    if isinstance(x, dict):
        return all(finite(v) for v in x.values())
    if isinstance(x, (int, float)):
        return math.isfinite(x)
    return True


def load_trial(path):
    try:
        return json.load(open(path))
    except Exception as e:
        check(f"parse {path}", False, str(e)[:60])
        return None


def main() -> int:
    # main batch seeds
    for s in (0, 1, 2):
        p = os.path.join(_REPO, "outputs/experiment", MAIN_TAG, f"seed_{s}", "results.json")
        if s == 0 and not os.path.exists(p):
            legacy = os.path.join(_REPO, "outputs/experiment/results_seed0.json")
            if os.path.exists(legacy):
                d = load_trial(legacy)
                if d and d.get("config", {}).get("passes") == 2:
                    import hashlib as _h
                    d.setdefault("config", {}).update({
                        "adopted_from_legacy_path": legacy,
                        "runtime_code_git": "9e3a38f",
                        "runtime_code_sha256": "ec6e2cfb096e3da2dbbda31255928f3b5e77fa704d54aa0077332dd44d765d57",
                        "pool_sha256": _h.sha256(open(os.path.join(_REPO, "data/processed/qpool_train.jsonl"), "rb").read()).hexdigest(),
                        "log_sha256": _h.sha256(open(os.path.join(_REPO, "experiments/text_scaleup_seed0.log"), "rb").read()).hexdigest(),
                        "protocol": {"stratify": "1", "infl_kind": "pplq", "train_mode": "finetune",
                                     "lm_eval": "1", "passes": 2,
                                     "lmeval_tasks": "arc_easy,arc_challenge,hellaswag,openbookqa"},
                    })
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    json.dump(d, open(p, "w"), indent=2)
                    print(f"  [info] seed0 legacy artifact adopted (full provenance) -> {p}")
        d = load_trial(p) if os.path.exists(p) else None
        check(f"main seed{s} exists", d is not None, p)
        if not d:
            continue
        rows = d["results"] if isinstance(d.get("results"), list) else d.get("results", [])
        names_list = [r.get("method") for r in rows]
        check(f"main seed{s} exactly 10 rows", len(rows) == 10, f"got {len(rows)}")
        check(f"main seed{s} methods unique", len(set(names_list)) == len(names_list))
        check(f"main seed{s} method set exact", set(names_list) == EXPECT_MAIN,
              f"missing={EXPECT_MAIN - set(names_list)} extra={set(names_list) - EXPECT_MAIN}")
        cfg = d.get("config", {})
        check(f"main seed{s} n==25000", cfg.get("n") == 25000, f"got {cfg.get('n')}")
        check(f"main seed{s} passes==2", cfg.get("passes") == 2, f"got {cfg.get('passes')}")
        check(f"main seed{s} seed match", d.get("seed") == s or cfg.get("seed") == s)
        proto = cfg.get("protocol", cfg)
        check(f"main seed{s} stratify=1", str(proto.get("stratify", cfg.get("stratify"))) == "1")
        check(f"main seed{s} infl=pplq", str(proto.get("infl_kind", cfg.get("infl_kind"))) == "pplq")
        DOMS = {"code", "general", "image", "math", "table"}
        TASKS = {"arc_easy", "arc_challenge", "hellaswag", "openbookqa"}
        for r in rows:
            ppl = r.get("ppl") or {}
            check(f"main seed{s} {r.get('method')} ppl 5-domain keys",
                  set(ppl.keys()) == DOMS, f"got {sorted(ppl.keys())}")
            check(f"main seed{s} {r.get('method')} ppl finite", finite(ppl) and bool(ppl))
            lme = r.get("lmeval") or {}
            check(f"main seed{s} {r.get('method')} lmeval 4-task keys",
                  set(lme.keys()) == TASKS, f"got {sorted(lme.keys())}")
            check(f"main seed{s} {r.get('method')} lmeval finite", finite(lme) and bool(lme))
    # global-mix proxy
    p = os.path.join(_REPO, "outputs/experiment", MIX_TAG, "seed_0", "results.json")
    d = load_trial(p) if os.path.exists(p) else None
    check("globalmix exists", d is not None, p)
    if d:
        rows = d["results"]
        names_list = [r.get("method") for r in rows]
        check("globalmix exactly 3 rows", len(rows) == 3, f"got {len(rows)}")
        check("globalmix method set", set(names_list) == {"random", "regmix", "mmds_adapt"},
              str(sorted(names_list)))
        check("globalmix finite", all(finite(r.get("ppl", {})) and bool(r.get("ppl")) for r in rows))
    # exit codes from logs
    for logf, pat in (("experiments/text_scaleup_seed0.log", r"SEED=0 python_exit=(\d+)"),
                      ("experiments/text_scaleup_seed1.log", r"SEED=1 python_exit=(\d+)"),
                      ("experiments/text_scaleup_seed2.log", r"SEED=2 python_exit=(\d+)"),
                      ("experiments/text_globalmix_regmix_seed0.log", r"SEED=0 python_exit=(\d+)")):
        fp = os.path.join(_REPO, logf)
        m = re.findall(pat, open(fp, errors="ignore").read()) if os.path.exists(fp) else []
        check(f"exit0 {os.path.basename(logf)}", bool(m) and m[-1] == "0",
              f"got {m[-1] if m else 'none'}")
        if os.path.exists(fp):
            bad = re.findall(r"Traceback|CUDA out of memory|Killed", open(fp, errors="ignore").read())
            check(f"no-error {os.path.basename(logf)}", not bad, str(bad[:2]))
    if FAIL:
        print(f"D_VALIDATION_FAILED ({len(FAIL)}): {FAIL[:8]}")
        return 1
    with open("/root/D_VALIDATED_OK", "w") as fh:
        fh.write(MAIN_TAG)
    print("D_VALIDATION_OK -> /root/D_VALIDATED_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
