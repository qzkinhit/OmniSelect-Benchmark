"""Fail-closed selection-manifest replay validator, v2
(CODEX_AUDIT_20260717_0110_SELECTION_FAILCLOSED_AND_SCOPE).

Scope: the pubcore MAIN-TABLE four arms only (CIFAR-100 / ETTh1 / TEP21 / electricity,
3 seeds). The success marker is therefore PUBCORE_4ARM_SELECTION_MANIFESTS_VERIFIED_OK -
this validator makes NO claim about text / CIFAR-100N / CIFAR-10-original / IN100 /
ETTm1 / ETTh2 / DaISy / Chronos / TEP-calib2 families.

Fail-closed rules (any violation -> exit 1, no marker):
  1. original missing, replay missing, expected method missing, or method-set mismatch
     all hard-fail; expected methods are read MECHANICALLY from the original JSON.
  2. exactly ONE original and ONE replay file per (arm, seed) - multiple candidates are
     an error, never silently picked.
  3. hashes are RECOMPUTED from the full selected_ids via pairing.sel_sha12/order_sha12,
     never trusted from self-reports; recomputed == replay-reported == original-reported.
  4. selected_ids: list of ints, len == n_selected, all in [0, pool), duplicate-free
     (no-replacement protocol; any future with-replacement method must declare it).
  5. vision/TEP/tabular: training_order must equal sorted selected ids and its recomputed
     order_sha12 must match both reports -> REPLAY_VERIFIED. timeseries: selection hash
     only -> REPLAY_VERIFIED_SELECTION_ONLY (epoch permutation NOT recovered; original
     train_order_sha12 stays HASH_ONLY_NOT_REPLAYED).
  6. metadata must match exactly: dataset, seed, config, pairing_manifest pool/validation/
     test SHA, baseline_impl_sha256, fidelity_mode, published_core_protocol. The replay's
     instrumented code_sha256_12 is RECORDED alongside the original's, never claimed equal.
  7. verdicts record manifest SHA256, original SHA256, method and ID counts. Extra replay
     methods (e.g. tabaicl transparency rows) go to a supplement list and never count
     toward the pubcore gate.
"""
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from mmdataselect.utils.pairing import order_sha12, sel_sha12  # noqa: E402

ARMS = {"vision": True, "tep": True, "tabular": True, "timeseries": False}
POOL_OF = {"vision": 4000, "tep": 4000, "tabular": 3000, "timeseries": 3000}
META_KEYS = ("dataset", "seed", "config", "baseline_impl_sha256", "fidelity_mode",
             "published_core_protocol")

fails, verdicts, supplements = [], [], []


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def one(pattern, what, arm, s):
    hits = glob.glob(pattern)
    if len(hits) != 1:
        fails.append(f"{arm} seed{s}: {what} file count {len(hits)} != 1 ({pattern})")
        return None
    return hits[0]


def main():
    for arm, order_replayable in ARMS.items():
        pool_n = POOL_OF[arm]
        for s in ("0", "1", "2"):
            op = one(os.path.join(ROOT, "outputs", arm, "*",
                                  "run_id=pubcore-paired-*", f"seed_{s}", "results.json"),
                     "original", arm, s)
            rp = one(os.path.join(ROOT, "outputs", arm, "*",
                                  "run_id=selreplay-*", f"seed_{s}", "results.json"),
                     "replay", arm, s)
            if not op or not rp:
                continue
            O, R = json.load(open(op)), json.load(open(rp))
            o_rows = {r["method"]: r for r in O["results"]}
            r_rows = {r["method"]: r for r in R["results"]}
            # 6. metadata parity (recorded code SHAs kept separate, never compared equal)
            for k in META_KEYS:
                if O.get(k) != R.get(k):
                    fails.append(f"{arm} seed{s}: metadata mismatch on {k}")
            om, rm = O.get("pairing_manifest") or {}, R.get("pairing_manifest") or {}
            for k in ("pool_sha256", "validation_sha256", "test_sha256"):
                if om.get(k) != rm.get(k) or not om.get(k):
                    fails.append(f"{arm} seed{s}: pairing_manifest mismatch/missing {k}")
            # 1. expected method set from the original, mechanically
            expected = set(o_rows)
            missing = expected - set(r_rows)
            if missing:
                fails.append(f"{arm} seed{s}: replay missing methods {sorted(missing)}")
            for extra in sorted(set(r_rows) - expected):
                supplements.append({"arm": arm, "seed": s, "method": extra,
                                    "note": "transparency supplement, not in pubcore gate"})
            n_ver = 0
            for m in sorted(expected & set(r_rows)):
                o, r = o_rows[m], r_rows[m]
                ids = r.get("selected_ids")
                ok, why = True, []
                if not isinstance(ids, list) or not all(isinstance(i, int) for i in (ids or [])):
                    ok, why = False, why + ["selected_ids not an int list"]
                else:
                    if len(ids) != r.get("n_selected"):
                        ok, why = False, why + ["len != n_selected"]
                    if len(set(ids)) != len(ids):
                        ok, why = False, why + ["duplicates under no-replacement protocol"]
                    if ids and (min(ids) < 0 or max(ids) >= pool_n):
                        ok, why = False, why + ["id out of pool range"]
                    rec_sel = sel_sha12(ids)
                    if not (rec_sel == r.get("sel_sha12") == o.get("sel_sha12")):
                        ok, why = False, why + ["recomputed sel_sha12 mismatch"]
                    if order_replayable:
                        to = r.get("training_order")
                        if to != sorted(int(i) for i in ids):
                            ok, why = False, why + ["training_order != sorted ids"]
                        else:
                            rec_ord = order_sha12(to)
                            if not (rec_ord == r.get("train_order_sha12") == o.get("train_order_sha12")):
                                ok, why = False, why + ["recomputed order_sha12 mismatch"]
                if ok:
                    status = "REPLAY_VERIFIED" if order_replayable else "REPLAY_VERIFIED_SELECTION_ONLY"
                    n_ver += 1
                else:
                    status = "HASH_ONLY_NOT_REPLAYED"
                    fails.append(f"{arm} seed{s} {m}: {'; '.join(why)}")
                verdicts.append({"arm": arm, "seed": s, "method": m, "status": status,
                                 "n_ids": len(ids) if isinstance(ids, list) else None,
                                 "reasons": why or None})
            verdicts.append({"arm": arm, "seed": s, "level": "file",
                             "manifest_sha256": fsha(rp), "original_sha256": fsha(op),
                             "replay_code_sha256_12": R.get("code_sha256_12"),
                             "original_code_sha256_12": O.get("code_sha256_12"),
                             "n_methods_expected": len(expected), "n_verified": n_ver})
    out = {"scope": "pubcore main-table 4 arms x 3 seeds ONLY",
           "verdicts": verdicts, "supplements": supplements,
           "fails": fails,
           "summary": {"verified_cells": sum(1 for v in verdicts if str(v.get("status", "")).startswith("REPLAY_VERIFIED")),
                       "fail_count": len(fails)}}
    json.dump(out, open(os.path.join(ROOT, "experiments", "selection_manifest_verdicts.json"), "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))
    if fails:
        print("FAIL-CLOSED (%d):" % len(fails))
        for f in fails[:20]:
            print("  -", f)
        sys.exit(1)
    print("PUBCORE 4-ARM SELECTION MANIFESTS VERIFIED (recomputed from full ID lists)")


if __name__ == "__main__":
    main()
