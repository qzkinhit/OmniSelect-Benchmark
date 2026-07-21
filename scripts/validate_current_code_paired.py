#!/usr/bin/env python3
"""Fail-closed validator for the current-code paired paper-main batch."""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
import re


COMMON = {
    "full", "random", "coreset", "auth_only", "influence_only",
    "mmdataselect", "herding", "kcenter", "semdedup", "density",
    "quadmix", "dmf", "mmds_adapt",
}
CLASSIFICATION = COMMON | {"el2n", "grand", "ccs"}
SPECS = {
    "vision": ("outputs/vision/uoft-cs_cifar100", CLASSIFICATION, "acc"),
    "tep": ("outputs/tep/tep21", CLASSIFICATION, "f1"),
    "tabular": ("outputs/tabular/electricity", CLASSIFICATION, "auc"),
    "timeseries": ("outputs/timeseries/ETTh1", COMMON, "mase"),
}
FORBIDDEN = re.compile(r"Traceback|CUDA out of memory|OutOfMemoryError|\bOOM\b|(^|\s)Killed(?:\s|$)", re.I | re.M)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--log", action="append", required=True)
    ap.add_argument("--marker", required=True)
    ns = ap.parse_args()

    for log in ns.log:
        text = Path(log).read_text(errors="replace")
        if FORBIDDEN.search(text):
            raise SystemExit(f"forbidden error in {log}")
        if text.count("python_exit=0") != 6:
            raise SystemExit(f"expected 6 successful trials in {log}, got {text.count('python_exit=0')}")

    report = {"run_id": ns.run_id, "arms": {}, "status": "PASS"}
    for arm, (root, expected, metric) in SPECS.items():
        files = sorted(glob.glob(f"{root}/run_id={ns.run_id}-*/seed_*/results.json"))
        if len(files) != 3:
            raise SystemExit(f"{arm}: expected 3 JSON, got {len(files)}")
        arm_report = []
        for path in files:
            payload = json.loads(Path(path).read_text())
            if payload.get("arm") != arm or int(payload.get("config", {}).get("paired_rng", 0)) != 1:
                raise SystemExit(f"{path}: arm/config identity failure")
            rows = payload.get("results")
            if not isinstance(rows, list):
                raise SystemExit(f"{path}: results is not a list")
            by_method = {str(row.get("method")): row for row in rows}
            if len(by_method) != len(rows) or set(by_method) != expected:
                raise SystemExit(f"{path}: method set mismatch {set(by_method) ^ expected}")
            fit_seeds = set()
            for name, row in by_method.items():
                for key in ("sel_sha12", "fit_seed", "train_order_sha12"):
                    if key not in row:
                        raise SystemExit(f"{path}: {name} missing {key}")
                value = float(row[metric])
                if not math.isfinite(value):
                    raise SystemExit(f"{path}: {name} has non-finite {metric}")
                fit_seeds.add(int(row["fit_seed"]))
            if len(fit_seeds) != 1:
                raise SystemExit(f"{path}: methods do not share one initialization seed")
            pairing = payload.get("pairing_manifest") or {}
            for key in ("pool_sha256", "validation_sha256", "test_sha256"):
                if len(str(pairing.get(key, ""))) != 64:
                    raise SystemExit(f"{path}: invalid {key}")

            adapt = payload.get("adapt_manifest") or {}
            chosen = str((adapt.get("chosen") or {}).get("strategy", ""))
            if chosen in by_method:
                ctrl = by_method["mmds_adapt"]
                base = by_method[chosen]
                if adapt.get("sel_sha12") != base["sel_sha12"] or ctrl["sel_sha12"] != base["sel_sha12"]:
                    raise SystemExit(f"{path}: controller selection parity failed for {chosen}")
                if float(ctrl[metric]) != float(base[metric]):
                    raise SystemExit(f"{path}: controller metric parity failed for {chosen}")
            arm_report.append({"file": path, "seed": payload["seed"], "chosen": chosen,
                               "fit_seed": next(iter(fit_seeds)), "code_sha256_12": payload.get("code_sha256_12")})
        report["arms"][arm] = arm_report

    report_path = Path("experiments") / f"current_code_paired_{ns.run_id}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    Path(ns.marker).write_text(json.dumps({"status": "PASS", "report": str(report_path), "run_id": ns.run_id}, indent=2) + "\n")
    print(f"PASS report={report_path} marker={ns.marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
