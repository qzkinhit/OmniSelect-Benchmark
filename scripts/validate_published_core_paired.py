#!/usr/bin/env python3
"""Fail-closed validator for the equation-level published-core transfer batch."""
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
    "quadmix", "quadmix_pub", "dmf", "dmf_pub", "mmds_adapt",
}
CLASSIFICATION = COMMON | {"el2n", "grand", "ccs"}
SPECS = {
    "vision": ("outputs/vision/uoft-cs_cifar100", CLASSIFICATION, "acc"),
    "tep": ("outputs/tep/tep21", CLASSIFICATION, "f1"),
    "tabular": ("outputs/tabular/electricity", CLASSIFICATION, "auc"),
    "timeseries": ("outputs/timeseries/ETTh1", COMMON, "mase"),
}
FORBIDDEN = re.compile(
    r"Traceback|CUDA out of memory|OutOfMemoryError|\bOOM\b|(^|\s)Killed(?:\s|$)",
    re.I | re.M,
)


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
            raise SystemExit(
                f"expected 6 successful trials in {log}, got {text.count('python_exit=0')}"
            )

    report = {"run_id": ns.run_id, "arms": {}, "status": "PASS"}
    baseline_shas = set()
    for arm, (root, expected, metric) in SPECS.items():
        files = sorted(glob.glob(f"{root}/run_id={ns.run_id}-*/seed_*/results.json"))
        if len(files) != 3:
            raise SystemExit(f"{arm}: expected 3 JSON, got {len(files)}")
        arm_report = []
        for path in files:
            payload = json.loads(Path(path).read_text())
            cfg = payload.get("config", {})
            if payload.get("arm") != arm or int(cfg.get("paired_rng", 0)) != 1:
                raise SystemExit(f"{path}: arm/config identity failure")
            if payload.get("fidelity_mode") != "published-core-unified-protocol-v1":
                raise SystemExit(f"{path}: fidelity mode mismatch")
            protocol = payload.get("published_core_protocol") or {}
            if str((protocol.get("quadmix") or {}).get("equations")) != "1-3":
                raise SystemExit(f"{path}: QuaDMix equation protocol missing")
            if str((protocol.get("quadmix") or {}).get("budget_adapter")) != "gumbel-top-k-without-replacement":
                raise SystemExit(f"{path}: QuaDMix budget adapter mismatch")
            if str((protocol.get("dmf") or {}).get("equations")) != "6-8":
                raise SystemExit(f"{path}: DMF equation protocol missing")
            if int((protocol.get("dmf") or {}).get("rounds", 0)) != 6:
                raise SystemExit(f"{path}: DMF round protocol mismatch")
            baseline_sha = str(payload.get("baseline_impl_sha256", ""))
            if len(baseline_sha) != 64:
                raise SystemExit(f"{path}: missing baseline implementation SHA256")
            baseline_shas.add(baseline_sha)
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
            if not chosen:
                raise SystemExit(f"{path}: controller choice is missing")
            ctrl = by_method["mmds_adapt"]
            if adapt.get("sel_sha12") != ctrl["sel_sha12"]:
                raise SystemExit(f"{path}: controller manifest/row selection mismatch")
            if chosen in by_method:
                base = by_method[chosen]
                if ctrl["sel_sha12"] != base["sel_sha12"]:
                    raise SystemExit(f"{path}: controller selection parity failed for {chosen}")
                if float(ctrl[metric]) != float(base[metric]):
                    raise SystemExit(f"{path}: controller metric parity failed for {chosen}")
            arm_report.append({
                "file": path,
                "seed": payload["seed"],
                "chosen": chosen,
                "fit_seed": next(iter(fit_seeds)),
                "code_sha256_12": payload.get("code_sha256_12"),
                "baseline_impl_sha256": baseline_sha,
            })
        report["arms"][arm] = arm_report

    if len(baseline_shas) != 1:
        raise SystemExit(f"baseline implementation drift across trials: {sorted(baseline_shas)}")
    report["baseline_impl_sha256"] = next(iter(baseline_shas))
    report_path = Path("experiments") / f"published_core_paired_{ns.run_id}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    Path(ns.marker).write_text(json.dumps({
        "status": "PASS",
        "report": str(report_path),
        "run_id": ns.run_id,
        "baseline_impl_sha256": report["baseline_impl_sha256"],
    }, indent=2) + "\n")
    print(f"PASS report={report_path} marker={ns.marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
