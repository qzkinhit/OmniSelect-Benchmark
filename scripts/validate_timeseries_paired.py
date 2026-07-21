#!/usr/bin/env python3
"""Strict validator for an isolated paired ETTm1 x DLinear three-seed lane.

The validator never infers success from a done file.  It checks the canonical JSON
payloads, protocol/config/code hashes, method coverage, finite metrics, pairing
instrumentation, log exit evidence, and controller-to-standalone selection parity
whenever the controller chose a named method that is present in the same trial.
Only a fully passing run may create the requested marker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


EXPECTED_METHODS = {
    "full",
    "random",
    "coreset",
    "auth_only",
    "influence_only",
    "mmdataselect",
    "mmds_adapt",
    "herding",
    "kcenter",
    "semdedup",
    "density",
    "quadmix",
    "dmf",
}

EXPECTED_CONFIG = {
    "model": "dlinear",
    "pool": 3000,
    "budget": 0.3,
    "noise": 0.4,
    "L": 96,
    "H": 24,
    "paired_rng": 1,
}

FATAL_LOG_RE = re.compile(
    r"Traceback|CUDA out of memory|OutOfMemory|Killed|Segmentation fault|"
    r"illegal memory access",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"VALIDATION_FAILED: {message}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--marker", type=Path, required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    log = args.log.resolve()
    marker = args.marker.resolve()

    if not log.is_file():
        fail(f"missing log: {log}")
    log_text = log.read_text(errors="replace")
    fatal = FATAL_LOG_RE.search(log_text)
    if fatal:
        fail(f"fatal log token: {fatal.group(0)}")

    runner = repo / "scripts/run_timeseries_experiment.py"
    pairing = repo / "src/mmdataselect/utils/pairing.py"
    data = repo / "data/processed/ettm1.csv"
    for path in (runner, pairing, data):
        if not path.is_file():
            fail(f"missing required file: {path}")

    runner_sha12 = sha256(runner)[:12]
    evidence: list[dict] = []
    observed_code_sha12: set[str] = set()

    for seed in range(3):
        exit_token = f"SEED={seed} python_exit=0"
        if log_text.count(exit_token) != 1:
            fail(f"expected exactly one {exit_token!r}")

        pattern = (
            repo
            / "outputs"
            / "timeseries"
            / "ETTm1"
            / f"run_id={args.run_id}-*"
            / f"seed_{seed}"
            / "results.json"
        )
        matches = list(pattern.parent.parent.parent.glob(
            f"run_id={args.run_id}-*/seed_{seed}/results.json"
        ))
        if len(matches) != 1:
            fail(f"seed {seed}: expected one result JSON, found {len(matches)}")
        path = matches[0]
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            fail(f"seed {seed}: invalid JSON: {exc}")

        if payload.get("arm") != "timeseries":
            fail(f"seed {seed}: arm mismatch")
        if payload.get("dataset") != "ETTm1":
            fail(f"seed {seed}: dataset mismatch")
        if payload.get("seed") != seed:
            fail(f"seed {seed}: seed mismatch")
        if payload.get("config") != EXPECTED_CONFIG:
            fail(f"seed {seed}: config mismatch: {payload.get('config')!r}")

        code_sha12 = payload.get("code_sha256_12")
        observed_code_sha12.add(str(code_sha12))
        if code_sha12 != runner_sha12:
            fail(
                f"seed {seed}: recorded runner hash {code_sha12} != current {runner_sha12}"
            )

        rows = payload.get("results")
        if not isinstance(rows, list) or len(rows) != len(EXPECTED_METHODS):
            fail(f"seed {seed}: expected {len(EXPECTED_METHODS)} result rows")
        by_method: dict[str, dict] = {}
        for row in rows:
            if not isinstance(row, dict):
                fail(f"seed {seed}: non-object result row")
            method = row.get("method")
            if method in by_method:
                fail(f"seed {seed}: duplicate method {method}")
            by_method[str(method)] = row
            if not isinstance(row.get("mase"), (int, float)) or not math.isfinite(
                float(row["mase"])
            ):
                fail(f"seed {seed} {method}: non-finite MASE")
            if not isinstance(row.get("clean%"), (int, float)) or not 0 <= float(
                row["clean%"]
            ) <= 1:
                fail(f"seed {seed} {method}: invalid clean%")
            if not isinstance(row.get("n"), int) or row["n"] <= 0:
                fail(f"seed {seed} {method}: invalid n")
            if not re.fullmatch(r"[0-9a-f]{12}", str(row.get("sel_sha12", ""))):
                fail(f"seed {seed} {method}: missing selection hash")
            if not isinstance(row.get("fit_seed"), int):
                fail(f"seed {seed} {method}: missing fit seed")
            if not re.fullmatch(
                r"[0-9a-f]{12}", str(row.get("train_order_sha12", ""))
            ):
                fail(f"seed {seed} {method}: missing train-order hash")
        if set(by_method) != EXPECTED_METHODS:
            fail(
                f"seed {seed}: method mismatch, missing={sorted(EXPECTED_METHODS-set(by_method))}, "
                f"extra={sorted(set(by_method)-EXPECTED_METHODS)}"
            )

        adapt = payload.get("adapt_manifest")
        if not isinstance(adapt, dict) or not isinstance(adapt.get("chosen"), dict):
            fail(f"seed {seed}: missing controller manifest")
        chosen = str(adapt["chosen"].get("strategy", ""))
        ctrl = by_method["mmds_adapt"]
        if adapt.get("sel_sha12") != ctrl["sel_sha12"]:
            fail(f"seed {seed}: controller manifest selection hash mismatch")
        parity = "not-applicable-composite-strategy"
        if chosen in by_method:
            baseline = by_method[chosen]
            if ctrl["sel_sha12"] != baseline["sel_sha12"]:
                fail(
                    f"seed {seed}: controller chose {chosen} but selection hashes differ"
                )
            if ctrl["fit_seed"] != baseline["fit_seed"]:
                fail(f"seed {seed}: controller/{chosen} fit seeds differ")
            if ctrl["train_order_sha12"] != baseline["train_order_sha12"]:
                fail(f"seed {seed}: controller/{chosen} train-order hashes differ")
            if float(ctrl["mase"]) != float(baseline["mase"]):
                fail(f"seed {seed}: controller/{chosen} metrics differ")
            parity = f"exact:{chosen}"

        evidence.append(
            {
                "seed": seed,
                "json": str(path),
                "json_sha256": sha256(path),
                "chosen_strategy": chosen,
                "controller_parity": parity,
            }
        )

    if observed_code_sha12 != {runner_sha12}:
        fail(f"multiple recorded code hashes: {sorted(observed_code_sha12)}")

    report = {
        "status": "PASS",
        "scope": "ETTm1 x DLinear unified-budget paired three-seed coverage",
        "run_id": args.run_id,
        "methods": sorted(EXPECTED_METHODS),
        "runner_sha256": sha256(runner),
        "pairing_sha256": sha256(pairing),
        "data_sha256": sha256(data),
        "log": str(log),
        "log_sha256": sha256(log),
        "evidence": evidence,
        "claim_limit": (
            "Unified-budget coverage only. This marker is not original-paper fidelity "
            "evidence for every named baseline."
        ),
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker.with_suffix(marker.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(marker)
    print(json.dumps(report, indent=2))
    print(f"VALIDATION_OK -> {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
