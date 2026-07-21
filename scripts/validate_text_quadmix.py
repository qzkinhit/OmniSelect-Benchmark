#!/usr/bin/env python3
"""Strict validator for the isolated three-seed QuaDMix text transfer lane."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


DOMAINS = {"code", "general", "image", "math", "table"}
TASKS = {"arc_easy", "arc_challenge", "hellaswag", "openbookqa"}
FATAL = re.compile(
    r"Traceback|CUDA out of memory|OutOfMemory|Killed|Segmentation fault|"
    r"illegal memory access",
    re.IGNORECASE,
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_dict(value: object, keys: set[str]) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == keys
        and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in value.values())
    )


def fail(message: str) -> None:
    raise SystemExit(f"VALIDATION_FAILED: {message}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--marker", type=Path, required=True)
    args = ap.parse_args()

    repo = args.repo.resolve()
    log = args.log.resolve()
    marker = args.marker.resolve()
    if not log.is_file():
        fail(f"missing log {log}")
    log_text = log.read_text(errors="replace")
    bad = FATAL.search(log_text)
    if bad:
        fail(f"fatal log token {bad.group(0)}")

    runner = repo / "scripts/run_experiment.py"
    pool = repo / "data/processed/qpool_train.jsonl"
    heldout = repo / "data/processed/qpool_heldout.jsonl"
    for path in (runner, pool, heldout):
        if not path.is_file():
            fail(f"missing required file {path}")
    runner_sha12 = digest(runner)[:12]

    tag = (
        f"run_id={args.run_id}-"
        "stratify=1-infl=pplq-train=finetune-lmeval=1"
    )
    evidence = []
    for seed in range(3):
        token = f"SEED={seed} python_exit=0"
        if log_text.count(token) != 1:
            fail(f"expected exactly one {token!r}")
        path = repo / "outputs/experiment" / tag / f"seed_{seed}" / "results.json"
        if not path.is_file():
            fail(f"seed {seed}: missing result {path}")
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            fail(f"seed {seed}: invalid JSON: {exc}")
        if payload.get("arm") != "text" or payload.get("seed") != seed:
            fail(f"seed {seed}: arm/seed mismatch")
        cfg = payload.get("config", {})
        expected = {
            "budget_frac": 0.5,
            "passes": 2.0,
            "ctx": 512,
            "n": 25000,
            "methods": ["quadmix"],
            "stratify": "1",
            "infl_kind": "pplq",
            "train_mode": "finetune",
            "lmeval_tasks": "arc_easy,arc_challenge,hellaswag,openbookqa",
            "run_id": args.run_id,
        }
        for key, value in expected.items():
            if cfg.get(key) != value:
                fail(f"seed {seed}: config {key}={cfg.get(key)!r}, expected {value!r}")
        if cfg.get("pool_sha256_12") != digest(pool)[:12]:
            fail(f"seed {seed}: pool hash mismatch")
        if cfg.get("code_sha256_12") != runner_sha12:
            fail(f"seed {seed}: runner hash mismatch")
        rows = payload.get("results")
        if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("method") != "quadmix":
            fail(f"seed {seed}: expected exactly one quadmix row")
        row = rows[0]
        if not finite_dict(row.get("ppl"), DOMAINS):
            fail(f"seed {seed}: invalid five-domain PPL")
        if not finite_dict(row.get("ppl_ctrl"), DOMAINS):
            fail(f"seed {seed}: invalid adjudication PPL")
        if not finite_dict(row.get("lmeval"), TASKS):
            fail(f"seed {seed}: invalid lm-eval metrics")
        if not isinstance(row.get("n_sel"), int) or row["n_sel"] <= 0:
            fail(f"seed {seed}: invalid selected count")
        if not isinstance(row.get("tok_sel"), int) or row["tok_sel"] <= 0:
            fail(f"seed {seed}: invalid selected token count")
        if not re.fullmatch(r"[0-9a-f]{12}", str(row.get("sel_sha256_12", ""))):
            fail(f"seed {seed}: missing selected-ID hash")
        evidence.append(
            {
                "seed": seed,
                "json": str(path),
                "json_sha256": digest(path),
                "n_sel": row["n_sel"],
                "tok_sel": row["tok_sel"],
                "sel_sha256_12": row["sel_sha256_12"],
            }
        )

    report = {
        "status": "PASS",
        "scope": "QuaDMix text five-domain unified-budget transfer, three seeds",
        "run_id": args.run_id,
        "runner_sha256": digest(runner),
        "pool_sha256": digest(pool),
        "heldout_sha256": digest(heldout),
        "log": str(log),
        "log_sha256": digest(log),
        "evidence": evidence,
        "claim_limit": (
            "Local QuaDMix-style implementation under the paper's fixed stratified token "
            "budget. This is not the original 570B-token QuaDMix reproduction."
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
