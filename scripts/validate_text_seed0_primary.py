#!/usr/bin/env python3
"""Fail-closed validation for the formal seed-0 five-domain text run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mmdataselect.utils.repro_bundle import (
    canonical_sha256,
    file_sha256,
    validate_repro_bundle,
)


MODEL = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
RUN_METHODS = [
    "random",
    "influence_only",
    "coverage_text",
    "fixed_fusion",
    "herding_text",
    "density_text",
    "quadmix_pub",
    "dmf_pub",
]
TABLE_BASELINES = [
    "random",
    "influence_only",
    "coverage_text",
    "fixed_fusion",
    "herding_text",
    "el2n",
    "grand",
    "ccs",
    "density_text",
    "quadmix_pub",
    "dmf_pub",
]
SKIPPED_METHODS = ["el2n", "grand", "ccs"]
REFERENCES = [
    "random",
    "influence_only",
    "coverage_text",
    "herding_text",
    "density_text",
    "quadmix_pub",
    "dmf_pub",
]
CHALLENGERS = ["fixed_fusion"]
TASKS = ["arc_easy", "arc_challenge", "hellaswag", "openbookqa"]


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def _require_file(path: Path, expected_sha256: str | None = None) -> dict:
    if not path.is_file():
        fail(f"missing required file: {path}")
    digest = file_sha256(path)
    if expected_sha256 and digest != expected_sha256:
        fail(f"SHA-256 mismatch for {path}: {digest} != {expected_sha256}")
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def preflight(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    files = {
        "pool": _require_file(repo / "data/processed/qpool_train.jsonl", args.pool_sha),
        "heldout": _require_file(
            repo / "data/processed/qpool_heldout.jsonl", args.heldout_sha
        ),
        "influence_cache": _require_file(Path(args.influence_cache), args.influence_sha),
        "runner": _require_file(repo / "scripts/run_experiment.py"),
        "quadmix_impl": _require_file(
            repo / "src/mmdataselect/selectors/external_baselines.py"
        ),
    }
    try:
        import torch
        import transformers
    except Exception as exc:
        fail(f"runtime import failed: {type(exc).__name__}: {exc}")
    if not args.allow_noncuda and not torch.cuda.is_available():
        fail("CUDA is unavailable; formal training must run in 有卡模式")
    git_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    )
    payload = {
        "schema_version": "omniselect.text-seed0-preflight.v1",
        "model": MODEL,
        "revision": REVISION,
        "seed": 0,
        "methods": RUN_METHODS,
        "table_baselines": TABLE_BASELINES,
        "skipped_methods": SKIPPED_METHODS,
        "files": files,
        "git_head": git_head,
        "git_dirty": bool(status),
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    payload["preflight_sha256"] = canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"PASS preflight -> {args.output}")


def _finite_positive(value, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        fail(f"{label} is not finite and positive: {value!r}")
    return float(value)


def _runner_sha(value) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def resolve_artifact_path(value: str, artifact_root: Path) -> Path:
    """Resolve an artifact path from a result bundle after it has been moved.

    Formal result JSONs retain the original server-side checkpoint path so their
    source artifact hash remains stable. A downloaded bundle may live elsewhere;
    in that case, recover the portable suffix beginning at checkpoints or
    repro_bundle beneath the supplied bundle root.
    """
    source = Path(value)
    if not source.is_absolute():
        return artifact_root / source
    if source.exists():
        return source
    for marker in ("checkpoints", "repro_bundle"):
        if marker in source.parts:
            return artifact_root.joinpath(*source.parts[source.parts.index(marker) :])
    return source


def validate_result(args: argparse.Namespace) -> None:
    path = args.result.resolve()
    artifact_root = (args.artifact_root or path.parent).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("arm") != "text" or payload.get("seed") != 0:
        fail("arm/seed mismatch")
    if payload.get("execution_phase") != "method_v3_terminal":
        fail("not a method-v3 terminal result")
    config = payload.get("config") or {}
    expected = {
        "methods": RUN_METHODS,
        "table_baselines": TABLE_BASELINES,
        "skipped_methods": SKIPPED_METHODS,
        "reference_methods": REFERENCES,
        "challenger_methods": CHALLENGERS,
        "method_v3": 1,
        "stratify": 1,
        "infl_kind": "pplq",
        "budget_frac": 0.5,
        "passes": 2.0,
        "ctx": 512,
        "batch_size": 16,
        "text_embed_batch_size": 32,
        "train_mode": "finetune",
        "ref_model": MODEL,
        "ref_model_revision": REVISION,
        "tokenizer_revision": REVISION,
        "only_domain": "",
        "pool_n": 25000,
        "lmeval": 1,
        "lmeval_tasks": TASKS,
        "lmeval_limit": 0,
        "ft_lr": 2e-5,
        "ft_steps_cap": 0,
        "ft_freeze": 0,
        "report_all_candidates": 1,
        "lmeval_all_candidates": 0,
        "save_all_candidate_models": 1,
        "max_portfolio": 8,
    }
    for key, value in expected.items():
        actual = config.get(key)
        if isinstance(actual, tuple):
            actual = list(actual)
        if actual != value:
            fail(f"config {key}={actual!r}, expected {value!r}")
    config_for_sha = dict(config)
    recorded_config_sha = config_for_sha.pop("config_sha256", None)
    config_for_sha.pop("run_id", None)
    if recorded_config_sha != _runner_sha(config_for_sha):
        fail("config_sha256 mismatch")

    rows = payload.get("candidate_v1_evidence") or []
    by_method = {row.get("method"): row for row in rows}
    if (
        len(by_method) != len(rows)
        or len(rows) != len(RUN_METHODS)
        or set(by_method) != set(RUN_METHODS)
    ):
        fail(f"candidate registry mismatch: {list(by_method)}")
    initial_hashes, order_hashes, optimizer_hashes, caps = set(), set(), set(), set()
    for method in RUN_METHODS:
        row = by_method[method]
        selected = row.get("selection_indices")
        if not isinstance(selected, list) or not selected:
            fail(f"{method}: missing selected indices")
        if len(selected) != len(set(selected)):
            fail(f"{method}: selected indices are not unique")
        if row.get("selection_order_sha256") != _runner_sha(
            row.get("selected_record_ids")
        ):
            fail(f"{method}: selected-record order hash mismatch")
        ppl = row.get("report_ppl")
        if not isinstance(ppl, dict) or len(ppl) != 5:
            fail(f"{method}: missing five-domain report PPL")
        for domain, value in ppl.items():
            _finite_positive(value, f"{method}/{domain} PPL")
        _finite_positive(row.get("report_gmean_ppl"), f"{method} geometric PPL")
        if row.get("lmeval") is not None:
            fail(f"{method}: unexpected per-candidate lm-eval in fast primary run")
        training = row.get("training_manifest") or {}
        initial_hashes.add(training.get("initial_state_sha256"))
        order_hashes.add(training.get("training_block_order_sha256"))
        optimizer_hashes.add(training.get("optimizer_config_sha256"))
        caps.add(row.get("effective_train_tokens"))
        checkpoint = resolve_artifact_path(
            str(row.get("checkpoint_path", "")), artifact_root
        )
        if not checkpoint.is_dir():
            fail(f"{method}: checkpoint directory missing")
    dmf_metadata = by_method["dmf_pub"].get("selector_metadata") or {}
    if (
        dmf_metadata.get("fidelity")
        != "published-update unified-token-budget transfer"
        or dmf_metadata.get("construction_split_only") is not True
        or dmf_metadata.get("rounds") != 6
        or len(dmf_metadata.get("trace") or []) != 6
    ):
        fail("DMF-pub published-update construction trace is incomplete")
    protocols = config.get("text_transfer_protocols") or {}
    expected_transfers = {
        "coverage_text",
        "herding_text",
        "density_text",
    }
    if set(protocols) != expected_transfers:
        fail(f"text-transfer protocol registry mismatch: {set(protocols)}")
    if set(config.get("skipped_method_reasons") or {}) != set(SKIPPED_METHODS):
        fail("skipped text methods do not have exact reason codes")
    for label, values in (
        ("initial state", initial_hashes),
        ("training block order", order_hashes),
        ("optimizer", optimizer_hashes),
        ("effective token cap", caps),
    ):
        if len(values) != 1 or None in values:
            fail(f"candidates do not share one {label}: {values}")

    bundle = artifact_root / "repro_bundle"
    errors = validate_repro_bundle(bundle)
    if errors:
        fail("repro bundle invalid: " + "; ".join(errors))
    for method in RUN_METHODS:
        safe = method.replace("/", "_")
        if not (bundle / "selections" / f"{safe}.json").is_file():
            fail(f"{method}: selection manifest absent from repro bundle")
        if not (bundle / "selected_data" / f"{safe}.jsonl").is_file():
            fail(f"{method}: selected JSONL dataset absent from repro bundle")
    for relative in (
        bundle / "selections" / "mmds_adapt.json",
        bundle / "selected_data" / "mmds_adapt.jsonl",
    ):
        if not relative.is_file():
            fail(f"OmniSelect artifact absent from repro bundle: {relative}")

    stored_payload_sha = payload.get("artifact_sha256")
    payload_without_sha = dict(payload)
    payload_without_sha.pop("artifact_sha256", None)
    if stored_payload_sha != _runner_sha(payload_without_sha):
        fail("results payload artifact_sha256 mismatch")

    final_rows = payload.get("results") or []
    if len(final_rows) != 1 or final_rows[0].get("method") != "mmds_adapt":
        fail("missing the single OmniSelect terminal row")
    final_row = final_rows[0]
    if final_row.get("picked") not in RUN_METHODS:
        fail("OmniSelect selected an arm outside the applicable main-table registry")
    final_checkpoint = resolve_artifact_path(
        str(final_row.get("checkpoint_path", "")), artifact_root
    )
    if not final_checkpoint.is_dir():
        fail("OmniSelect selected checkpoint directory is missing")
    _finite_positive(final_row.get("gmean_ppl"), "OmniSelect geometric PPL")
    final_ppl = final_row.get("ppl")
    if not isinstance(final_ppl, dict) or len(final_ppl) != 5:
        fail("OmniSelect is missing five-domain report PPL")
    final_lmeval = final_row.get("lmeval")
    if not isinstance(final_lmeval, dict) or list(final_lmeval) != TASKS:
        fail("OmniSelect lm-eval task registry mismatch")

    random_ppl = by_method["random"]["report_gmean_ppl"]
    quadmix_ppl = by_method["quadmix_pub"]["report_gmean_ppl"]
    summary = {
        "result": str(path),
        "result_sha256": file_sha256(path),
        "random_gmean_ppl": random_ppl,
        "quadmix_pub_gmean_ppl": quadmix_ppl,
        "quadmix_minus_random": quadmix_ppl - random_ppl,
        "selected_arm": payload["decision_record"]["selected_arm"],
        "bundle": str(bundle),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"PASS result -> {args.output}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preflight")
    pre.add_argument("--repo", type=Path, required=True)
    pre.add_argument("--pool-sha", required=True)
    pre.add_argument("--heldout-sha", required=True)
    pre.add_argument("--influence-cache", required=True)
    pre.add_argument("--influence-sha", required=True)
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--allow-noncuda", action="store_true")
    pre.set_defaults(func=preflight)
    result = commands.add_parser("result")
    result.add_argument("--result", type=Path, required=True)
    result.add_argument(
        "--artifact-root",
        type=Path,
        help="moved run directory containing checkpoints/ and repro_bundle/; defaults to the result's parent",
    )
    result.add_argument("--output", type=Path, required=True)
    result.set_defaults(func=validate_result)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
