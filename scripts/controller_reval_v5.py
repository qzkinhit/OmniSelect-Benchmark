#!/usr/bin/env python3
"""Fail-closed revalidation of the current formal AdaptiveController.

This runner recomputes 21 paper-facing controller configurations over three seeds.
Historical logs are used only as an impact comparison. They are not treated as a
causal finite-fix control because the historical runners, environments, and
controller portfolios changed over time.

The script never writes legacy flat result files. Every child receives a unique
RUN_ID, and every accepted result must pass identity, protocol, code, log, metric,
manifest, and selected-index-hash checks.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import glob
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any


VERSION = "5.1-current-portfolio"
TOL = 0.005
METRIC_FIELD = {"test_acc": "acc", "f1": "f1", "auc": "auc", "mase": "mase"}
VAL_PAT = r"\(val(?:_[a-zA-Z0-9]+)?=(-?[0-9.]+)\)"
FORBIDDEN_PAT = re.compile(
    r"Traceback|CUDA out of memory|OutOfMemoryError|\bOOM\b|(^|\s)Killed(?:\s|$)",
    re.IGNORECASE | re.MULTILINE,
)


SCRUB_VARS = sorted(
    {
        "ADAPT_GRPO",
        "ADAPT_MARGIN",
        "ADAPT_SH",
        "AUTH_Q",
        "BUDGET_FRAC",
        "CHRONOS_EPOCHS",
        "CHRONOS_LR",
        "CHRONOS_MAX_EPOCHS",
        "CNN_EPOCHS",
        "DROP_CHANNEL",
        "DSDM_RUNS",
        "EPOCHS",
        "H",
        "INFL_KIND",
        "KNN",
        "L",
        "LAM",
        "METHODS",
        "MODEL",
        "NOISE_FRAC",
        "N_FAULTS",
        "POOL_N",
        "ROBUST_VAL",
        "RUN_ID",
        "SEED",
        "STRATIFY",
        "TAB_DATASET",
        "TEP_CALIB",
        "TEST_N",
        "TS_DATASET",
        "TS_MODEL",
        "TS_VAL_MODE",
        "VAL_N",
        "VAL_NOISE",
        "VIS_DATASET",
        "VIS_ENCODER",
        "VIS_NOISE",
        "W_INFL",
    }
)


COMMON: dict[str, str | None] = {
    "METHODS": "mmds_adapt",
    "ADAPT_GRPO": "0",
    "ADAPT_MARGIN": "0.015",
    "ADAPT_SH": "0",
    "ROBUST_VAL": "0",
    "VAL_NOISE": "0",
}

VISION: dict[str, str | None] = {
    **COMMON,
    "VIS_DATASET": "uoft-cs/cifar100",
    "VIS_ENCODER": "openai/clip-vit-base-patch32",
    "POOL_N": "4000",
    "VAL_N": "800",
    "TEST_N": "2000",
    "NOISE_FRAC": "0.4",
    "BUDGET_FRAC": "0.5",
    "KNN": "15",
    "LAM": "0.5",
    "AUTH_Q": "0.25",
    "W_INFL": "0.5",
    "VIS_NOISE": "inject",
    "DROP_CHANNEL": None,
    "DSDM_RUNS": "20",
}

TEP: dict[str, str | None] = {
    **COMMON,
    "MODEL": "mlp",
    "N_FAULTS": "21",
    "POOL_N": "4000",
    "VAL_N": "2000",
    "TEST_N": "3000",
    "NOISE_FRAC": "0.4",
    "BUDGET_FRAC": "0.3",
    "KNN": "15",
    "LAM": "0.5",
    "AUTH_Q": "0.25",
    "W_INFL": "0.5",
    "DROP_CHANNEL": None,
    "TEP_CALIB": "0",
    "CNN_EPOCHS": "80",
    "DSDM_RUNS": "16",
}

TABULAR: dict[str, str | None] = {
    **COMMON,
    "TAB_DATASET": "electricity",
    "MODEL": "tabpfn",
    "POOL_N": "3000",
    "VAL_N": "2500",
    "TEST_N": "2000",
    "NOISE_FRAC": "0.4",
    "BUDGET_FRAC": "0.5",
    "KNN": "15",
    "LAM": "0.5",
    "AUTH_Q": "0.25",
    "W_INFL": "0.5",
    "DSDM_RUNS": "12",
}

DLINEAR: dict[str, str | None] = {
    **COMMON,
    "TS_MODEL": "dlinear",
    "POOL_N": "3000",
    "VAL_N": "1000",
    "TEST_N": "1500",
    "L": "96",
    "H": "24",
    "NOISE_FRAC": "0.4",
    "BUDGET_FRAC": "0.3",
    "KNN": "15",
    "EPOCHS": "60",
    "LAM": "0.5",
    "AUTH_Q": "0.25",
    "W_INFL": "0.5",
    "TS_VAL_MODE": "full",
    "DSDM_RUNS": "12",
    "CHRONOS_EPOCHS": None,
    "CHRONOS_MAX_EPOCHS": None,
    "CHRONOS_LR": None,
}

CHRONOS: dict[str, str | None] = {
    **DLINEAR,
    "TS_MODEL": "chronos",
    "DSDM_RUNS": "8",
    "CHRONOS_EPOCHS": None,
    "CHRONOS_MAX_EPOCHS": "4",
    "CHRONOS_LR": "1e-4",
}


def with_updates(base: dict[str, str | None], **changes: str | None) -> dict[str, str | None]:
    out = dict(base)
    out.update(changes)
    return out


PLAN: list[dict[str, Any]] = [
    {"tag": "vision-base", "env": VISION, "script": "scripts/run_vision_experiment.py",
     "oldlog": "experiments/split_protocol_3seed.log", "headers": [r"##### GRPO=0 SEED={s} VISION #####"],
     "metric": "test_acc", "arm": "vision", "dataset": "uoft-cs/cifar100", "semantic_base": "06a8a85"},
    {"tag": "tep-base", "env": TEP, "script": "scripts/run_tep_experiment.py",
     "oldlog": "experiments/split_protocol_3seed.log", "headers": [r"##### GRPO=0 SEED={s} TEP #####"],
     "metric": "f1", "arm": "tep", "dataset": "tep21", "semantic_base": "06a8a85"},
    {"tag": "tabular-base", "env": TABULAR, "script": "scripts/run_tabular_experiment.py",
     "oldlog": "experiments/split_protocol_3seed.log", "headers": [r"##### GRPO=0 SEED={s} TAB #####"],
     "metric": "auc", "arm": "tabular", "dataset": "electricity", "semantic_base": "06a8a85"},
    {"tag": "ts-ETTh1", "env": with_updates(DLINEAR, TS_DATASET="ETTh1"),
     "script": "scripts/run_timeseries_experiment.py", "oldlog": "experiments/split_protocol_3seed.log",
     "headers": [r"##### GRPO=0 SEED={s} ETTH1 #####"], "metric": "mase",
     "arm": "timeseries", "dataset": "ETTh1", "semantic_base": "06a8a85"},
    {"tag": "ts-ETTh2", "env": with_updates(DLINEAR, TS_DATASET="ETTh2"),
     "script": "scripts/run_timeseries_experiment.py", "oldlog": "experiments/etth2_3seed.log",
     "headers": [r"##### SEED {s}[^\n]*#####"], "metric": "mase",
     "arm": "timeseries", "dataset": "ETTh2", "semantic_base": "1c75852"},
    {"tag": "ts-daisy_cstr", "env": with_updates(DLINEAR, TS_DATASET="daisy_cstr"),
     "script": "scripts/run_timeseries_experiment.py", "oldlog": "experiments/daisy_cstr_3seed.log",
     "headers": [r"##### SEED {s}[^\n]*#####"], "metric": "mase",
     "arm": "timeseries", "dataset": "daisy_cstr", "semantic_base": "1c75852"},
    {"tag": "ts-daisy_steamgen", "env": with_updates(DLINEAR, TS_DATASET="daisy_steamgen"),
     "script": "scripts/run_timeseries_experiment.py", "oldlog": "experiments/daisy_steamgen_3seed.log",
     "headers": [r"##### SEED {s}[^\n]*#####"], "metric": "mase",
     "arm": "timeseries", "dataset": "daisy_steamgen", "semantic_base": "1c75852"},
    {"tag": "chronos-ETTh1", "env": with_updates(CHRONOS, TS_DATASET="ETTh1"),
     "script": "scripts/run_timeseries_experiment.py", "oldlog": "experiments/chronos_fm_3seed.log",
     "headers": [r"##### CHRONOS ds=ETTh1 SEED={s} #####"], "metric": "mase",
     "arm": "timeseries", "dataset": "ETTh1", "semantic_base": "b95aa92"},
    {"tag": "chronos-ETTh2", "env": with_updates(CHRONOS, TS_DATASET="ETTh2"),
     "script": "scripts/run_timeseries_experiment.py", "oldlog": "experiments/chronos_fm_3seed.log",
     "headers": [r"##### CHRONOS ds=ETTh2 SEED={s} #####"], "metric": "mase",
     "arm": "timeseries", "dataset": "ETTh2", "semantic_base": "b95aa92"},
    {"tag": "chronos-daisy_cstr", "env": with_updates(CHRONOS, TS_DATASET="daisy_cstr"),
     "script": "scripts/run_timeseries_experiment.py", "oldlog": "experiments/chronos_fm_3seed.log",
     "headers": [r"##### CHRONOS ds=daisy_cstr SEED={s} #####"], "metric": "mase",
     "arm": "timeseries", "dataset": "daisy_cstr", "semantic_base": "b95aa92"},
    {"tag": "chronos-daisy_steamgen", "env": with_updates(CHRONOS, TS_DATASET="daisy_steamgen"),
     "script": "scripts/run_timeseries_experiment.py", "oldlog": "experiments/chronos_fm_3seed.log",
     "headers": [r"##### CHRONOS ds=daisy_steamgen SEED={s} #####"], "metric": "mase",
     "arm": "timeseries", "dataset": "daisy_steamgen", "semantic_base": "b95aa92"},
    {"tag": "chronos-ETTm1", "env": with_updates(CHRONOS, TS_DATASET="ETTm1"),
     "script": "scripts/run_timeseries_experiment.py", "oldlog": "experiments/chronos_ettm1_3seed.log",
     "headers": [r"##### CHRONOS ds=ETTm1 SEED={s} #####"], "metric": "mase",
     "arm": "timeseries", "dataset": "ETTm1", "semantic_base": "b95aa92"},
    {"tag": "vision-nf0.2", "env": with_updates(VISION, NOISE_FRAC="0.2"),
     "script": "scripts/run_vision_experiment.py", "oldlog": "experiments/noise_ratio_ablation_3seed.log",
     "headers": [r"##### VIS-NOISE nf=0.2 SEED={s} #####"], "metric": "test_acc",
     "arm": "vision", "dataset": "uoft-cs/cifar100", "semantic_base": "b95aa92"},
    {"tag": "vision-nf0.6", "env": with_updates(VISION, NOISE_FRAC="0.6"),
     "script": "scripts/run_vision_experiment.py", "oldlog": "experiments/noise_ratio_ablation_3seed.log",
     "headers": [r"##### VIS-NOISE nf=0.6 SEED={s} #####"], "metric": "test_acc",
     "arm": "vision", "dataset": "uoft-cs/cifar100", "semantic_base": "b95aa92"},
    {"tag": "tep-nf0.2", "env": with_updates(TEP, NOISE_FRAC="0.2"),
     "script": "scripts/run_tep_experiment.py", "oldlog": "experiments/noise_ratio_ablation_3seed.log",
     "headers": [r"##### TEP-NOISE nf=0.2 SEED={s} #####"], "metric": "f1",
     "arm": "tep", "dataset": "tep21", "semantic_base": "b95aa92"},
    {"tag": "tep-nf0.6", "env": with_updates(TEP, NOISE_FRAC="0.6"),
     "script": "scripts/run_tep_experiment.py", "oldlog": "experiments/noise_ratio_ablation_3seed.log",
     "headers": [r"##### TEP-NOISE nf=0.6 SEED={s} #####"], "metric": "f1",
     "arm": "tep", "dataset": "tep21", "semantic_base": "b95aa92"},
    {"tag": "vision-drop-infl", "env": with_updates(VISION, DROP_CHANNEL="infl"),
     "script": "scripts/run_vision_experiment.py", "oldlog": "experiments/channel_drop_ablation_3seed.log",
     "headers": [r"##### VIS-DROP ch=infl SEED={s} #####"], "metric": "test_acc",
     "arm": "vision", "dataset": "uoft-cs/cifar100", "semantic_base": "c133a0a"},
    {"tag": "vision-drop-red", "env": with_updates(VISION, DROP_CHANNEL="red"),
     "script": "scripts/run_vision_experiment.py", "oldlog": "experiments/channel_drop_ablation_3seed.log",
     "headers": [r"##### VIS-DROP ch=red SEED={s} #####"], "metric": "test_acc",
     "arm": "vision", "dataset": "uoft-cs/cifar100", "semantic_base": "c133a0a"},
    {"tag": "tep-drop-infl", "env": with_updates(TEP, DROP_CHANNEL="infl"),
     "script": "scripts/run_tep_experiment.py", "oldlog": "experiments/channel_drop_ablation_3seed.log",
     "headers": [r"##### TEP-DROP ch=infl SEED={s} #####"], "metric": "f1",
     "arm": "tep", "dataset": "tep21", "semantic_base": "c133a0a"},
    {"tag": "tep-drop-red", "env": with_updates(TEP, DROP_CHANNEL="red"),
     "script": "scripts/run_tep_experiment.py", "oldlog": "experiments/channel_drop_ablation_3seed.log",
     "headers": [r"##### TEP-DROP ch=red SEED={s} #####"], "metric": "f1",
     "arm": "tep", "dataset": "tep21", "semantic_base": "c133a0a"},
    {"tag": "vision-realnoise", "env": with_updates(VISION, VIS_NOISE="real", NOISE_FRAC="0.4"),
     "script": "scripts/run_vision_experiment.py", "oldlog": "experiments/real_noise_cifar100n_3seed.log",
     "headers": [r"##### REALN SEED={s} #####"], "metric": "test_acc",
     "arm": "vision", "dataset": "uoft-cs/cifar100", "semantic_base": "67f8bf5"},
]


EXCLUDED = {
    "text-pilot and text-scaleup controller": "No fusion grid or authenticity gate is used",
    "original-protocol ImageNet controller": "It is an argmax over five fixed subsets",
    "drop-auth rows": "The authenticity prefilter is fixed to q=0 when auth is removed",
    "SH and GRPO archived negatives": "They are not paper-facing formal rows",
    "TEP calibrated rows": "Calibration changes FDR and FAR reporting, not selection",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.environ.get("OMNI_ROOT", "/root/autodl-tmp/OmniSelect"))
    p.add_argument("--marker-dir", default=os.environ.get("OMNI_MARKER_DIR", "/root"))
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument(
        "--adopt-current-sha",
        default="",
        help="Adopt a fully gated WITH_CHANGES report after an explicit diff audit",
    )
    p.add_argument("--tags", default="")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--max-trials", type=int, default=0)
    return p.parse_args()


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_sha(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def atomic_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def code_manifest(root: Path) -> dict[str, Any]:
    files: list[Path] = []
    for rel in ("src", "baselines"):
        top = root / rel
        if top.exists():
            files.extend(p for p in top.rglob("*.py") if p.is_file())
    files.extend(root / spec["script"] for spec in PLAN)
    files.append(Path(__file__).resolve())
    unique = sorted({p.resolve() for p in files if p.exists()}, key=str)
    mapping: dict[str, str] = {}
    for path in unique:
        try:
            rel = str(path.relative_to(root.resolve()))
        except ValueError:
            rel = str(path)
        mapping[rel] = sha256(path)
    return {"tree_sha256": json_sha(mapping), "files": mapping}


def asset_manifest(root: Path) -> dict[str, Any]:
    """Hash the data, cached features, model refs, and runtime used by all trials."""
    files: list[Path] = []
    files.extend(path for path in (root / "data/tep").glob("*") if path.is_file())
    for rel in (
        "data/daisy/cstr.dat",
        "data/daisy/steamgen.dat",
        "data/cifar_n/CIFAR-100_human.pt",
        "data/processed/etth1.csv",
        "data/processed/etth2.csv",
        "data/processed/ettm1.csv",
    ):
        files.append(root / rel)
    for seed in (0, 1, 2):
        files.append(
            root
            / "data/processed"
            / f"vision_cifar100_clip-vit-base-patch32_p4000v800t2000_s{seed}.npz"
        )

    sklearn_data = Path(
        os.environ.get("SCIKIT_LEARN_DATA", str(Path.home() / "scikit_learn_data"))
    )
    files.extend(
        [
            sklearn_data / "openml/openml.org/api/v1/json/data/151.gz",
            sklearn_data / "openml/openml.org/data/v1/download/2419/electricity.arff.gz",
            Path.home() / ".cache/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
        ]
    )
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    for rel in (
        "hub/datasets--uoft-cs--cifar100/refs/main",
        "hub/models--openai--clip-vit-base-patch32/refs/main",
        "hub/models--amazon--chronos-bolt-tiny/refs/main",
    ):
        files.append(hf_home / rel)
    chronos_ref = hf_home / "hub/models--amazon--chronos-bolt-tiny/refs/main"
    if chronos_ref.exists():
        commit = chronos_ref.read_text().strip()
        for name in ("config.json", "model.safetensors"):
            files.append(
                hf_home
                / "hub/models--amazon--chronos-bolt-tiny/snapshots"
                / commit
                / name
            )

    mapping: dict[str, dict[str, Any]] = {}
    for path in sorted(set(files), key=str):
        key = str(path)
        if path.exists() and path.is_file():
            size = path.stat().st_size
            mapping[key] = {"sha256": sha256(path), "size": size}
            if size <= 1024:
                mapping[key]["content"] = path.read_text(errors="replace").strip()
        else:
            mapping[key] = {"missing": True}

    packages = {}
    for name in ("numpy", "scikit-learn", "torch", "transformers", "datasets", "tabpfn", "chronos-forecasting"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        pip_freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, timeout=60
        )
    except Exception as exc:
        pip_freeze = f"ERROR:{exc!r}"
    try:
        gpu_identity = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=30,
        ).strip()
    except Exception as exc:
        gpu_identity = f"ERROR:{exc!r}"
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "pip_freeze_sha256": hashlib.sha256(pip_freeze.encode()).hexdigest(),
        "gpu_identity": gpu_identity,
        "HF_HOME": os.environ.get("HF_HOME"),
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT"),
        "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
    }
    missing = sorted(path for path, meta in mapping.items() if meta.get("missing"))
    payload = {"files": mapping, "runtime": runtime, "missing": missing}
    return {**payload, "asset_sha256": json_sha(payload)}


def resolved_protocol(profile: dict[str, str | None], seed: int) -> dict[str, str | None]:
    protocol = {name: None for name in SCRUB_VARS}
    protocol.update(profile)
    protocol["SEED"] = str(seed)
    protocol["RUN_ID"] = "<derived-per-attempt>"
    return protocol


def clean_env(protocol: dict[str, str | None], run_id: str) -> dict[str, str]:
    env = dict(os.environ)
    for name in SCRUB_VARS:
        env.pop(name, None)
    for name, value in protocol.items():
        if name == "RUN_ID":
            continue
        if value is not None:
            env[name] = str(value)
    env["RUN_ID"] = run_id
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "4"
    return env


def expected_config(arm: str, protocol: dict[str, str | None]) -> dict[str, Any]:
    get = protocol.get
    if arm == "vision":
        return {
            "noise_frac": float(get("NOISE_FRAC") or "0.4"),
            "vis_noise": get("VIS_NOISE") or "inject",
            "drop": get("DROP_CHANNEL") or "",
            "pool": int(get("POOL_N") or "4000"),
            "budget": float(get("BUDGET_FRAC") or "0.5"),
            "val_n": int(get("VAL_N") or "800"),
        }
    if arm == "tep":
        return {
            "model": get("MODEL") or "rf",
            "noise_frac": float(get("NOISE_FRAC") or "0.4"),
            "drop": get("DROP_CHANNEL") or "",
            "pool": int(get("POOL_N") or "4000"),
            "budget": float(get("BUDGET_FRAC") or "0.3"),
        }
    if arm == "tabular":
        return {
            "model": get("MODEL") or "tabpfn",
            "noise_frac": float(get("NOISE_FRAC") or "0.4"),
            "pool": int(get("POOL_N") or "3000"),
            "budget": float(get("BUDGET_FRAC") or "0.5"),
        }
    if arm == "timeseries":
        return {
            "model": get("TS_MODEL") or "dlinear",
            "pool": int(get("POOL_N") or "3000"),
            "budget": float(get("BUDGET_FRAC") or "0.3"),
            "noise": float(get("NOISE_FRAC") or "0.4"),
            "L": int(get("L") or "96"),
            "H": int(get("H") or "24"),
        }
    raise ValueError(f"unknown arm {arm}")


def block_for(root: Path, logpath: str, headers: list[str], seed: int) -> str | None:
    path = root / logpath
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    for header in headers:
        pat = header.replace("{s}", str(seed))
        match = re.search(pat + r"(.*?)(?=##### |\Z)", text, re.DOTALL)
        if match:
            return match.group(1)
    return None


def old_evidence(root: Path, spec: dict[str, Any], seed: int) -> tuple[str | None, float | None, float | None]:
    body = block_for(root, spec["oldlog"], spec["headers"], seed)
    if body is None:
        return None, None, None
    picks = re.findall(r"picked '([^']+)'", body)
    vals = re.findall(VAL_PAT, body)
    patterns = {
        "test_acc": r"mmds_adapt\s+(?:n=\s*\d+ clean%=[0-9.]+ )?(?:test_)?acc=([0-9.]+)",
        "f1": r"mmds_adapt\s+F1=([0-9.]+)",
        "auc": r"mmds_adapt\s+auc=([0-9.]+)",
        "mase": r"mmds_adapt\s+(?:n=\s*\d+ clean%=[0-9.]+ )?MASE=([0-9.]+)",
    }
    tests = re.findall(patterns[spec["metric"]], body)
    return (
        picks[-1] if picks else None,
        float(vals[-1]) if vals else None,
        float(tests[-1]) if tests else None,
    )


def task_fingerprint(
    root: Path,
    spec: dict[str, Any],
    seed: int,
    code: dict[str, Any],
    assets: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    protocol = resolved_protocol(spec["env"], seed)
    runner = root / spec["script"]
    oldlog = root / spec["oldlog"]
    payload = {
        "version": VERSION,
        "scope": "current-formal-implementation",
        "tag": spec["tag"],
        "seed": seed,
        "protocol": protocol,
        "runner_sha256": sha256(runner),
        "code_tree_sha256": code["tree_sha256"],
        "asset_sha256": assets["asset_sha256"],
        "old_log_sha256": sha256(oldlog),
        "old_semantic_base": spec["semantic_base"],
        "metric": spec["metric"],
        "tolerance": TOL,
    }
    return json_sha(payload), payload


def validate_artifact(
    path: Path,
    spec: dict[str, Any],
    seed: int,
    protocol: dict[str, str | None],
    runner_sha: str,
    stdout: str,
) -> tuple[dict[str, bool], dict[str, Any]]:
    gates: dict[str, bool] = {}
    detail: dict[str, Any] = {}
    try:
        payload = json.loads(path.read_text())
        gates["artifact_json"] = True
    except Exception as exc:
        gates["artifact_json"] = False
        detail["artifact_error"] = repr(exc)
        return gates, detail

    gates["identity"] = (
        payload.get("arm") == spec["arm"]
        and str(payload.get("dataset")) == spec["dataset"]
        and payload.get("seed") == seed
    )
    gates["runner_code"] = payload.get("code_sha256_12") == runner_sha[:12]
    gates["config_exact"] = payload.get("config") == expected_config(spec["arm"], protocol)

    all_rows = payload.get("results", [])
    rows = [row for row in all_rows if row.get("method") == "mmds_adapt"] if isinstance(all_rows, list) else []
    gates["one_result_row"] = isinstance(all_rows, list) and len(all_rows) == 1 and len(rows) == 1
    metric_value = None
    if gates["one_result_row"]:
        raw = rows[0].get(METRIC_FIELD[spec["metric"]])
        if isinstance(raw, (int, float)):
            metric_value = float(raw)
    gates["test_metric_finite"] = metric_value is not None and math.isfinite(metric_value)
    if metric_value is None:
        gates["test_metric_range"] = False
    elif spec["metric"] == "mase":
        gates["test_metric_range"] = metric_value >= 0
    else:
        gates["test_metric_range"] = 0 <= metric_value <= 1
    expected_n = int(
        int(protocol.get("POOL_N") or "0")
        * float(protocol.get("BUDGET_FRAC") or "0")
    )
    gates["selected_n_exact"] = gates["one_result_row"] and rows[0].get("n") == expected_n

    manifest = payload.get("adapt_manifest")
    gates["manifest_dict"] = isinstance(manifest, dict)
    chosen = manifest.get("chosen") if isinstance(manifest, dict) else None
    leaderboard = manifest.get("leaderboard") if isinstance(manifest, dict) else None
    sel_sha = manifest.get("sel_sha12") if isinstance(manifest, dict) else None
    gates["chosen_dict"] = isinstance(chosen, dict)
    gates["leaderboard_nonempty"] = isinstance(leaderboard, list) and len(leaderboard) > 0
    leaderboard_valid = gates["leaderboard_nonempty"] and all(
        isinstance(item, list)
        and len(item) == 2
        and isinstance(item[0], str)
        and isinstance(item[1], (int, float))
        and math.isfinite(float(item[1]))
        for item in leaderboard
    )
    gates["leaderboard_finite"] = bool(leaderboard_valid)
    chosen_strategy = chosen.get("strategy") if isinstance(chosen, dict) else None
    chosen_val = chosen.get("val_gain") if isinstance(chosen, dict) else None
    gates["chosen_strategy"] = isinstance(chosen_strategy, str) and bool(chosen_strategy)
    gates["chosen_val_finite"] = isinstance(chosen_val, (int, float)) and math.isfinite(float(chosen_val))
    chosen_rows = (
        [item for item in leaderboard if item[0] == chosen_strategy]
        if leaderboard_valid and isinstance(chosen_strategy, str)
        else []
    )
    gates["chosen_in_leaderboard"] = (
        len(chosen_rows) == 1
        and gates["chosen_val_finite"]
        and abs(float(chosen_rows[0][1]) - float(chosen_val)) <= 1e-12
    )
    gates["selection_sha12"] = isinstance(sel_sha, str) and re.fullmatch(r"[0-9a-f]{12}", sel_sha) is not None

    picks = re.findall(r"picked '([^']+)'", stdout)
    gates["one_stdout_pick"] = len(picks) == 1
    stdout_pick = picks[0] if len(picks) == 1 else None
    gates["stdout_manifest_pick_match"] = stdout_pick is not None and stdout_pick == chosen_strategy

    detail.update(
        {
            "new_picked": chosen_strategy,
            "stdout_picked": stdout_pick,
            "new_val": float(chosen_val) if gates["chosen_val_finite"] else None,
            "new_test": metric_value,
            "sel_sha12": sel_sha,
            "artifact_config": payload.get("config"),
        }
    )
    return gates, detail


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": VERSION, "entries": {}}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {"version": VERSION, "entries": {}}
    if data.get("version") != VERSION or not isinstance(data.get("entries"), dict):
        return {"version": VERSION, "entries": {}}
    return data


def resume_valid(
    entry: dict[str, Any],
    root: Path,
    spec: dict[str, Any],
    seed: int,
    fingerprint: str,
    code_sha: str,
    asset_sha: str,
) -> bool:
    if not entry.get("gate_pass") or entry.get("fingerprint") != fingerprint:
        return False
    if entry.get("code_tree_sha256") != code_sha:
        return False
    if entry.get("asset_sha256") != asset_sha:
        return False
    for key, hash_key in (("artifact", "artifact_sha256"), ("stdout", "stdout_sha256"), ("stderr", "stderr_sha256")):
        path = entry.get(key)
        expected = entry.get(hash_key)
        if not path or not expected or not os.path.isfile(path) or sha256(path) != expected:
            return False
    stdout = Path(entry["stdout"]).read_text(errors="ignore")
    stderr = Path(entry["stderr"]).read_text(errors="ignore")
    if FORBIDDEN_PAT.search(stdout + "\n" + stderr) is not None:
        return False
    protocol = resolved_protocol(spec["env"], seed)
    runner_sha = sha256(root / spec["script"])
    gates, detail = validate_artifact(
        Path(entry["artifact"]), spec, seed, protocol, runner_sha, stdout
    )
    if not gates or not all(gates.values()):
        return False
    old_pick, old_val, old_test = old_evidence(root, spec, seed)
    new_pick = detail.get("new_picked")
    new_val = detail.get("new_val")
    new_test = detail.get("new_test")
    picked_match = new_pick == old_pick if new_pick is not None and old_pick is not None else False
    delta_val = abs(new_val - old_val) if new_val is not None and old_val is not None else None
    delta_test = abs(new_test - old_test) if new_test is not None and old_test is not None else None
    unchanged = (
        picked_match
        and delta_val is not None
        and delta_val <= TOL
        and delta_test is not None
        and delta_test <= TOL
    )
    return (
        entry.get("protocol") == protocol
        and entry.get("new_picked") == detail.get("new_picked")
        and entry.get("new_val") == detail.get("new_val")
        and entry.get("new_test") == detail.get("new_test")
        and entry.get("old_picked") == old_pick
        and entry.get("old_val") == old_val
        and entry.get("old_test") == old_test
        and entry.get("picked_match") == picked_match
        and entry.get("delta_val") == delta_val
        and entry.get("delta_test") == delta_test
        and entry.get("unchanged") == unchanged
        and entry.get("runner_code_sha256") == runner_sha
        and entry.get("returncode") == 0
    )


ACTIVE_PIDS: set[int] = set()
ACTIVE_LOCK = threading.Lock()
STOP_EVENT = threading.Event()
STOP_SCHEDULING = threading.Event()


def terminate_active(signum: int, _frame: Any) -> None:
    STOP_EVENT.set()
    STOP_SCHEDULING.set()
    with ACTIVE_LOCK:
        pids = list(ACTIVE_PIDS)
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    print(f"STOP_REQUESTED signal={signum}", flush=True)


def run_trial(
    root: Path,
    py: Path,
    logdir: Path,
    spec: dict[str, Any],
    seed: int,
    fingerprint: str,
    fingerprint_payload: dict[str, Any],
    code_before: dict[str, Any],
    assets_before: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    if STOP_EVENT.is_set():
        return {
            "version": VERSION,
            "config": spec["tag"],
            "seed": seed,
            "metric": spec["metric"],
            "fingerprint": fingerprint,
            "gate_pass": False,
            "stopped_before_launch": True,
        }
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec["tag"])
    attempt = f"{time.strftime('%Y%m%dT%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}"
    run_id = f"ctrlv5-{safe_tag}-s{seed}-{fingerprint[:12]}-{attempt}"
    outpath = logdir / f"{run_id}.out.log"
    errpath = logdir / f"{run_id}.err.log"
    protocol = fingerprint_payload["protocol"]
    env = clean_env(protocol, run_id)
    runner = root / spec["script"]
    started = time.time()
    timed_out = False
    rc = 125

    with open(outpath, "w") as out, open(errpath, "w") as err:
        with ACTIVE_LOCK:
            if STOP_EVENT.is_set():
                return {
                    "version": VERSION,
                    "config": spec["tag"],
                    "seed": seed,
                    "metric": spec["metric"],
                    "fingerprint": fingerprint,
                    "gate_pass": False,
                    "stopped_before_launch": True,
                }
            proc = subprocess.Popen(
                [str(py), "-u", str(runner)],
                cwd=root,
                env=env,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            ACTIVE_PIDS.add(proc.pid)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            rc = 124
        finally:
            with ACTIVE_LOCK:
                ACTIVE_PIDS.discard(proc.pid)

    stdout = outpath.read_text(errors="ignore")
    stderr = errpath.read_text(errors="ignore")
    dataset_dir = spec["dataset"].replace("/", "_")
    artifact_glob = root / "outputs" / spec["arm"] / dataset_dir / f"run_id={run_id}-*" / f"seed_{seed}" / "results.json"
    artifacts = [Path(path) for path in glob.glob(str(artifact_glob))]
    runner_sha = sha256(runner)
    artifact_gates: dict[str, bool] = {}
    detail: dict[str, Any] = {}
    if len(artifacts) == 1:
        artifact_gates, detail = validate_artifact(
            artifacts[0], spec, seed, protocol, runner_sha, stdout
        )

    old_pick, old_val, old_test = old_evidence(root, spec, seed)
    after_code = code_manifest(root)
    after_assets = asset_manifest(root)
    gates: dict[str, bool] = {
        "exit_zero": rc == 0,
        "no_timeout": not timed_out,
        "stdout_nonempty": bool(stdout.strip()),
        "logs_clean": FORBIDDEN_PAT.search(stdout + "\n" + stderr) is None,
        "one_unique_artifact": len(artifacts) == 1,
        "old_evidence_complete": old_pick is not None and old_val is not None and old_test is not None,
        "code_tree_stable": after_code["tree_sha256"] == code_before["tree_sha256"],
        "assets_stable": after_assets["asset_sha256"] == assets_before["asset_sha256"],
    }
    gates.update(artifact_gates)
    new_pick = detail.get("new_picked")
    new_val = detail.get("new_val")
    new_test = detail.get("new_test")
    picked_match = new_pick == old_pick if new_pick is not None and old_pick is not None else False
    delta_val = abs(new_val - old_val) if new_val is not None and old_val is not None else None
    delta_test = abs(new_test - old_test) if new_test is not None and old_test is not None else None
    unchanged = (
        picked_match
        and delta_val is not None
        and delta_val <= TOL
        and delta_test is not None
        and delta_test <= TOL
    )
    artifact = str(artifacts[0]) if len(artifacts) == 1 else ""
    entry = {
        "version": VERSION,
        "scope": "current-formal-implementation",
        "config": spec["tag"],
        "seed": seed,
        "arm": spec["arm"],
        "dataset": spec["dataset"],
        "metric": spec["metric"],
        "semantic_base": spec["semantic_base"],
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "run_id": run_id,
        "protocol": protocol,
        "gate": gates,
        "gate_pass": bool(gates) and all(gates.values()),
        "returncode": rc,
        "elapsed_seconds": round(time.time() - started, 3),
        "old_picked": old_pick,
        "new_picked": new_pick,
        "picked_match": picked_match,
        "old_val": old_val,
        "new_val": new_val,
        "delta_val": delta_val,
        "old_test": old_test,
        "new_test": new_test,
        "delta_test": delta_test,
        "unchanged": unchanged,
        "sel_sha12": detail.get("sel_sha12"),
        "runner_code_sha256": runner_sha,
        "code_tree_sha256": code_before["tree_sha256"],
        "asset_sha256": assets_before["asset_sha256"],
        "stdout": str(outpath),
        "stdout_sha256": sha256(outpath),
        "stderr": str(errpath),
        "stderr_sha256": sha256(errpath),
        "artifact": artifact,
        "artifact_sha256": sha256(artifact) if artifact else None,
        "artifact_detail": detail,
    }
    return entry


def marker_paths(marker_dir: Path) -> tuple[Path, Path]:
    return marker_dir / "CONTROLLER_REVALIDATED_OK", marker_dir / "REPLAY_COMPLETE_WITH_CHANGES"


def quarantine_markers(marker_dir: Path) -> None:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    for path in marker_paths(marker_dir):
        if path.exists():
            target = path.with_name(f"{path.name}.stale_{stamp}")
            if target.exists():
                target = path.with_name(f"{path.name}.stale_{stamp}_{time.time_ns()}")
            path.rename(target)


def adopted_marker_valid(ok_marker: Path, report_sha: str, canonical_path: Path) -> bool:
    if not ok_marker.exists() or not canonical_path.exists():
        return False
    try:
        marker = json.loads(ok_marker.read_text())
        canonical = json.loads(canonical_path.read_text())
    except Exception:
        return False
    return (
        marker.get("report_sha256") == report_sha
        and marker.get("canonical") == str(canonical_path)
        and marker.get("canonical_sha256") == sha256(canonical_path)
        and canonical.get("report_sha256") == report_sha
        and len(canonical.get("rows", [])) == len(PLAN) * 3
    )


def finalise(
    root: Path,
    marker_dir: Path,
    report_path: Path,
    canonical_path: Path,
    report: dict[str, Any],
    expected_total: int,
    adopt_current_sha: str = "",
) -> int:
    entries = report["replays"]
    gate_pass = sum(bool(entry.get("gate_pass")) for entry in entries)
    unchanged = sum(bool(entry.get("unchanged")) for entry in entries)
    report["summary"] = {
        "expected": expected_total,
        "present": len(entries),
        "gate_pass": gate_pass,
        "unchanged_vs_historical_log": unchanged,
        "changed_vs_historical_log": len(entries) - unchanged,
        "scope": "current-formal-implementation",
    }
    atomic_json(report_path, report)
    print(json.dumps(report["summary"], sort_keys=True), flush=True)
    report_sha = sha256(report_path)
    ok_marker, changed_marker = marker_paths(marker_dir)
    if len(entries) != expected_total or gate_pass != expected_total:
        quarantine_markers(marker_dir)
        print("CONTROLLER_INCOMPLETE_NO_MARKER", flush=True)
        return 3

    if unchanged != expected_total and adopt_current_sha != report_sha:
        if adopted_marker_valid(ok_marker, report_sha, canonical_path):
            if changed_marker.exists():
                changed_marker.rename(
                    changed_marker.with_name(
                        f"{changed_marker.name}.adopted_{time.strftime('%Y%m%dT%H%M%S')}"
                    )
                )
            print("CONTROLLER_REVALIDATED_OK_ALREADY_ADOPTED", flush=True)
            return 0
        if ok_marker.exists():
            quarantine_markers(marker_dir)
        marker_payload = {
            **report["summary"],
            "report": str(report_path),
            "report_sha256": report_sha,
            "adoption_required": True,
        }
        atomic_text(changed_marker, json.dumps(marker_payload, sort_keys=True) + "\n")
        print(
            f"REPLAY_COMPLETE_WITH_CHANGES report_sha256={report_sha} "
            "explicit diff audit and --adopt-current-sha are required",
            flush=True,
        )
        return 2

    canonical = {
        "version": VERSION,
        "scope": "current-formal-implementation",
        "adoption_note": (
            "These rows are the canonical current-code controller results. Historical-log "
            "differences are impact evidence only and are not causally attributed to one patch."
        ),
        "report": str(report_path),
        "report_sha256": report_sha,
        "rows": entries,
    }
    atomic_json(canonical_path, canonical)
    marker_payload = {
        **report["summary"],
        "report": str(report_path),
        "report_sha256": report_sha,
        "canonical": str(canonical_path),
        "canonical_sha256": sha256(canonical_path),
    }
    atomic_text(ok_marker, json.dumps(marker_payload, sort_keys=True) + "\n")
    if changed_marker.exists():
        changed_marker.rename(
            changed_marker.with_name(
                f"{changed_marker.name}.adopted_{time.strftime('%Y%m%dT%H%M%S')}"
            )
        )
    print("CONTROLLER_REVALIDATED_OK", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    marker_dir = Path(args.marker_dir).resolve()
    py = root / ".venv/bin/python"
    logdir = root / "experiments/controller_reval_v5"
    checkpoint_path = logdir / "checkpoint.json"
    report_path = root / "experiments/controller_reval_report_v5.json"
    canonical_path = root / "experiments/controller_current_canonical_v5.json"
    lock_path = marker_dir / "controller_reval_v5.lock"

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if any(seed not in (0, 1, 2) for seed in seeds):
        raise SystemExit("seeds must be drawn from 0,1,2")
    tags = {value for value in args.tags.split(",") if value}
    unknown = tags - {spec["tag"] for spec in PLAN}
    if unknown:
        raise SystemExit(f"unknown tags: {sorted(unknown)}")

    selected_specs = [spec for spec in PLAN if not tags or spec["tag"] in tags]
    selected_tasks = [(spec, seed) for seed in seeds for spec in selected_specs]
    if args.max_trials > 0:
        selected_tasks = selected_tasks[: args.max_trials]

    gaps = []
    for spec, seed in selected_tasks:
        old = old_evidence(root, spec, seed)
        if any(value is None for value in old):
            gaps.append({"tag": spec["tag"], "seed": seed, "old": old})
    if args.dry_run:
        print(json.dumps({"selected": len(selected_tasks), "gaps": gaps, "excluded": EXCLUDED}, indent=2))
        print("DRY_RUN_OK" if not gaps else "DRY_RUN_GAPS")
        return 0 if not gaps else 1
    if gaps:
        print(json.dumps(gaps, indent=2))
        print("OLD_EVIDENCE_GAPS_NO_RUN")
        return 1
    if not py.exists():
        print(f"missing interpreter {py}")
        return 1
    if Path(sys.executable).resolve() != py.resolve():
        print(f"wrong orchestrator interpreter {sys.executable}, expected {py}")
        return 1
    if args.jobs < 1 or args.jobs > 2:
        print("jobs must be 1 or 2")
        return 1
    d_marker = marker_dir / "D_VALIDATED_OK"
    d_log = marker_dir / "D_validate.log"
    expected_d = "stratify=1-infl=pplq-train=finetune-lmeval=1"
    d_log_text = d_log.read_text(errors="ignore") if d_log.exists() else ""
    if (
        not d_marker.exists()
        or d_marker.read_text().strip() != expected_d
        or "D_VALIDATION_OK" not in d_log_text
        or "d_validator_exit=0" not in d_log_text
        or "[FAIL]" in d_log_text
    ):
        print("D validation evidence invalid, refusing controller launch")
        return 1

    marker_dir.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another controller_reval_v5 instance holds the lock")
        os.close(lock_fd)
        return 1

    signal.signal(signal.SIGTERM, terminate_active)
    signal.signal(signal.SIGINT, terminate_active)
    logdir.mkdir(parents=True, exist_ok=True)
    code = code_manifest(root)
    assets = asset_manifest(root)
    if assets["missing"]:
        print(f"ASSET_MANIFEST_MISSING {assets['missing']}")
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        return 1
    checkpoint = load_checkpoint(checkpoint_path)
    entries_by_key = checkpoint["entries"]

    prepared = []
    resumed = []
    missing_for_validation = []
    for spec, seed in selected_tasks:
        fingerprint, payload = task_fingerprint(root, spec, seed, code, assets)
        old = entries_by_key.get(fingerprint)
        if old and resume_valid(
            old,
            root,
            spec,
            seed,
            fingerprint,
            code["tree_sha256"],
            assets["asset_sha256"],
        ):
            resumed.append(old)
            print(f"[resume] {spec['tag']} seed{seed}", flush=True)
        elif args.validate_only:
            missing_for_validation.append((spec["tag"], seed))
        else:
            prepared.append((spec, seed, fingerprint, payload))

    completed = list(resumed)
    if args.validate_only and missing_for_validation:
        print(f"VALIDATE_ONLY_MISSING_TRIALS {missing_for_validation}")
        quarantine_markers(marker_dir)

    if prepared:
        quarantine_markers(marker_dir)

    if prepared:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs)
        future_map: dict[concurrent.futures.Future[Any], tuple[dict[str, Any], int, str]] = {}
        work = iter(prepared)

        def submit_one() -> bool:
            if STOP_EVENT.is_set() or STOP_SCHEDULING.is_set():
                return False
            try:
                spec, seed, fingerprint, payload = next(work)
            except StopIteration:
                return False
            future = pool.submit(
                run_trial,
                root,
                py,
                logdir,
                spec,
                seed,
                fingerprint,
                payload,
                code,
                assets,
                args.timeout,
            )
            future_map[future] = (spec, seed, fingerprint)
            return True

        for _ in range(args.jobs):
            submit_one()
        try:
            while future_map:
                done, _ = concurrent.futures.wait(
                    future_map, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    spec, seed, fingerprint = future_map.pop(future)
                    try:
                        entry = future.result()
                    except Exception as exc:
                        entry = {
                            "version": VERSION,
                            "config": spec["tag"],
                            "seed": seed,
                            "metric": spec["metric"],
                            "fingerprint": fingerprint,
                            "gate_pass": False,
                            "orchestrator_exception": repr(exc),
                        }
                    entries_by_key[fingerprint] = entry
                    checkpoint = {"version": VERSION, "entries": entries_by_key}
                    atomic_json(checkpoint_path, checkpoint)
                    completed.append(entry)
                    print(
                        f"[trial] {spec['tag']} seed{seed} gate={entry.get('gate_pass')} "
                        f"elapsed={entry.get('elapsed_seconds')} changed={not entry.get('unchanged', False)}",
                        flush=True,
                    )
                    if not entry.get("gate_pass"):
                        STOP_SCHEDULING.set()
                if not STOP_SCHEDULING.is_set():
                    while len(future_map) < args.jobs and submit_one():
                        pass
        finally:
            for future in future_map:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)

    all_entries = []
    invalid_entries = []
    for spec in PLAN:
        for seed in (0, 1, 2):
            fingerprint, _ = task_fingerprint(root, spec, seed, code, assets)
            entry = entries_by_key.get(fingerprint)
            if entry and resume_valid(
                entry,
                root,
                spec,
                seed,
                fingerprint,
                code["tree_sha256"],
                assets["asset_sha256"],
            ):
                all_entries.append(entry)
            elif entry:
                invalid_entries.append(entry)

    all_entries.sort(key=lambda item: ({spec["tag"]: i for i, spec in enumerate(PLAN)}[item["config"]], item["seed"]))
    report = {
        "version": VERSION,
        "scope": "current-formal-implementation",
        "checkpoint_sha256": (
            sha256(checkpoint_path) if checkpoint_path.exists() else json_sha(checkpoint)
        ),
        "comparison_limit": (
            "Historical logs span runner, environment, hardware, and portfolio revisions. "
            "Differences are not attributed solely to the finite-aware selector patch."
        ),
        "code_manifest": code,
        "asset_manifest": assets,
        "excluded": EXCLUDED,
        "plan": [{key: value for key, value in spec.items() if key != "env"} for spec in PLAN],
        "replays": all_entries,
        "invalid_replays": invalid_entries,
    }
    rc = finalise(
        root,
        marker_dir,
        report_path,
        canonical_path,
        report,
        len(PLAN) * 3,
        adopt_current_sha=args.adopt_current_sha,
    )
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    os.close(lock_fd)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
