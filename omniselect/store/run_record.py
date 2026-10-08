"""Run record writer: one directory per cell holding everything needed to recompute and replay.

Layout (docs/RUN_RECORD.md): config.json, splits.json, signals/*.npy and
reference_sample_ids.json, candidates/<name>/{selection.npz, per_unit/<split>.npz, scores.json,
model/}, leaderboard.json, decision.json, timings.json, metrics.json, log.txt. Every file is
written atomically. decision.json is written last and marks the cell complete for resume.
"""
from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from omniselect.store.index import append_index
from omniselect.store.metrics import standard_metrics
from omniselect.utils.hashing import ids_sha256, sel_sha12

SCHEMA_VERSION = "omniselect.run-record.v1"
SPLITS = ("con", "rank", "conf", "test")
_SAFE = re.compile(r"[^A-Za-z0-9_.=+-]+")


def safe_name(name: str) -> str:
    """Directory name for a candidate: unsafe characters become '_', repeats collapse."""
    cleaned = _SAFE.sub("_", name).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned or "candidate"


def jsonable(value: Any) -> Any:
    """Convert numpy scalars and arrays, Paths and tuples to JSON types. Non-finite floats raise."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"non-finite value in run record: {value}")
    return value


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON through a temporary file in the same directory and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(jsonable(payload), handle, indent=2, allow_nan=False, sort_keys=False)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    """np.savez_compressed through a temporary file. Object arrays are refused."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clean: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.dtype == object:
            array = array.astype(str)
        clean[key] = array
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **clean)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_npy(path: Path, array: np.ndarray) -> None:
    """np.save through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npy")
    os.close(fd)
    try:
        np.save(tmp, np.asarray(array))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def git_sha(repo: Path) -> dict[str, Any]:
    """HEAD sha and whether tracked files differ from it."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
                               capture_output=True, text=True, check=True).stdout.strip() != ""
        return {"sha": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}


def environment() -> dict[str, Any]:
    """Python, platform, hostname, package versions and the visible accelerator."""
    import importlib.metadata as md

    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "transformers", "datasets", "tabpfn",
                 "chronos-forecasting", "xgboost", "lm-eval"):
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = {"cuda": torch.version.cuda, "name": torch.cuda.get_device_name(0),
                   "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            gpu = {"mps": True}
    except ImportError:
        gpu = None
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "hostname": socket.gethostname(), "packages": versions, "accelerator": gpu, "threads": threads()}


THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def threads() -> dict[str, Any]:
    """Thread settings recorded next to the timings (the values are read, never set).

    The three BLAS and OpenMP variables as raw strings or null, torch.get_num_threads() (null
    without torch), os.cpu_count() and the 1-minute load average (null where os.getloadavg is
    unavailable).
    """
    out: dict[str, Any] = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    try:
        import torch

        out["torch_num_threads"] = int(torch.get_num_threads())
    except ImportError:
        out["torch_num_threads"] = None
    out["cpu_count"] = os.cpu_count()
    try:
        out["load_avg_1min"] = float(os.getloadavg()[0])
    except (AttributeError, OSError):
        out["load_avg_1min"] = None
    return out


class RunRecord:
    """Writer for one cell directory."""

    def __init__(self, cell_dir: Path | str, *, repo_root: Path | str | None = None) -> None:
        self.dir = Path(cell_dir)
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
        self.started = time.time()
        self._candidate_dirs: dict[str, str] = {}

    @staticmethod
    def cell_path(root: Path | str, batch: str, track: str, dataset: str, learner: str, seed: int) -> Path:
        """results_and_logs/<batch>/<track>/<dataset>/<learner>/seed_<s>."""
        return Path(root) / batch / track / safe_name(dataset.replace("/", "_")) / safe_name(learner) / f"seed_{seed}"

    def is_complete(self) -> bool:
        """True when decision.json exists (the queue skips such cells)."""
        return (self.dir / "decision.json").is_file()

    def prepare(self, overwrite: bool = False) -> None:
        """Create the directory. Refuse to write into a complete cell unless ``overwrite``."""
        if self.is_complete() and not overwrite:
            raise FileExistsError(f"cell already complete: {self.dir}")
        self.dir.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        """Append a timestamped line to log.txt and echo it to stdout."""
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with open(self.dir / "log.txt", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def write_config(self, *, omniselect: Mapping[str, Any], track: Mapping[str, Any],
                     cli: Sequence[str], extra: Optional[Mapping[str, Any]] = None) -> None:
        """config.json: resolved configs, CLI, git sha, environment and start time."""
        atomic_write_json(self.dir / "config.json", {
            "schema_version": SCHEMA_VERSION,
            "omniselect": omniselect,
            "track": track,
            "cli": list(cli),
            "git": git_sha(self.repo_root),
            "environment": environment(),
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.started)),
            **(dict(extra) if extra else {}),
        })

    def add_config_fields(self, fields: Mapping[str, Any]) -> None:
        """Merge top-level ``fields`` into config.json (settings known only after loading, such as encoders)."""
        path = self.dir / "config.json"
        payload = json.loads(path.read_text())
        payload.update(dict(fields))
        atomic_write_json(path, payload)

    def finalize_config(self) -> None:
        """Add the end time to config.json."""
        path = self.dir / "config.json"
        payload = json.loads(path.read_text())
        payload["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        payload["elapsed_secs"] = time.time() - self.started
        atomic_write_json(path, payload)

    def write_splits(self, *, ids: Mapping[str, Sequence[Any]], pool_tags: Sequence[str],
                     pool_mechanisms: Optional[Sequence[str]] = None,
                     blocks: Optional[Mapping[str, Any]] = None,
                     extra: Optional[Mapping[str, Any]] = None) -> None:
        """splits.json: id lists (pool, con, rank, conf, test), their sha256, per-record tags."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "ids": {name: list(values) for name, values in ids.items()},
            "sha256": {name: ids_sha256(list(values)) for name, values in ids.items()},
            "counts": {name: len(values) for name, values in ids.items()},
            "pool_tags": list(pool_tags),
            "pool_mechanisms": list(pool_mechanisms) if pool_mechanisms is not None else list(pool_tags),
            "time_blocks": dict(blocks) if blocks else None,
        }
        if extra:
            payload.update(extra)
        atomic_write_json(self.dir / "splits.json", payload)

    def write_signals(self, arrays: Mapping[str, np.ndarray], reference_ids: Sequence[Any],
                      reference_source: str) -> None:
        """signals/<name>.npy per pool record and signals/reference_sample_ids.json."""
        for name, array in arrays.items():
            atomic_write_npy(self.dir / "signals" / f"{name}.npy", np.asarray(array))
        atomic_write_json(self.dir / "signals" / "reference_sample_ids.json",
                          {"source": reference_source, "ids": list(reference_ids)})

    def candidate_dir(self, name: str) -> Path:
        """Directory of candidate ``name``. Distinct names never share a directory."""
        if name not in self._candidate_dirs:
            base = safe_name(name)
            used = set(self._candidate_dirs.values())
            candidate = base
            suffix = 1
            while candidate in used:
                suffix += 1
                candidate = f"{base}__{suffix}"
            self._candidate_dirs[name] = candidate
        return self.dir / "candidates" / self._candidate_dirs[name]

    def write_candidate(
        self,
        name: str,
        *,
        selection: Sequence[int],
        n_pool: int,
        budget: int,
        role: str,
        stage: str,
        selection_secs: float,
        per_unit: Mapping[str, Mapping[str, np.ndarray]],
        scores: Mapping[str, Any],
        weights: Optional[np.ndarray] = None,
        scoring_per_unit: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None,
        write_per_unit: bool = True,
    ) -> Path:
        """selection.npz, per_unit/<split>.npz, scores.json (utilities, metrics, timings, fidelity).

        With ``write_per_unit`` false the per-unit arrays only feed the standard metrics of scores.json
        and no per_unit/ file is written (scalar-only candidate).
        """
        cdir = self.candidate_dir(name)
        idx = np.asarray(sorted(int(i) for i in selection), dtype=np.int64)
        if len(np.unique(idx)) != len(idx) or (len(idx) and (idx[0] < 0 or idx[-1] >= n_pool)):
            raise ValueError(f"{name}: selection must be unique ids in [0, {n_pool})")
        arrays: dict[str, Any] = {
            "idx": idx, "n": np.int64(n_pool), "budget": np.int64(budget),
            "sel_sha12": np.array(sel_sha12(idx)), "selection_secs": np.float64(selection_secs),
            "role": np.array(role), "stage": np.array(stage), "name": np.array(name),
        }
        if weights is not None:
            arrays["weights"] = np.asarray(weights, dtype=np.float64)
        atomic_write_npz(cdir / "selection.npz", arrays)
        metric_values: dict[str, dict[str, float]] = {}
        for split, pu in per_unit.items():
            if write_per_unit:
                atomic_write_npz(cdir / "per_unit" / f"{split}.npz", pu)
            metric_values[split] = standard_metrics(pu)
        for split, pu in (scoring_per_unit or {}).items():
            if write_per_unit:
                atomic_write_npz(cdir / "per_unit" / f"scoring_{split}.npz", pu)
        payload = {"name": name, "role": role, "stage": stage, "sel_sha12": sel_sha12(idx),
                   "n_selected": int(len(idx)), "metrics": metric_values, "per_unit_written": bool(write_per_unit),
                   **dict(scores)}
        atomic_write_json(cdir / "scores.json", payload)
        return cdir

    def write_json(self, filename: str, payload: Any) -> None:
        """Atomic JSON write of a top-level cell file (leaderboard, timings, metrics, decision)."""
        atomic_write_json(self.dir / filename, payload)

    def candidate_index(self) -> dict[str, str]:
        """Mapping candidate name -> directory name under candidates/."""
        return dict(self._candidate_dirs)

    def finish(self, *, decision: Mapping[str, Any], index_root: Path | str, status: str,
               elected: str, test: Optional[float], extra: Optional[Mapping[str, Any]] = None) -> None:
        """Write decision.json (the completion marker) and append the batch index line."""
        self.finalize_config()
        self.write_json("decision.json", decision)
        append_index(Path(index_root), {
            "path": str(self.dir),
            "status": status,
            "elapsed_secs": round(time.time() - self.started, 3),
            "elected": elected,
            "test": test,
            **(dict(extra) if extra else {}),
        })
