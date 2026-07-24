"""Content-addressed experiment evidence bundles.

The experiment JSON is the summary, not the evidence.  This module persists the
material needed to replay a run: exact selections, the selected training data,
split/config/runtime provenance, downstream predictions, optional checkpoints,
and SHA-256 hashes for every artifact.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "omniselect.repro-bundle.v1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(value), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unnamed"


def save_downstream_checkpoint(
    model: Any,
    directory: os.PathLike[str] | str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Persist a fitted sklearn/XGBoost/PyTorch-style downstream model."""
    root = Path(directory).resolve()
    if root.exists():
        raise FileExistsError(f"checkpoint directory already exists: {root}")
    root.mkdir(parents=True)
    serializer = ""
    if hasattr(model, "save_model") and callable(model.save_model):
        model.save_model(str(root / "model.ubj"))
        serializer = "native-save_model"
    else:
        torch_model = model
        try:
            import torch

            if not isinstance(torch_model, torch.nn.Module) and isinstance(
                getattr(model, "model", None), torch.nn.Module
            ):
                torch_model = model.model
            if isinstance(torch_model, torch.nn.Module):
                torch.save(torch_model.state_dict(), root / "state_dict.pt")
                serializer = "torch-state-dict"
            else:
                raise TypeError
        except (ImportError, TypeError):
            import joblib

            joblib.dump(model, root / "model.joblib", compress=3)
            serializer = "joblib"
    _atomic_json(
        root / "checkpoint.json",
        {
            "serializer": serializer,
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "metadata": metadata or {},
        },
    )
    return root


def _git_manifest(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo_root), *args],
                stderr=subprocess.DEVNULL,
                text=True,
            ).rstrip("\n")
        except (OSError, subprocess.CalledProcessError):
            return ""

    status = run("status", "--porcelain=v1", "--untracked-files=all")
    diff = run("diff", "--binary", "--no-ext-diff", "HEAD")
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "status_porcelain": status.splitlines(),
    }


def _environment_manifest() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch

        result.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_runtime": getattr(torch.version, "cuda", None),
                "cudnn": (
                    int(torch.backends.cudnn.version())
                    if torch.backends.cudnn.is_available()
                    else None
                ),
                "gpu": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - torch is optional for utilities
        result["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import transformers

        result["transformers"] = transformers.__version__
    except Exception:
        pass
    return result


def _write_selected_npz(
    path: Path, source: Mapping[str, Any], selected: np.ndarray
) -> None:
    arrays: dict[str, np.ndarray] = {"selected_indices": selected}
    for name, value in source.items():
        array = np.asarray(value)
        if array.ndim == 0:
            continue
        if len(array) <= int(selected.max(initial=-1)):
            raise ValueError(f"selection source {name!r} is shorter than selected indices")
        arrays[str(name)] = array[selected]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_selected_jsonl(path: Path, records: Sequence[Any], selected: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for index in selected:
                record = records[int(index)]
                row = record.to_dict() if hasattr(record, "to_dict") else record
                handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_repro_bundle(
    output_dir: os.PathLike[str] | str,
    *,
    repo_root: os.PathLike[str] | str,
    runner_path: os.PathLike[str] | str,
    arm: str,
    dataset: str,
    seed: int,
    config: Mapping[str, Any],
    result_path: os.PathLike[str] | str,
    selections: Mapping[str, Sequence[int]],
    selection_source: Mapping[str, Any] | None = None,
    text_records: Sequence[Any] | None = None,
    predictions: Mapping[str, Mapping[str, Any]] | None = None,
    split_manifest: Mapping[str, Any] | None = None,
    evaluation_data: Mapping[str, Any] | None = None,
    checkpoint_paths: Mapping[str, os.PathLike[str] | str] | None = None,
    input_paths: Mapping[str, os.PathLike[str] | str] | None = None,
) -> Path:
    """Write and hash one complete, non-canonical evidence bundle."""
    root = Path(output_dir).resolve() / "repro_bundle"
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(repo_root).resolve()
    result = Path(result_path).resolve()
    runner = Path(runner_path).resolve()

    _atomic_json(
        root / "run.json",
        {
            "schema_version": SCHEMA_VERSION,
            "arm": arm,
            "dataset": dataset,
            "seed": int(seed),
            "config": config,
            "config_sha256": canonical_sha256(config),
            "result_path": str(result),
        },
    )
    _atomic_json(root / "environment.json", _environment_manifest())
    _atomic_json(root / "git.json", _git_manifest(repo))
    _atomic_json(root / "splits.json", split_manifest or {})
    if evaluation_data:
        np.savez_compressed(
            root / "evaluation_data.npz",
            **{name: np.asarray(value) for name, value in evaluation_data.items()},
        )

    input_manifest: dict[str, Any] = {
        "runner": {
            "path": str(runner),
            "sha256": file_sha256(runner),
            "bytes": runner.stat().st_size,
        }
    }
    for name, raw_path in sorted((input_paths or {}).items()):
        path = Path(raw_path).resolve()
        input_manifest[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    _atomic_json(root / "inputs.json", input_manifest)

    for method, raw_selection in selections.items():
        selected = np.asarray([int(index) for index in raw_selection], dtype=np.int64)
        if len(selected) != len(np.unique(selected)):
            raise ValueError(f"{method}: selected indices are not unique")
        name = _safe_name(method)
        ordered = selected.tolist()
        selection_payload = {
            "method": method,
            "selected_indices": ordered,
            "count": len(ordered),
            "ordered_sha256": canonical_sha256(ordered),
            "set_sha256": canonical_sha256(sorted(ordered)),
        }
        _atomic_json(root / "selections" / f"{name}.json", selection_payload)
        if selection_source is not None:
            _write_selected_npz(
                root / "selected_data" / f"{name}.npz", selection_source, selected
            )
        if text_records is not None:
            _write_selected_jsonl(
                root / "selected_data" / f"{name}.jsonl", text_records, ordered
            )

    for method, values in sorted((predictions or {}).items()):
        path = root / "predictions" / f"{_safe_name(method)}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{key: np.asarray(value) for key, value in values.items()})

    checkpoints: dict[str, Any] = {}
    checkpoint_files: list[Path] = []
    for method, raw_path in sorted((checkpoint_paths or {}).items()):
        path = Path(raw_path).resolve()
        files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
        checkpoint_files.extend(files)
        checkpoints[method] = {
            "path": str(path),
            "files": {
                str(item.relative_to(path if path.is_dir() else path.parent)): {
                    "sha256": file_sha256(item),
                    "bytes": item.stat().st_size,
                }
                for item in files
            },
        }
    _atomic_json(root / "checkpoints.json", checkpoints)

    artifact_files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name not in {"artifact_manifest.json", "SHA256SUMS"}
    )
    artifacts = {
        str(path.relative_to(root)): {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in artifact_files
    }
    artifacts["../results.json"] = {
        "sha256": file_sha256(result),
        "bytes": result.stat().st_size,
    }
    for path in checkpoint_files:
        relative = os.path.relpath(path, root)
        artifacts[relative] = {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    _atomic_json(root / "artifact_manifest.json", manifest)
    sums = root / "SHA256SUMS"
    sums.write_text(
        "".join(f"{meta['sha256']}  {name}\n" for name, meta in sorted(artifacts.items())),
        encoding="utf-8",
    )
    return root


def validate_repro_bundle(bundle_root: os.PathLike[str] | str) -> list[str]:
    root = Path(bundle_root).resolve()
    errors: list[str] = []
    try:
        manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"manifest unreadable: {type(exc).__name__}: {exc}"]
    for relative, expected in manifest.get("artifacts", {}).items():
        path = (root / relative).resolve()
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        actual = file_sha256(path)
        if actual != expected.get("sha256"):
            errors.append(f"sha256 mismatch: {relative}")
        if path.stat().st_size != expected.get("bytes"):
            errors.append(f"size mismatch: {relative}")
    return errors
