"""Selection manifests: manifest.json and selected.jsonl under <out_dir>/manifests/.

``write_manifest`` writes the experiment id, method, pool size, selected ids and optional extras,
and the selected records as JSON lines. ``read_manifest`` and ``read_selected`` read them back.
The run records of tracks/common/experiment.py supersede this format for benchmark cells.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from omniselect.utils.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl

MANIFEST_NAME = "manifest.json"
SELECTED_NAME = "selected.jsonl"


def write_manifest(
    out_dir: str,
    *,
    experiment_id: str,
    method: str,
    n_total: int,
    selected_ids: Sequence[str],
    selected_rows: Optional[Iterable[Dict[str, Any]]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write ``manifest.json`` (+ optional ``selected.jsonl``) and return the manifest."""
    mdir = ensure_dir(os.path.join(out_dir, "manifests"))
    manifest: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "method": method,
        "n_total": int(n_total),
        "n_selected": len(list(selected_ids)),
        "selected_ids": list(selected_ids),
    }
    if extra:
        manifest.update(extra)
    write_json(manifest, os.path.join(mdir, MANIFEST_NAME))
    if selected_rows is not None:
        write_jsonl(selected_rows, os.path.join(mdir, SELECTED_NAME))
    return manifest


def read_manifest(out_dir: str) -> Dict[str, Any]:
    return read_json(os.path.join(out_dir, "manifests", MANIFEST_NAME))


def read_selected(out_dir: str) -> List[Dict[str, Any]]:
    return list(read_jsonl(os.path.join(out_dir, "manifests", SELECTED_NAME)))
