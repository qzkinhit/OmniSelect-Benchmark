"""Batch index: results_and_logs/<batch>/index.jsonl with one JSON line per finished cell.

Lines are appended under an exclusive ``fcntl`` lock so concurrent cells on one host never
interleave. Each line holds the cell path, status, elapsed seconds, elected strategy and its
test utility. ``read_index`` returns the lines as dicts and skips blank lines.
"""
from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path
from typing import Any, Mapping


def append_index(batch_dir: Path | str, record: Mapping[str, Any]) -> Path:
    """Append ``record`` plus a timestamp to <batch_dir>/index.jsonl and return the path."""
    path = Path(batch_dir) / "index.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **dict(record)}, allow_nan=False)
    with open(path, "a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.write(line + "\n")
            handle.flush()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    return path


def read_index(batch_dir: Path | str) -> list[dict[str, Any]]:
    """All index lines of a batch directory, oldest first."""
    path = Path(batch_dir) / "index.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
