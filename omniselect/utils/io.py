"""Atomic JSON, JSON-lines and YAML helpers.

``write_json``, ``write_jsonl`` and ``write_yaml`` write through a temporary file in the target
directory and ``os.replace``, refusing NaN in JSON. ``read_json``, ``read_jsonl`` and ``read_yaml``
read them back. ``ensure_dir`` creates a directory. The module imports no model code.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Iterable, Iterator


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> int:
    parent = ensure_dir(os.path.dirname(os.path.abspath(path)))
    fd, temporary = tempfile.mkstemp(dir=parent, suffix=".tmp")
    n = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(
                    json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                )
                n += 1
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return n


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str) -> str:
    parent = ensure_dir(os.path.dirname(os.path.abspath(path)))
    fd, temporary = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def read_yaml(path: str) -> Any:
    import yaml  # local import: yaml is a core dep but keep import sites obvious

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(obj: Any, path: str) -> str:
    import yaml

    parent = ensure_dir(os.path.dirname(os.path.abspath(path)))
    fd, temporary = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path
