"""Tiny dependency-light I/O helpers (jsonl / json / yaml).

Kept free of any method logic so the core package imports without torch/transformers.
"""
from __future__ import annotations

import json
import os
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
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def read_yaml(path: str) -> Any:
    import yaml  # local import: yaml is a core dep but keep import sites obvious

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(obj: Any, path: str) -> str:
    import yaml

    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)
    return path
