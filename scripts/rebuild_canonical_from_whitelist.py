#!/usr/bin/env python3
"""Audit 1910-3: prove a clean clone can rebuild the paper's FINAL four-arm tables
from the whitelisted small results in git (results_canonical/) alone.

Procedure:
  1. Build a temp OMNISELECT_ROOT: symlink every results_canonical/**/results.json
     into <root>/outputs/... (results_canonical mirrors the outputs/ layout without
     the outputs/ prefix; any ettm2_oneshot subtree is excluded), and copy
     experiments/results_matrix.json + experiments/controller_current_canonical_v5.json
     into <root>/experiments/.
  2. Run scripts/canonical_paper_tables.py with OMNISELECT_ROOT=<root>.
  3. Compare the regenerated <root>/experiments/canonical_tables.json against the
     committed experiments/canonical_tables.json field by field on the FINAL keys:
     FINAL_main_table_source, final_cells, latex_FINAL_main_table,
     latex_FINAL_external_table. Exit 0 only if every leaf is exactly equal.

Stdlib only. Never writes to the committed experiments/canonical_tables.json.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHITELIST = os.path.join(REPO, "results_canonical")
COMMITTED = os.path.join(REPO, "experiments", "canonical_tables.json")
TABLE_SCRIPT = os.path.join(REPO, "scripts", "canonical_paper_tables.py")
EXPERIMENT_FILES = ("results_matrix.json", "controller_current_canonical_v5.json")
COMPARE_KEYS = ("FINAL_main_table_source", "final_cells",
                "latex_FINAL_main_table", "latex_FINAL_external_table")
EXCLUDE_SUBTREE = "ettm2_oneshot"


def populate(root):
    """Restore the outputs/ layout the canonical script's globs expect."""
    linked = 0
    for dirpath, dirnames, filenames in os.walk(WHITELIST):
        dirnames[:] = [d for d in dirnames if EXCLUDE_SUBTREE not in d]
        if EXCLUDE_SUBTREE in dirpath:
            continue
        rel = os.path.relpath(dirpath, WHITELIST)
        for fn in filenames:
            if fn != "results.json":
                continue  # README.md etc. are not glob targets
            src = os.path.join(dirpath, fn)
            dst = os.path.join(root, "outputs", rel, fn)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            linked += 1
    os.makedirs(os.path.join(root, "experiments"), exist_ok=True)
    for fn in EXPERIMENT_FILES:
        src = os.path.join(REPO, "experiments", fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(root, "experiments", fn))
    return linked


def deep_diff(a, b, path, out):
    """Record every leaf-level difference between a (rebuilt) and b (committed)."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = "%s.%s" % (path, k)
            if k not in a:
                out.append("%s: missing in REBUILT (committed=%r)" % (p, b[k]))
            elif k not in b:
                out.append("%s: extra in REBUILT (rebuilt=%r)" % (p, a[k]))
            else:
                deep_diff(a[k], b[k], p, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append("%s: length %d (rebuilt) != %d (committed)" % (path, len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            deep_diff(x, y, "%s[%d]" % (path, i), out)
    else:
        if a != b:
            out.append("%s: rebuilt=%r != committed=%r" % (path, a, b))


def main():
    if not os.path.isdir(WHITELIST):
        print("FAIL: whitelist dir absent: %s" % WHITELIST)
        return 2
    if not os.path.exists(COMMITTED):
        print("FAIL: committed canonical_tables.json absent: %s" % COMMITTED)
        return 2
    root = tempfile.mkdtemp(prefix="omniselect_rebuild_")
    try:
        linked = populate(root)
        print("temp root: %s (linked %d whitelisted results.json)" % (root, linked))
        env = dict(os.environ, OMNISELECT_ROOT=root)
        proc = subprocess.run([sys.executable, TABLE_SCRIPT], env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print("FAIL: canonical_paper_tables.py exited %d" % proc.returncode)
            print(proc.stdout)
            print(proc.stderr)
            return 2
        rebuilt_path = os.path.join(root, "experiments", "canonical_tables.json")
        rebuilt = json.load(open(rebuilt_path))
        committed = json.load(open(COMMITTED))
        total = 0
        for key in COMPARE_KEYS:
            diffs = []
            if key not in rebuilt:
                diffs.append("%s: key missing in REBUILT output" % key)
            elif key not in committed:
                diffs.append("%s: key missing in COMMITTED file" % key)
            else:
                deep_diff(rebuilt[key], committed[key], key, diffs)
            verdict = "MATCH" if not diffs else "MISMATCH (%d)" % len(diffs)
            print("%-28s %s" % (key, verdict))
            for d in diffs:
                print("    DIFF %s" % d)
            total += len(diffs)
        print("total diffs: %d" % total)
        if total == 0:
            print("OK: FINAL four-arm tables rebuilt from results_canonical/ are "
                  "numerically identical to the committed canonical_tables.json")
            return 0
        print("FAIL: rebuilt FINAL tables differ from the committed file")
        return 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
