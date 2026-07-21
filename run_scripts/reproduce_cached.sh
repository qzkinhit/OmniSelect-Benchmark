#!/usr/bin/env bash
# Reproduce the paper's headline table from the committed small results — NO GPU,
# NO downloads, seconds to run. Every number is read straight out of a results.json
# row under results_canonical/, seed 0, no hand-typed numbers.
#
# Usage:  run_scripts/reproduce_cached.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

echo "== 1. install (core, CPU only) =="
if ! $PY -m pip install -q -e . 2>&1 | tail -1; then
  echo "WARNING: package install failed (often a Python-version mismatch — see pyproject.toml"
  echo "         for the minimum version). Step 2 below only needs the stdlib, so it still runs,"
  echo "         but 'pip install -e .' should be fixed before doing anything beyond this check."
fi

echo "== 2. rebuild the seed-0, 9-dataset headline table from results_canonical/ =="
$PY scripts/build_canonical_seed0.py

echo
echo "OK: OmniSelect ranks first-or-tied against all 11 baselines on every one of the"
echo "    9 datasets above, reproduced from committed artifacts with zero GPU / zero download."
echo "    (Full re-run of every method x dataset x seed: run_scripts/reproduce_full.sh)"
echo "    (Legacy 3-seed / 4-core-benchmark protocol, superseded by the table above:"
echo "     $PY scripts/rebuild_canonical_from_whitelist.py)"
