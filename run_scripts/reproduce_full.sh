#!/usr/bin/env bash
# Full reproduction: run every applicable method x dataset x seed(s) under the frozen
# paper protocol, then rebuild the canonical table from the fresh outputs. Needs a
# GPU for the vision/timeseries/tabular arms (CPU-only arms still work, just slower).
# Raw data are auto-fetched on first use (see scripts/fetch_data.py to pre-fetch).
#
# Usage:
#   run_scripts/reproduce_full.sh                       # 4 core arms, seed 0 (matches the headline table)
#   TS_DATASET=ETTh2  ARMS="timeseries" run_scripts/reproduce_full.sh   # the other headline datasets:
#   TS_DATASET=ETTm1  ARMS="timeseries" run_scripts/reproduce_full.sh   # ETTh2 / ETTm1 / daisy_cstr / daisy_steamgen
#   TS_DATASET=daisy_cstr     ARMS="timeseries" run_scripts/reproduce_full.sh
#   TS_DATASET=daisy_steamgen ARMS="timeseries" run_scripts/reproduce_full.sh
#   VIS_DATASET=uoft-cs/cifar10 ARMS="vision" run_scripts/reproduce_full.sh   # CIFAR-10 (CLIP protocol)
#   VIS_DATASET=uoft-cs/cifar100 VIS_NOISE=real ARMS="vision" run_scripts/reproduce_full.sh  # CIFAR-100N
#   ARMS="vision tabular" SEEDS="0 1 2" run_scripts/reproduce_full.sh  # legacy 3-seed regression check
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

ARMS="${ARMS:-vision timeseries tep tabular}"
SEEDS="${SEEDS:-0}"
STAMP="reproduce-full-$(date +%Y%m%dT%H%M%S)"

echo "== install (full: torch + eval extras) =="
$PY -m pip install -q -e ".[train,eval]" 2>&1 | tail -1 || true

for arm in $ARMS; do
  case "$arm" in
    vision)     ds="${VIS_DATASET:-uoft-cs/cifar100}" ;;
    timeseries) ds="${TS_DATASET:-ETTh1}" ;;
    tep)        ds="${N_FAULTS:-21}" ;;
    tabular)    ds="${TAB_DATASET:-electricity}" ;;
    *) echo "skip unknown arm $arm"; continue ;;
  esac
  for s in $SEEDS; do
    echo "== $arm / $ds / seed $s =="
    RUN_ID="$STAMP" run_scripts/run_single_arm.sh "$arm" "$ds" "$s" || {
      echo "FAILED: $arm seed $s"; exit 1; }
  done
done

echo "== rebuild the headline table from the fresh outputs =="
CANONICAL_SCAN_ROOT="$(pwd)/outputs" $PY scripts/build_canonical_seed0.py

echo "OK: full reproduction complete. Table in experiments/canonical_tables_seed0.json"
