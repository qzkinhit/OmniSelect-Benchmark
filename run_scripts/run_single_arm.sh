#!/usr/bin/env bash
# OmniSelect single-arm runner. Run one modality x one dataset x one seed, with any
# subset of methods (baselines + OmniSelect), under the unified equal-budget protocol.
#
# Usage:
#   run_scripts/run_single_arm.sh <arm> [dataset] [seed] [methods]
#
#   <arm>      vision | timeseries | tep | tabular | text
#   [dataset]  vision: cifar100|cifar10   timeseries: ETTh1|ETTh2|ETTm1
#              tep: 21 (faults)           tabular: electricity   text: (fixed pool)
#   [seed]     integer, default 0
#   [methods]  comma list, default = the arm's full set incl. mmds_adapt (OmniSelect)
#
# Examples:
#   run_scripts/run_single_arm.sh vision cifar100 0
#   run_scripts/run_single_arm.sh timeseries ETTh1 1 random,auth_only,mmds_adapt
set -euo pipefail
cd "$(dirname "$0")/.."

ARM="${1:?arm required: vision|timeseries|tep|tabular|text}"
DATASET="${2:-}"
SEED="${3:-0}"
METHODS_ARG="${4:-}"
PY="${PYTHON:-python}"

export SEED
[ -n "$METHODS_ARG" ] && export METHODS="$METHODS_ARG"

case "$ARM" in
  vision)
    export VIS_DATASET="${DATASET:-uoft-cs/cifar100}"
    exec "$PY" scripts/run_vision_experiment.py ;;
  timeseries)
    export TS_DATASET="${DATASET:-ETTh1}"
    exec "$PY" scripts/run_timeseries_experiment.py ;;
  tep)
    export N_FAULTS="${DATASET:-21}"
    exec "$PY" scripts/run_tep_experiment.py ;;
  tabular)
    export TAB_DATASET="${DATASET:-electricity}"
    exec "$PY" scripts/run_tabular_experiment.py ;;
  text)
    exec "$PY" scripts/run_experiment.py ;;
  *)
    echo "unknown arm: $ARM (expected vision|timeseries|tep|tabular|text)" >&2
    exit 2 ;;
esac
