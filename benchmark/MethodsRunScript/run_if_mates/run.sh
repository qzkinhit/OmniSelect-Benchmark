#!/bin/bash
# Standalone IF / MATES runner.
#   bash benchmark/MethodsRunScript/run_if_mates/run.sh <track> <dataset> [seed] [protocol] [driver flags ...]
#   bash benchmark/MethodsRunScript/run_if_mates/run.sh text five_domain 0 v2 --smoke
# Writes results_and_logs/standalone_if_mates/<track>/<dataset>/<learner>/seed_<s>/.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
TRACK="${1:?track (vision native timeseries process tabular text)}"
DATASET="${2:?dataset}"
SEED="${3:-0}"
PROTOCOL="${4:-v2}"
shift $(( $# < 4 ? $# : 4 ))
python -m benchmark.MethodsRunScript.run_if_mates.run_if_mates_base --track "$TRACK" --dataset "$DATASET" \
  --seed "$SEED" --protocol "$PROTOCOL" --batch "standalone_if_mates" "$@"
