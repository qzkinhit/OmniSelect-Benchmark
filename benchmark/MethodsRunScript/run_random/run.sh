#!/bin/bash
# Standalone Random runner.
#   bash benchmark/MethodsRunScript/run_random/run.sh <track> <dataset> [seed] [protocol] [driver flags ...]
#   bash benchmark/MethodsRunScript/run_random/run.sh vision cifar100 0 v2 --smoke
# Writes results_and_logs/standalone_random/<track>/<dataset>/<learner>/seed_<s>/.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
TRACK="${1:?track (vision native timeseries process tabular text)}"
DATASET="${2:?dataset}"
SEED="${3:-0}"
PROTOCOL="${4:-v2}"
shift $(( $# < 4 ? $# : 4 ))
python -m benchmark.MethodsRunScript.run_random.run_random_base --track "$TRACK" --dataset "$DATASET" \
  --seed "$SEED" --protocol "$PROTOCOL" --batch "standalone_random" "$@"
