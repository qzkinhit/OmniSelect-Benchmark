#!/bin/bash
# One cell: a track, a dataset, a seed and a protocol, with the full run record.
#
#   bash run_omniselect/run_cell.sh vision cifar100 0 v2_2
#   bash run_omniselect/run_cell.sh timeseries ETTh1 2 v2 ablation
#   bash run_omniselect/run_cell.sh process tep21 0 v2_2 local --set gate.kind=lcb --track-set learner=rf
#
# Args: $1 track, $2 dataset, $3 seed (0), $4 protocol (v2_2), $5 batch (cells). Further arguments go
# to tracks/common/experiment.py (--set, --track-set, --methods, --smoke, --standalone, --overwrite).
# Output: results_and_logs/<batch>/<track>/<dataset>/<learner>/seed_<s>/ as described in docs/RUN_RECORD.md.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
TRACK="${1:?track (vision native timeseries process tabular text)}"
DATASET="${2:?dataset}"
SEED="${3:-0}"
PROTOCOL="${4:-v2_2}"
BATCH="${5:-cells}"
shift $(( $# < 5 ? $# : 5 ))
exec ${PY:-python} -m tracks.common.experiment --track "$TRACK" --dataset "$DATASET" --seed "$SEED" \
  --protocol "$PROTOCOL" --batch "$BATCH" --out "${OUT:-results_and_logs}" "$@"
