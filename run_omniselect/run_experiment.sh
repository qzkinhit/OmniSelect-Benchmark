#!/bin/bash
# A batch of cells run one after another: every (track:dataset) of the list times every seed.
#
#   bash run_omniselect/run_experiment.sh main "process:tep21 timeseries:ETTh1" 0,1,2 v2_2
#   bash run_omniselect/run_experiment.sh gates "vision:cifar100" 0 v2_2 --set gate.kind=eprocess --set gate.split=conf
#
# Args: $1 batch, $2 cells (space or comma separated track:dataset), $3 seeds (comma separated),
# $4 protocol (v2_2). Further arguments go to every cell. A cell with decision.json is skipped by the
# driver. An unavailable model or dataset (exit code 3) is reported and skipped.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
BATCH="${1:?batch}"; CELLS="${2:?cells}"; SEEDS="${3:-0}"; PROTOCOL="${4:-v2_2}"
shift $(( $# < 4 ? $# : 4 ))
status=0
for cell in ${CELLS//,/ }; do
  for seed in ${SEEDS//,/ }; do
    ${PY:-python} -m tracks.common.experiment --track "${cell%%:*}" --dataset "${cell#*:}" --seed "$seed" \
      --protocol "$PROTOCOL" --batch "$BATCH" --out "${OUT:-results_and_logs}" "$@"
    code=$?
    if [ "$code" -eq 3 ]; then echo "[skip] $cell seed $seed"; elif [ "$code" -ne 0 ]; then status=1; fi
  done
done
exit $status
