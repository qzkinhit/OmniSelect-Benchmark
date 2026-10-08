#!/bin/bash
# OmniSelect-Benchmark runner. Four modes, one entry point.
#
#   bash run.sh setup                 install the package with the dev extra, run the CPU test suite,
#                                     verify the checksums of the committed data files
#   bash run.sh smoke                 one small cell per track (vision, timeseries, process, tabular, text)
#                                     under each protocol of PROTOCOLS, then recompute macro-F1 from the
#                                     stored per-unit files and replay every stored selection
#   bash run.sh local                 the cells of TRACK_CELLS at full size, one after another
#   bash run.sh queue FILE [FLAGS]    run the lines of a queue file (python -m run_omniselect.run_queue)
#
# Overrides: PY (python), OUT (results_and_logs), BATCH, PROTOCOLS (v2_2), TRACK_CELLS ("track:dataset ..."), SEEDS
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
PY="${PY:-python}"
MODE="${1:-smoke}"
OUT="${OUT:-results_and_logs}"

run_cells() {  # $1 batch, $2 protocol, $3 seeds, $4 extra flags; cells from TRACK_CELLS
  local batch="$1" protocol="$2" seeds="$3" extra="$4" cell track dataset seed code
  for cell in $TRACK_CELLS; do
    track="${cell%%:*}"; dataset="${cell#*:}"
    for seed in ${seeds//,/ }; do
      set +e
      $PY -m tracks.common.experiment --track "$track" --dataset "$dataset" --seed "$seed" \
        --protocol "$protocol" --batch "$batch" --out "$OUT" $extra
      code=$?
      set -e
      if [ "$code" -eq 3 ]; then
        echo "[run.sh] $track/$dataset seed $seed skipped (model or dataset unavailable, see the message above)"
      elif [ "$code" -ne 0 ]; then
        echo "[run.sh] $track/$dataset seed $seed failed with exit code $code"; return "$code"
      fi
    done
  done
}

case "$MODE" in
setup)
  $PY -m pip install -e ".[dev]"
  $PY -m pytest omniselect/tests -m "not slow"
  $PY data/fetch_data.py --verify-only --only tep daisy cifar_n
  ;;
smoke)
  BATCH="${BATCH:-smoke}"
  TRACK_CELLS="${TRACK_CELLS:-vision:cifar100 timeseries:ETTh1 process:tep21 tabular:electricity text:five_domain}"
  for protocol in ${PROTOCOLS:-v2_2}; do
    LOKY_MAX_CPU_COUNT=4 run_cells "$BATCH/$protocol" "$protocol" "${SEEDS:-0}" "--smoke"
  done
  $PY -m omniselect.store.recompute "$OUT/$BATCH" --metric macro_f1 --check > "$OUT/$BATCH/recompute_macro_f1.jsonl"
  tail -n 1 "$OUT/$BATCH/recompute_macro_f1.jsonl"
  $PY -m omniselect.store.replay "$OUT/$BATCH"
  $PY -m run_omniselect.generate_csv --batch "$OUT/$BATCH" --out "$OUT/$BATCH/summary"
  ;;
local)
  BATCH="${BATCH:-local}"
  TRACK_CELLS="${TRACK_CELLS:-process:tep21 timeseries:ETTh1}"
  for protocol in ${PROTOCOLS:-v2_2}; do
    run_cells "$BATCH/$protocol" "$protocol" "${SEEDS:-0}" ""
  done
  $PY -m run_omniselect.generate_csv --batch "$OUT/$BATCH" --out "$OUT/$BATCH/summary"
  ;;
queue)
  shift
  if [ "$#" -eq 0 ]; then echo "usage: bash run.sh queue QUEUE_FILE [--gpus 0,1] [--jobs-per-gpu N] [--threads N]"; exit 1; fi
  exec $PY -m run_omniselect.run_queue "$@"
  ;;
*)
  echo "unknown mode $MODE; one of setup smoke local queue"; exit 1 ;;
esac
