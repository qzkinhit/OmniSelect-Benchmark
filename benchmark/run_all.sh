#!/bin/bash
# Run every standalone method runner on one track, dataset and seed.
#   bash benchmark/run_all.sh <track> <dataset> [seed] [protocol] [driver flags ...]
# A method that cannot run on the track is skipped by the driver with a reason in its log.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
TRACK="${1:?track}"; DATASET="${2:?dataset}"; SEED="${3:-0}"; PROTOCOL="${4:-v2}"
shift $(( $# < 4 ? $# : 4 ))
status=0
for runner in run_random run_authenticity_only run_influence_only run_coverage run_fixed_fusion run_herding run_kcenter run_semdedup run_density run_el2n run_grand run_ccs run_dmf run_quadmix run_dsdm run_d4 run_tabaicl run_dsir run_zip run_if_mates run_glister run_gradmatch run_infomax run_less run_clean_oracle run_full_data; do
  echo "== $runner"
  bash "benchmark/MethodsRunScript/$runner/run.sh" "$TRACK" "$DATASET" "$SEED" "$PROTOCOL" "$@" || status=1
done
exit $status
