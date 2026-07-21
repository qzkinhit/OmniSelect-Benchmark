#!/usr/bin/env bash
# Full reproduction of the 12-task benchmark under the frozen paper protocol.
# This is intentionally expensive. It runs every applicable method for each requested
# seed, then rebuilds the seed-0 canonical table. Baselines share the same pool, split,
# budget, downstream model and seed within each task.
#
# Usage:
#   run_scripts/reproduce_full.sh                 # all 12 tasks, seed 0
#   SEEDS="0 1 2" run_scripts/reproduce_full.sh   # complete three-seed run
#   TASKS="cifar100 etth1 text" SEEDS="0 1 2" run_scripts/reproduce_full.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
SEEDS="${SEEDS:-0}"
TASKS="${TASKS:-cifar100 cifar100n cifar10 imagenet100 etth1 ettm1 etth2 tep21 electricity daisy_cstr daisy_steamgen text}"
STAMP="reproduce-full-$(date +%Y%m%dT%H%M%S)"

echo "== install full dependencies =="
"$PY" -m pip install -q -e ".[train,eval,arms]"

run_task() {
  local task="$1" seed="$2"
  echo "== ${task} / seed ${seed} =="
  case "$task" in
    cifar100)
      RUN_ID="${STAMP}-cifar100" VIS_NOISE=inject \
        run_scripts/run_single_arm.sh vision uoft-cs/cifar100 "$seed" ;;
    cifar100n)
      RUN_ID="${STAMP}-cifar100n" VIS_NOISE=real \
        run_scripts/run_single_arm.sh vision uoft-cs/cifar100 "$seed" ;;
    cifar10)
      RUN_ID="${STAMP}-cifar10" VIS_NOISE=inject \
        run_scripts/run_single_arm.sh vision uoft-cs/cifar10 "$seed" ;;
    imagenet100)
      DATASET=imagenet100 SEED="$seed" \
        "$PY" baselines/deepcore_original/run_original_protocol.py ;;
    etth1)
      RUN_ID="${STAMP}-etth1" run_scripts/run_single_arm.sh timeseries ETTh1 "$seed" ;;
    ettm1)
      RUN_ID="${STAMP}-ettm1" run_scripts/run_single_arm.sh timeseries ETTm1 "$seed" ;;
    etth2)
      RUN_ID="${STAMP}-etth2" run_scripts/run_single_arm.sh timeseries ETTh2 "$seed" ;;
    tep21)
      RUN_ID="${STAMP}-tep21" run_scripts/run_single_arm.sh tep 21 "$seed" ;;
    electricity)
      RUN_ID="${STAMP}-electricity" run_scripts/run_single_arm.sh tabular electricity "$seed" ;;
    daisy_cstr)
      RUN_ID="${STAMP}-daisy-cstr" run_scripts/run_single_arm.sh timeseries daisy_cstr "$seed" ;;
    daisy_steamgen)
      RUN_ID="${STAMP}-daisy-steamgen" run_scripts/run_single_arm.sh timeseries daisy_steamgen "$seed" ;;
    text)
      RUN_ID="${STAMP}-text" STRATIFY=1 INFL_KIND=pplq TRAIN_MODE=finetune LM_EVAL=1 \
        REF_MODEL=HuggingFaceTB/SmolLM2-135M \
        run_scripts/run_single_arm.sh text five_domain "$seed" ;;
    *)
      echo "unknown task: $task" >&2
      exit 2 ;;
  esac
}

for seed in $SEEDS; do
  for task in $TASKS; do
    run_task "$task" "$seed"
  done
done

echo "== rebuild seed-0 canonical table =="
CANONICAL_SCAN_ROOT="$(pwd)/outputs" "$PY" scripts/build_canonical_seed0.py

echo "OK: full reproduction completed for tasks=[$TASKS], seeds=[$SEEDS]"
