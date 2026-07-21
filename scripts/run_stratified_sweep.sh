#!/usr/bin/env bash
# STRATIFIED multi-modal sweep — one shared from-scratch model trained on the union of
# per-modality fair-budget selections (STRATIFY=1), evaluated per modality. This is the
# regime where a selection benefit is observable: enough tokens to exit undertraining
# (~165K, the proven scale) AND no cross-modal starvation (each modality keeps its own
# BUDGET_FRAC share). Influence channel = reference-PPL quality.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2}"
# zip (Entropy-Law set-greedy) is O(n^2) in the set size and prohibitive at this pool
# size when asked for a full ranking; it is evaluated separately on smaller per-modality
# pools. Headline baselines below.
METHODS="${METHODS:-random,dsir,dmf,if_mates,quality_ppl,mmds_noauth,mmdataselect}"
export MINI_HID=256 MINI_LAYERS=4 MINI_HEADS=4 PASSES=8 INFL_KIND=pplq STRATIFY=1 TRAIN_MODE=scratch PYTHONUNBUFFERED=1

mkdir -p outputs/stratified
for s in $SEEDS; do
  echo "######## seed=$s ($(date +%H:%M:%S)) ########"
  SEED="$s" METHODS="$METHODS" .venv/bin/python scripts/run_experiment.py \
    2>"outputs/stratified/seed${s}.err" | grep --line-buffered -E "device=|train_mode|^  |EXPERIMENT|method |_ppl" \
    | tee "outputs/stratified/seed${s}.log"
done
echo "=== STRATIFIED SWEEP DONE ($(date +%H:%M:%S)) ==="
