#!/usr/bin/env bash
# PER-MODALITY *fine-tune* evaluation — the quality-sensitive selection eval used by
# DSIR / MATES / DoReMi: continue-train a pretrained base (REF_MODEL=SmolLM2-135M) on
# each method's selected data and measure the target-modality held-out PPL. Unlike the
# from-scratch mini-LM (token-hungry, undertrained at small budgets), a pretrained base
# already knows the modality, so held-out PPL responds to *data quality*, not quantity.
#
# `base` = the pretrained model with no training (floor). Methods select within each
# modality (ONLY_DOMAIN). Influence channel = reference-PPL quality.
set -euo pipefail
cd "$(dirname "$0")/.."

MODS="${MODS:-code math general image table}"
SEEDS="${SEEDS:-0 1}"
METHODS="${METHODS:-base,random,dsir,quality_ppl,mmds_noauth,mmdataselect}"
export TRAIN_MODE=finetune INFL_KIND=pplq PYTHONUNBUFFERED=1
export PASSES="${PASSES:-2}" FT_LR="${FT_LR:-2e-5}" FT_STEPS_CAP="${FT_STEPS_CAP:-60}" FT_FREEZE="${FT_FREEZE:-0}"
export CTX="${CTX:-512}" BS="${BS:-8}"

mkdir -p outputs/finetune
for m in $MODS; do
  for s in $SEEDS; do
    echo "######## modality=$m seed=$s ($(date +%H:%M:%S)) ########"
    ONLY_DOMAIN="$m" SEED="$s" METHODS="$METHODS" .venv/bin/python scripts/run_experiment.py \
      2>/dev/null | grep --line-buffered -E "device=|train_mode|^  |EXPERIMENT|method " \
      | tee "outputs/finetune/${m}_seed${s}.log"
  done
done
echo "=== FINETUNE SWEEP DONE ($(date +%H:%M:%S)) ==="
