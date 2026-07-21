#!/usr/bin/env bash
# PER-MODALITY publishable sweep — the honest setting for "data selection serving
# each modality's own base model". For every modality we restrict the pool/budget/
# held-out to that modality (ONLY_DOMAIN) and let every method select *within* it,
# so DSIR's target is unambiguous and there is no cross-modal budget starvation.
#
# Methods span the faithful baselines + ablation ladder + ours:
#   random | zip | if_mates | dsir | dmf | quality_ppl | mmdataselect
# Influence channel = reference-PPL quality (INFL_KIND=pplq). Runs each (modality,
# seed) sequentially (single MPS device). Results land in outputs/permod/.
set -euo pipefail
cd "$(dirname "$0")/.."

MODS="${MODS:-general math code image table}"
SEEDS="${SEEDS:-0 1}"
METHODS="${METHODS:-random,zip,if_mates,dsir,dmf,quality_ppl,mmdataselect}"
export MINI_HID=256 MINI_LAYERS=4 MINI_HEADS=4 PASSES=8 INFL_KIND=pplq PYTHONUNBUFFERED=1

mkdir -p outputs/permod
for m in $MODS; do
  for s in $SEEDS; do
    echo "######## modality=$m seed=$s ($(date +%H:%M:%S)) ########"
    ONLY_DOMAIN="$m" SEED="$s" METHODS="$METHODS" .venv/bin/python scripts/run_experiment.py \
      2>/dev/null | grep --line-buffered -E "device=|^  |EXPERIMENT|method " \
      | tee "outputs/permod/${m}_seed${s}.log"
  done
done
echo "=== PERMOD SWEEP DONE ($(date +%H:%M:%S)) ==="
