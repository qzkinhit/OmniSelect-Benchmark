#!/usr/bin/env bash
# Publishable sweep for the code-domain fix + fairness correction.
#   - influence channel = reference-PPL quality (INFL_KIND=pplq), the diagnosed fix
#   - DSIR uses the FAIR multi-modal clean-held-out target (`dsir`); `dsir_mc` keeps
#     the legacy math+code-biased target for the diagnostic contrast
#   - ablation ladder: quality_ppl -> +diversity (mmds_noauth) -> +authenticity (mmdataselect)
# Runs multiple seeds sequentially (single MPS device; no GPU parallelism possible).
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2}"
METHODS="${METHODS:-random,dsir,dsir_mc,quality_ppl,mmds_noauth,mmdataselect}"
export MINI_HID=256 MINI_LAYERS=4 MINI_HEADS=4 PASSES=8 INFL_KIND=pplq PYTHONUNBUFFERED=1

mkdir -p outputs/sweep
for s in $SEEDS; do
  echo "=== seed $s ($(date +%H:%M:%S)) ==="
  SEED="$s" METHODS="$METHODS" .venv/bin/python scripts/run_experiment.py \
    2>/dev/null | tee "outputs/sweep/seed${s}.log"
done
echo "=== sweep done ($(date +%H:%M:%S)) ==="
