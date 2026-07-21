#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/OmniSelect
source /root/omni_env.sh
RUN_ID=${RUN_ID:?RUN_ID is required}
export RUN_ID PAIRED_RNG=1 ADAPT_GRPO=0 ADAPT_MARGIN=0.015 ADAPT_SH=0 ROBUST_VAL=0 VAL_NOISE=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
METHODS_CLASS=full,random,coreset,auth_only,influence_only,mmdataselect,herding,kcenter,el2n,grand,ccs,semdedup,density,quadmix,dmf,mmds_adapt

for seed in 0 1 2; do
  SEED=$seed METHODS=$METHODS_CLASS MODEL=mlp N_FAULTS=21 POOL_N=4000 VAL_N=2000 TEST_N=3000 \
    NOISE_FRAC=0.4 BUDGET_FRAC=0.3 KNN=15 LAM=0.5 AUTH_Q=0.25 W_INFL=0.5 TEP_CALIB=0 CNN_EPOCHS=80 \
    .venv/bin/python -u scripts/run_tep_experiment.py
  echo "[trial] tep seed=$seed python_exit=0"
done

for seed in 0 1 2; do
  SEED=$seed METHODS=$METHODS_CLASS TAB_DATASET=electricity MODEL=tabpfn POOL_N=3000 VAL_N=2500 TEST_N=2000 \
    NOISE_FRAC=0.4 BUDGET_FRAC=0.5 KNN=15 LAM=0.5 AUTH_Q=0.25 W_INFL=0.5 \
    .venv/bin/python -u scripts/run_tabular_experiment.py
  echo "[trial] tabular seed=$seed python_exit=0"
done
