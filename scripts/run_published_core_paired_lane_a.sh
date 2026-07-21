#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/OmniSelect
source /root/omni_env.sh
RUN_ID=${RUN_ID:?RUN_ID is required}
export RUN_ID PAIRED_RNG=1 FIDELITY_MODE=published-core-unified-protocol-v1
export ADAPT_GRPO=0 ADAPT_MARGIN=0.015 ADAPT_SH=0 ROBUST_VAL=0 VAL_NOISE=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
METHODS_CLASS=full,random,coreset,auth_only,influence_only,mmdataselect,herding,kcenter,el2n,grand,ccs,semdedup,density,quadmix,quadmix_pub,dmf,dmf_pub,mmds_adapt
METHODS_TS=full,random,coreset,auth_only,influence_only,mmdataselect,herding,kcenter,semdedup,density,quadmix,quadmix_pub,dmf,dmf_pub,mmds_adapt

for seed in 0 1 2; do
  SEED=$seed METHODS=$METHODS_CLASS VIS_DATASET=uoft-cs/cifar100 VIS_ENCODER=openai/clip-vit-base-patch32 \
    POOL_N=4000 VAL_N=800 TEST_N=2000 NOISE_FRAC=0.4 BUDGET_FRAC=0.5 KNN=15 LAM=0.5 AUTH_Q=0.25 W_INFL=0.5 VIS_NOISE=inject \
    .venv/bin/python -u scripts/run_vision_experiment.py
  echo "[trial] vision seed=$seed python_exit=0"
done

for seed in 0 1 2; do
  SEED=$seed METHODS=$METHODS_TS TS_DATASET=ETTh1 TS_MODEL=dlinear POOL_N=3000 VAL_N=1000 TEST_N=1500 \
    L=96 H=24 NOISE_FRAC=0.4 BUDGET_FRAC=0.3 KNN=15 EPOCHS=60 LAM=0.5 AUTH_Q=0.25 W_INFL=0.5 TS_VAL_MODE=full \
    .venv/bin/python -u scripts/run_timeseries_experiment.py
  echo "[trial] timeseries seed=$seed python_exit=0"
done
