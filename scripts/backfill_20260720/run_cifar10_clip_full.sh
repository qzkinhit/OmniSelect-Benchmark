#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/OmniSelect_v3_next
export RUN_ID=backfill-cifar10-clip-full-20260720 PAIRED_RNG=1
source /root/omni_env.sh
export ADAPT_GRPO=0 ADAPT_MARGIN=0.015 ADAPT_SH=0 ROBUST_VAL=0 VAL_NOISE=0
export TRANSFORMERS_OFFLINE=1
SEED=0 METHODS=full,random,coreset,auth_only,influence_only,mmdataselect,herding,kcenter,el2n,grand,ccs,semdedup,density,quadmix,quadmix_pub,dmf,dmf_pub,mmds_adapt VIS_DATASET=uoft-cs/cifar10 VIS_ENCODER=openai/clip-vit-base-patch32 \
  POOL_N=4000 VAL_N=800 TEST_N=2000 NOISE_FRAC=0.4 BUDGET_FRAC=0.5 KNN=15 LAM=0.5 AUTH_Q=0.25 W_INFL=0.5 VIS_NOISE=inject \
  /root/autodl-tmp/OmniSelect/.venv/bin/python -u scripts/run_vision_experiment.py
echo "cifar10-clip seed0 python_exit=0"
touch /root/CIFAR10_CLIP_FULL_OK
