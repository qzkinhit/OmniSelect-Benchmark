#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${REPO:-/root/autodl-tmp/OmniSelect_v3_next}"
source /root/omni_env.sh
RUN_ID="${RUN_ID:-backfill-daisy_cstr-dlinear-full-20260720}"
LOG="${LOG:-/root/daisy_cstr_dlinear_full_${RUN_ID}.log}"
MARKER="${MARKER:-/root/daisy_cstr_DLINEAR_FULL_OK}"
PYTHON="${PYTHON:-/root/autodl-tmp/OmniSelect/.venv/bin/python}"
exec > >(tee -a "${LOG}") 2>&1
echo "lane=daisy_cstr-DLinear-full run_id=${RUN_ID} start=$(date -Is)"
export RUN_ID TS_DATASET=daisy_cstr TS_MODEL=dlinear
export POOL_N=3000 VAL_N=1000 TEST_N=1500 L=96 H=24
export NOISE_FRAC=0.4 BUDGET_FRAC=0.3 KNN=15 EPOCHS=60
export LAM=0.5 AUTH_Q=0.25 W_INFL=0.5
export TS_VAL_MODE=full DSDM_RUNS=12
export ADAPT_GRPO=0 ADAPT_MARGIN=0.015 ADAPT_SH=0 ROBUST_VAL=0 VAL_NOISE=0
export PAIRED_RNG=1
export METHODS=full,random,coreset,auth_only,influence_only,mmdataselect,mmds_adapt,herding,kcenter,semdedup,density,quadmix,quadmix_pub,dmf,dmf_pub
for SEED in 0 1 2; do
  export SEED
  echo "===== daisy_cstr DLinear SEED=${SEED} start=$(date -Is) ====="
  set +e
  "${PYTHON}" -u "${REPO}/scripts/run_timeseries_experiment.py"
  rc=$?
  set -e
  echo "SEED=${SEED} python_exit=${rc}"
  [[ ${rc} -ne 0 ]] && exit "${rc}"
done
touch "${MARKER}"
echo "lane=daisy_cstr-DLinear-full end=$(date -Is)"
