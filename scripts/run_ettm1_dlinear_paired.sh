#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-/root/autodl-tmp/OmniSelect}"
RUN_ID="${RUN_ID:-ettm1-dlinear-paired-v1}"
LOG="${LOG:-/root/ettm1_dlinear_paired_${RUN_ID}.log}"
MARKER="${MARKER:-/root/ETTM1_DLINEAR_PAIRED_OK}"
PYTHON="${PYTHON:-${REPO}/.venv/bin/python}"

exec > >(tee -a "${LOG}") 2>&1

echo "lane=ETTm1-DLinear-paired run_id=${RUN_ID} start=$(date -Is)"
echo "repo=${REPO} python=${PYTHON}"
sha256sum \
  "${REPO}/scripts/run_timeseries_experiment.py" \
  "${REPO}/src/mmdataselect/utils/pairing.py" \
  "${REPO}/data/processed/ettm1.csv"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

export RUN_ID TS_DATASET=ETTm1 TS_MODEL=dlinear
export POOL_N=3000 VAL_N=1000 TEST_N=1500 L=96 H=24
export NOISE_FRAC=0.4 BUDGET_FRAC=0.3 KNN=15 EPOCHS=60
export LAM=0.5 AUTH_Q=0.25 W_INFL=0.5
export TS_VAL_MODE=full DSDM_RUNS=12
export ADAPT_GRPO=0 ADAPT_MARGIN=0.015 ADAPT_SH=0 ROBUST_VAL=0 VAL_NOISE=0
export PAIRED_RNG=1
export METHODS=full,random,coreset,auth_only,influence_only,mmdataselect,mmds_adapt,herding,kcenter,semdedup,density,quadmix,dmf

for SEED in 0 1 2; do
  export SEED
  echo "===== ETTm1 DLinear paired SEED=${SEED} start=$(date -Is) ====="
  set +e
  "${PYTHON}" -u "${REPO}/scripts/run_timeseries_experiment.py"
  rc=$?
  set -e
  echo "SEED=${SEED} python_exit=${rc}"
  if [[ ${rc} -ne 0 ]]; then
    exit "${rc}"
  fi
done

"${PYTHON}" -u "${REPO}/scripts/validate_timeseries_paired.py" \
  --repo "${REPO}" --run-id "${RUN_ID}" --log "${LOG}" --marker "${MARKER}"
echo "lane=ETTm1-DLinear-paired end=$(date -Is) validator_exit=0"
