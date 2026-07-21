#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-/root/autodl-tmp/OmniSelect}"
RUN_ID="${RUN_ID:-codex-text-quadmix-v1}"
LOG="${LOG:-/root/text_quadmix_${RUN_ID}.log}"
MARKER="${MARKER:-/root/TEXT_QUADMIX_TRANSFER_OK}"
PYTHON="${PYTHON:-${REPO}/.venv/bin/python}"

exec > >(tee -a "${LOG}") 2>&1
if [[ -f /root/omni_env.sh ]]; then
  source /root/omni_env.sh
fi
cd "${REPO}"
echo "lane=text-quadmix run_id=${RUN_ID} start=$(date -Is)"
sha256sum scripts/run_experiment.py data/processed/qpool_train.jsonl data/processed/qpool_heldout.jsonl
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

export RUN_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export STRATIFY=1 INFL_KIND=pplq TRAIN_MODE=finetune LM_EVAL=1 LMEVAL_BS=32
export LMEVAL_TASKS=arc_easy,arc_challenge,hellaswag,openbookqa
export PASSES=2 PROBE_TOKENS=300000 METHODS=quadmix

for SEED in 0 1 2; do
  export SEED
  echo "===== TEXT QUADMIX SEED=${SEED} start=$(date -Is) ====="
  set +e
  "${PYTHON}" -u scripts/run_experiment.py
  rc=$?
  set -e
  echo "SEED=${SEED} python_exit=${rc}"
  if [[ ${rc} -ne 0 ]]; then
    exit "${rc}"
  fi
done

"${PYTHON}" -u scripts/validate_text_quadmix.py \
  --repo "${REPO}" --run-id "${RUN_ID}" --log "${LOG}" --marker "${MARKER}"
echo "lane=text-quadmix end=$(date -Is) validator_exit=0"
