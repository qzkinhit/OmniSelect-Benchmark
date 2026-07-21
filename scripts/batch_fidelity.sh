#!/bin/bash
# Tier-1 fidelity gate:CIFAR-10 clean 原协议剪枝曲线 {random,el2n,grand,ccs}
# keep in {0.1,0.2,0.3,0.5,0.8}, ResNet-18 from scratch, SCORE_RUNS=3, 40ep(曲线形状).
# 复现 Data-Diet(Paul21)/CCS(Zheng23):高剪枝下 EL2N/GraNd 跌破随机, CCS 稳健.
set -Eeuo pipefail
source /root/omni_env.sh; export PYTHONUNBUFFERED=1
cd /root/autodl-tmp/OmniSelect
for K in 0.1 0.2 0.3 0.5 0.8; do
  LOG=experiments/fidelity_curve_cifar10_keep${K}.log
  echo "##### FIDELITY CIFAR10 KEEP=$K $(date +%F_%T) #####" | tee -a "$LOG"
  set +e
  DATASET=cifar10 POOL=45000 VAL_N=5000 SCORE_EPOCH=10 TRAIN_EPOCH=40 SCORE_RUNS=3 NW=6 KEEP=$K SEED=0 \
    .venv/bin/python -u baselines/deepcore_original/run_original_protocol.py >> "$LOG" 2>&1
  RC=$?; set -e
  echo "KEEP=$K python_exit=$RC" | tee -a "$LOG"
done
echo FIDELITY_CURVE_DONE > /root/fidelity_curve.done
