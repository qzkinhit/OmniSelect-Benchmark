#!/usr/bin/env bash
# OmniSelect 一键跑全套。用法:
#   ./runall.sh setup   # 一键环境:venv + 依赖 + pytest 冒烟(上服务器第一步)
#   ./runall.sh smoke   # 每模态 1-seed 小规模跑通(~15 分钟,验证环境/数据/方法全链路)
#   ./runall.sh local   # 本地标准套件:四模态+DaISy 全方法 3-seed + 文本 pilot(M2 半天/GPU 更快)
#   ./runall.sh server  # 服务器放大提示(标准榜单级任务见 SERVER_PROMPT.md 第 3/5 节)
# 环境变量:SEEDS="0 1 2" 覆盖种子;单跑某模态直接用对应 run_xxx(见下方各函数)。
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
SEEDS="${SEEDS:-0 1 2}"
FILTER='Warning|warn|httpx|INFO|Downloading|%\|'
mkdir -p experiments

# 全部方法名单(统一输入格式:METHODS 逗号分隔,SEED 整数,数据集用各 runner 的 *_DATASET)
M_VIS="random,full,el2n,grand,ccs,herding,kcenter,coreset,semdedup,density,quadmix,dmf,auth_only,influence_only,mmdataselect,mmds_adapt"
M_TEP="full,random,coreset,el2n,grand,ccs,herding,kcenter,semdedup,density,quadmix,dmf,auth_only,influence_only,mmdataselect,mmds_adapt"
M_TAB="full,random,coreset,el2n,grand,ccs,herding,kcenter,semdedup,density,quadmix,dmf,auth_only,influence_only,mmdataselect,tabpfn_coreset,tabpfn_margin,tabpfn_hybrid,mmds_adapt"
M_TS="full,random,coreset,herding,kcenter,semdedup,density,quadmix,dmf,auth_only,influence_only,mmdataselect,mmds_adapt"   # el2n/grand/ccs 是分类误差类,回归不适用
M_TXT="noselect,random,dsir,zip,if_mates,quality_ppl,dmf,balance,mmdataselect"

setup() {
  [ -d .venv ] || python3 -m venv .venv
  $PY -m pip install -q -r requirements.txt
  echo "== pytest 冒烟 =="
  $PY -m pytest tests/ -q || { echo "!! pytest 未全过,先修环境"; exit 1; }
  echo "== 数据自检 =="
  [ -f data/daisy/cstr.dat ] || { gunzip -kf data/daisy/cstr.dat.gz 2>/dev/null || \
    curl -s -o data/daisy/cstr.dat.gz "ftp://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/cstr.dat.gz" && gunzip -kf data/daisy/cstr.dat.gz; }
  echo "setup 完成(CIFAR/ETT/electricity/TEP 首跑自动下载;文本 gated 集需 huggingface-cli login)"
}

run_vision()  { local s=$1 ds=${2:-uoft-cs/cifar100} extra=${3:-};
  METHODS="$M_VIS" SEED=$s VIS_DATASET="$ds" $extra $PY scripts/run_vision_experiment.py 2>&1 | grep -vE "$FILTER"; }
run_tep()     { local s=$1 mdl=${2:-mlp};
  METHODS="$M_TEP" SEED=$s MODEL=$mdl $PY scripts/run_tep_experiment.py 2>&1 | grep -vE "$FILTER"; }
run_tabular() { local s=$1 mdl=${2:-tabpfn} ds=${3:-electricity};
  METHODS="$M_TAB" SEED=$s MODEL=$mdl TAB_DATASET=$ds $PY scripts/run_tabular_experiment.py 2>&1 | grep -vE "$FILTER"; }
run_ts()      { local s=$1 ds=${2:-ETTh1};
  METHODS="$M_TS" SEED=$s TS_DATASET=$ds $PY scripts/run_timeseries_experiment.py 2>&1 | grep -vE "$FILTER"; }
run_text()    { local s=$1;
  METHODS="$M_TXT" SEED=$s MINI_HID=320 MINI_LAYERS=4 MINI_HEADS=5 PASSES=3 TRAIN_MODE=scratch \
    $PY scripts/run_experiment.py 2>&1 | grep -vE "$FILTER"; }

smoke() {
  set -o pipefail
  local FAIL=0
  ck() { [ "$1" -ne 0 ] && { echo "!! FAILED(exit=$1): $2"; FAIL=1; }; return 0; }
  echo "== smoke:每模态 1-seed 小规模 =="
  METHODS="random,auth_only,dmf,mmds_adapt" SEED=0 POOL_N=1200 TEST_N=600 VAL_N=400 $PY scripts/run_vision_experiment.py 2>&1 | grep -vE "$FILTER" | tail -8; ck $? vision
  METHODS="random,auth_only,dmf,mmds_adapt" SEED=0 MODEL=mlp $PY scripts/run_tep_experiment.py 2>&1 | grep -vE "$FILTER" | tail -8; ck $? tep
  METHODS="random,auth_only,dmf,mmds_adapt" SEED=0 POOL_N=1200 $PY scripts/run_tabular_experiment.py 2>&1 | grep -vE "$FILTER" | tail -8; ck $? tabular
  METHODS="random,auth_only,dmf,mmds_adapt" SEED=0 POOL_N=800 $PY scripts/run_timeseries_experiment.py 2>&1 | grep -vE "$FILTER" | tail -8; ck $? etth1
  METHODS="random,auth_only,dmf,mmds_adapt" SEED=0 TS_DATASET=daisy_cstr POOL_N=800 $PY scripts/run_timeseries_experiment.py 2>&1 | grep -vE "$FILTER" | tail -8; ck $? daisy
  METHODS="random,dsir" SEED=0 MINI_HID=64 MINI_LAYERS=2 MINI_HEADS=2 PASSES=1 $PY scripts/run_experiment.py 2>&1 | grep -vE "$FILTER" | tail -6; ck $? text
  if [ "$FAIL" -eq 0 ]; then echo "== smoke 完成:六条链路(视觉/TEP/表格/ETT/DaISy/文本)全通 =="; else echo "== smoke 存在失败臂,见上方 FAILED 行 =="; exit 1; fi
}

local_suite() {
  echo "== 本地标准套件(3-seed 全方法,日志进 experiments/*.log)=="
  for s in $SEEDS; do { echo "##### SEED $s VISION #####";  run_vision  $s; } >> experiments/vision_full_3seed.log;  done
  for s in $SEEDS; do { echo "##### SEED $s TEP #####";     run_tep     $s; } >> experiments/tep_full_3seed.log;     done
  for s in $SEEDS; do { echo "##### SEED $s TABULAR #####"; run_tabular $s; } >> experiments/tabular_full_3seed.log; done
  for s in $SEEDS; do { echo "##### SEED $s ETTh1 #####";   run_ts $s ETTh1; } >> experiments/timeseries_full_3seed.log; done
  for s in $SEEDS; do { echo "##### SEED $s DAISY #####";   run_ts $s daisy_cstr; } >> experiments/daisy_cstr_3seed.log; done
  { echo "##### TEXT PILOT seed0 #####"; run_text 0; } >> experiments/text_pilot.log
  echo "== 本地套件完成,汇总: =="
  $PY scripts/summarize_runs.py 2>/dev/null || true
}

server_hint() {
  cat <<'EOF'
服务器放大不由本脚本自动跑(涉及租卡与大数据下载),按 SERVER_PROMPT.md 执行:
  任务A 文本:FineWeb-Edu/FineMath/Stack-Edu + SmolLM2 持续预训练 + lm-eval,新增 DoReMi/RegMix/QuRating/DsDm 纳入组合
  任务B 视觉:ImageNet+ResNet 从零;真 VL 臂上 DataComp 过滤赛道公开榜(CLIPScore/T-MARS/MetaCLIP/DFN 对照)
  任务C 时序/过程:GIFT-Eval 榜 + 时序基础模型(Chronos);TEP 全 21 故障 + 1D-CNN/LSTM 报 FDR
先跑 ./runall.sh setup && ./runall.sh smoke 确认环境,再 ./runall.sh local 复现全部本地结果。
EOF
}

case "${1:-}" in
  setup)  setup ;;
  smoke)  smoke ;;
  local)  local_suite ;;
  server) server_hint ;;
  *) echo "用法: ./runall.sh {setup|smoke|local|server}"; exit 1 ;;
esac
