#!/usr/bin/env bash
# Formal seed-0 five-domain text batch. Runs every applicable main-table baseline
# and OmniSelect in one process, records three preregistered non-applicable rows,
# and keeps a full evidence bundle. It never writes results_canonical or invents
# a manuscript value.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT_OVERRIDE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
BATCH_ID="${BATCH_ID:-text-seed0-primary-$(date -u +%Y%m%dT%H%M%SZ)}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-$REPO_ROOT/evidence/$BATCH_ID}"
RUN_ID="v3-full-text-stratify1-main-$BATCH_ID"

REF_MODEL="${REF_MODEL:-HuggingFaceTB/SmolLM2-135M}"
REF_MODEL_REVISION="${REF_MODEL_REVISION:-93efa2f097d58c2a74874c7e644dbc9b0cee75a2}"
EXPECTED_POOL_SHA256="${EXPECTED_POOL_SHA256:-84e174dbb097288c6b4473af2af8d6cb46a0b00a000b6534545725e53f9939c5}"
EXPECTED_HELDOUT_SHA256="${EXPECTED_HELDOUT_SHA256:-1e4a45c9c959995a3c10c840dc2a8b84ce33bd82ba17682c8f50c3b4b3a1e785}"
EXPECTED_INFLUENCE_SHA256="${EXPECTED_INFLUENCE_SHA256:-fd5580790cf9782d38defcbafe7f1d8bdc16f8774a60f8dbb625855150ffcc82}"
INFLUENCE_CACHE="${INFLUENCE_CACHE:-$REPO_ROOT/data/processed/qpool_influence_84e174dbb097_SmolLM2-135M_c512_pplq.npz}"

METHODS="random,influence_only,coverage_text,fixed_fusion,herding_text,density_text,quadmix_pub,dmf_pub"
REFERENCES="random,influence_only,coverage_text,herding_text,density_text,quadmix_pub,dmf_pub"
CHALLENGERS="fixed_fusion"
TABLE_BASELINES="random,influence_only,coverage_text,fixed_fusion,herding_text,el2n,grand,ccs,density_text,quadmix_pub,dmf_pub"
SKIPPED_METHODS="el2n,grand,ccs"
TAG="run_id=$RUN_ID-stratify=1-infl=pplq-train=finetune-method_v3=1"
RESULT_PATH="$REPO_ROOT/outputs/experiment/$TAG/seed_0/results.json"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing executable PYTHON_BIN=$PYTHON_BIN" >&2
  exit 2
fi
if [[ -e "$EVIDENCE_ROOT" ]]; then
  echo "evidence directory already exists; choose a new BATCH_ID: $EVIDENCE_ROOT" >&2
  exit 3
fi
if [[ -e "$RESULT_PATH" ]]; then
  echo "result already exists; refusing overwrite: $RESULT_PATH" >&2
  exit 4
fi
mkdir -p "$EVIDENCE_ROOT"
printf 'batch_id=%s\nrun_id=%s\nstarted_utc=%s\n' \
  "$BATCH_ID" "$RUN_ID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >"$EVIDENCE_ROOT/STARTED"

failed() {
  local rc=$?
  printf 'exit_code=%s\nfailed_utc=%s\n' \
    "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$EVIDENCE_ROOT/FAILED"
  exit "$rc"
}
trap failed ERR

"$PYTHON_BIN" "$REPO_ROOT/scripts/validate_text_seed0_primary.py" preflight \
  --repo "$REPO_ROOT" \
  --pool-sha "$EXPECTED_POOL_SHA256" \
  --heldout-sha "$EXPECTED_HELDOUT_SHA256" \
  --influence-cache "$INFLUENCE_CACHE" \
  --influence-sha "$EXPECTED_INFLUENCE_SHA256" \
  --output "$EVIDENCE_ROOT/preflight.json"

env \
  CONFIG_PROBE=1 \
  METHOD_V3=1 \
  METHOD_V3_TEXT_MAX_PORTFOLIO=8 \
  METHOD_V3_TEXT_REFERENCE_METHODS="$REFERENCES" \
  METHOD_V3_TEXT_CHALLENGER_METHODS="$CHALLENGERS" \
  METHOD_V3_TEXT_TABLE_BASELINES="$TABLE_BASELINES" \
  METHOD_V3_TEXT_SKIPPED_METHODS="$SKIPPED_METHODS" \
  METHODS="$METHODS" \
  SEED=0 STRATIFY=1 INFL_KIND=pplq TRAIN_MODE=finetune \
  REF_MODEL="$REF_MODEL" REF_MODEL_REVISION="$REF_MODEL_REVISION" \
  BUDGET_FRAC=0.5 PASSES=2 CTX=512 BS=16 FT_LR=2e-5 TEXT_EMBED_BS=32 \
  "$PYTHON_BIN" -u "$REPO_ROOT/scripts/run_experiment.py" \
  >"$EVIDENCE_ROOT/config_probe.log" 2>&1

git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all >"$EVIDENCE_ROOT/git_status.txt"
git -C "$REPO_ROOT" rev-parse HEAD >"$EVIDENCE_ROOT/git_head.txt"
"$PYTHON_BIN" -m pip freeze >"$EVIDENCE_ROOT/pip_freeze.txt"
nvidia-smi -q >"$EVIDENCE_ROOT/nvidia_smi_q.txt"

env \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=0 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  METHOD_V3=1 \
  METHOD_V3_DELTA=0.05 \
  METHOD_V3_TEXT_CLIP=1.0 \
  METHOD_V3_TEXT_MAX_PORTFOLIO=8 \
  METHOD_V3_TEXT_REFERENCE_METHODS="$REFERENCES" \
  METHOD_V3_TEXT_CHALLENGER_METHODS="$CHALLENGERS" \
  METHOD_V3_TEXT_TABLE_BASELINES="$TABLE_BASELINES" \
  METHOD_V3_TEXT_SKIPPED_METHODS="$SKIPPED_METHODS" \
  METHODS="$METHODS" \
  RUN_ID="$RUN_ID" \
  SEED=0 \
  STRATIFY=1 \
  INFL_KIND=pplq \
  TRAIN_MODE=finetune \
  REF_MODEL="$REF_MODEL" \
  REF_MODEL_REVISION="$REF_MODEL_REVISION" \
  BUDGET_FRAC=0.5 \
  PASSES=2 \
  CTX=512 \
  BS=16 \
  TEXT_EMBED_BS=32 \
  FT_LR=2e-5 \
  FT_STEPS_CAP=0 \
  FT_FREEZE=0 \
  LM_EVAL=1 \
  LMEVAL_TASKS=arc_easy,arc_challenge,hellaswag,openbookqa \
  LMEVAL_LIMIT=0 \
  LMEVAL_BS="${LMEVAL_BS:-16}" \
  DMF_PROBE_TOKENS="${DMF_PROBE_TOKENS:-150000}" \
  DMF_ROUNDS=6 \
  DMF_ETA=0.5 \
  ONLY_DOMAIN= \
  REPORT_ALL_CANDIDATES=1 \
  LMEVAL_ALL_CANDIDATES=0 \
  SAVE_ALL_CANDIDATE_MODELS=1 \
  "$PYTHON_BIN" -u "$REPO_ROOT/scripts/run_experiment.py" \
  2>&1 | tee "$EVIDENCE_ROOT/run.log"

"$PYTHON_BIN" "$REPO_ROOT/scripts/validate_text_seed0_primary.py" result \
  --result "$RESULT_PATH" \
  --output "$EVIDENCE_ROOT/validated_result.json"

"$PYTHON_BIN" - "$RESULT_PATH" "$EVIDENCE_ROOT/SHA256SUMS" <<'PY'
import hashlib
import pathlib
import sys

result = pathlib.Path(sys.argv[1]).resolve()
output = pathlib.Path(sys.argv[2])
paths = [result]
bundle = result.parent / "repro_bundle"
paths.extend(sorted(path for path in bundle.rglob("*") if path.is_file()))
with output.open("w") as handle:
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        handle.write(f"{digest}  {path}\n")
PY

printf 'batch_id=%s\nresult=%s\ncompleted_utc=%s\n' \
  "$BATCH_ID" "$RESULT_PATH" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >"$EVIDENCE_ROOT/TEXT_SEED0_PRIMARY_OK"
trap - ERR
echo "PASS: $EVIDENCE_ROOT/TEXT_SEED0_PRIMARY_OK"
