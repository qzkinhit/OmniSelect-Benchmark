#!/bin/sh
# repro_smoke.sh - CPU-SCOPED reproducibility smoke for OmniSelect (audit 1910-2:
# this script never claims GPU coverage; arm runners may opportunistically use an
# accelerator if importable torch sees one, but PASS here is CPU-scoped evidence
# only. The strict fail-closed GPU gate is scripts/gpu_gate.sh with the exact
# audited cu128 lock in environment/constraints-cu128.txt.)
#
# Creates a venv, runs the pure-CPU sanity smoke, pytest, one TINY experiment per
# arm runner whose prerequisites are present (raw data are not in git, so absent
# arms are SKIPPED, not failed), then regenerates the canonical paper tables into
# a temp directory. No GPU required. Echoes PASS/FAIL/SKIP per step and exits
# nonzero if any step FAILs.
#
# Optional env:
#   SMOKE_ALLOW_DOWNLOAD=1  allow steps that must download from HF/OpenML
#   SMOKE_TEXT=1            run the text-lane smoke (heavy: full-pool influence
#                           pass with SmolLM2-135M; off by default)
#   VENV_DIR=<dir>          venv location (default .venv_smoke)

set -u

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT" || exit 1
VENV_DIR=${VENV_DIR:-.venv_smoke}
ALLOW_DL=${SMOKE_ALLOW_DOWNLOAD:-0}
RUN_ID_SMOKE="repro-smoke-$(date +%Y%m%dT%H%M%S)"
FAILS=0

say()  { printf '%s\n' "== $1"; }
pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; FAILS=$((FAILS + 1)); }
skip() { printf 'SKIP  %s (%s)\n' "$1" "$2"; }

# ---------------------------------------------------------------- step 0: venv
say "step 0: venv + install (requirements.txt + editable package + test deps)"
if [ ! -x "$VENV_DIR/bin/python" ]; then
    python3 -m venv "$VENV_DIR" || { fail "venv creation"; exit 1; }
fi
PY="$VENV_DIR/bin/python"
if "$PY" -m pip install -q --upgrade pip \
    && "$PY" -m pip install -q -r requirements.txt \
    && "$PY" -m pip install -q torch==2.8.0 torchvision==0.23.0 \
    && "$PY" -m pip install -q -e ".[dev]" \
    && "$PY" -m pip install -q scikit-learn pandas; then
    pass "environment install"
else
    fail "environment install"
    exit 1
fi
say "resolved versions (audit 2030: recorded so the clean-clone log binds the env)"
"$PY" -m pip freeze | grep -E "^(torch|torchvision|data-selection|numpy|pandas|scikit-learn|tabpfn)" || true

has_py() { "$PY" -c "import $1" >/dev/null 2>&1; }

# ------------------------------------------------- step 1: pure-CPU sanity smoke
say "step 1: pure-CPU sanity smoke (no downloads)"
if "$PY" scripts/sanity_smoke.py; then
    pass "scripts/sanity_smoke.py"
else
    fail "scripts/sanity_smoke.py"
fi

# ------------------------------------------------------------- step 2: pytest
say "step 2: pytest (CPU-only suite)"
if [ -d tests ]; then
    if "$PY" -m pytest -q; then pass "pytest"; else fail "pytest"; fi
else
    skip "pytest" "no tests/ directory"
fi

# ------------------------------------------- step 3: tiny per-arm experiments
# Shared tiny knobs; every run gets a SMOKE-suffixed RUN_ID and seeds 0 (or 7
# for vision, matching the small embedding cache naming used in-repo).
SMOKE_METHODS="random,auth_only,mmds_adapt"

say "step 3a: vision arm (CIFAR-100, frozen CLIP + linear probe)"
VIS_CACHE="data/processed/vision_cifar100_clip-vit-base-patch32_p300v120t150_s7.npz"
if ! has_py torch || ! has_py sklearn; then
    skip "vision" "torch/sklearn not importable"
elif [ ! -f "$VIS_CACHE" ] && { [ "$ALLOW_DL" != "1" ] || ! has_py transformers; }; then
    skip "vision" "no embedding cache and downloads not allowed (set SMOKE_ALLOW_DOWNLOAD=1)"
else
    if RUN_ID="${RUN_ID_SMOKE}-vision" SEED=7 METHODS="$SMOKE_METHODS" \
        VIS_DATASET=uoft-cs/cifar100 VIS_ENCODER=openai/clip-vit-base-patch32 \
        POOL_N=300 VAL_N=120 TEST_N=150 NOISE_FRAC=0.4 BUDGET_FRAC=0.5 KNN=15 \
        "$PY" scripts/run_vision_experiment.py; then
        pass "vision tiny run"
    else
        fail "vision tiny run"
    fi
fi

say "step 3b: time-series arm (ETTh1, DLinear)"
if ! has_py torch || ! has_py pandas; then
    skip "timeseries" "torch/pandas not importable"
elif [ ! -f data/processed/etth1.csv ] && [ "$ALLOW_DL" != "1" ]; then
    skip "timeseries" "data/processed/etth1.csv absent and downloads not allowed (the runner can fetch the pinned upstream with SMOKE_ALLOW_DOWNLOAD=1)"
else
    if RUN_ID="${RUN_ID_SMOKE}-ts" SEED=0 METHODS="$SMOKE_METHODS" \
        TS_DATASET=ETTh1 TS_MODEL=dlinear POOL_N=400 VAL_N=150 TEST_N=200 \
        L=96 H=24 NOISE_FRAC=0.4 BUDGET_FRAC=0.3 EPOCHS=5 \
        "$PY" scripts/run_timeseries_experiment.py; then
        pass "timeseries tiny run"
    else
        fail "timeseries tiny run"
    fi
fi

say "step 3c: process arm (TEP, RandomForest)"
if ! has_py sklearn; then
    skip "tep" "sklearn not importable"
elif [ ! -f data/tep/d00.dat ]; then
    skip "tep" "data/tep/*.dat absent (raw data not in git; see docs/ARTIFACTS_INDEX.md)"
else
    if RUN_ID="${RUN_ID_SMOKE}-tep" SEED=0 METHODS="$SMOKE_METHODS" \
        MODEL=rf N_FAULTS=5 POOL_N=400 VAL_N=200 TEST_N=300 \
        NOISE_FRAC=0.4 BUDGET_FRAC=0.3 \
        "$PY" scripts/run_tep_experiment.py; then
        pass "tep tiny run"
    else
        fail "tep tiny run"
    fi
fi

say "step 3d: tabular arm (OpenML electricity, RF stand-in for TabPFN)"
if ! has_py sklearn; then
    skip "tabular" "sklearn not importable"
elif [ ! -d "${HOME}/scikit_learn_data" ] && [ "$ALLOW_DL" != "1" ]; then
    skip "tabular" "no OpenML cache and downloads not allowed (set SMOKE_ALLOW_DOWNLOAD=1)"
else
    if RUN_ID="${RUN_ID_SMOKE}-tab" SEED=0 METHODS="$SMOKE_METHODS" \
        TAB_DATASET=electricity MODEL=rf POOL_N=400 VAL_N=300 TEST_N=300 \
        NOISE_FRAC=0.4 BUDGET_FRAC=0.5 \
        "$PY" scripts/run_tabular_experiment.py; then
        pass "tabular tiny run"
    else
        fail "tabular tiny run"
    fi
fi

say "step 3e: text arm (SmolLM2 lane)"
if [ "${SMOKE_TEXT:-0}" != "1" ]; then
    skip "text" "off by default: the runner has no pool-subsample knob and needs a full 25k-record influence pass (set SMOKE_TEXT=1; see docs/REPRODUCIBILITY.md section 4)"
elif ! has_py torch || ! has_py transformers; then
    skip "text" "torch/transformers not importable"
elif [ ! -f data/processed/qpool_train.jsonl ]; then
    skip "text" "data/processed/qpool_train.jsonl absent (derived pool; see docs/ARTIFACTS_INDEX.md)"
else
    if RUN_ID="${RUN_ID_SMOKE}-text" SEED=0 METHODS="random,mmdataselect" \
        TRAIN_MODE=scratch PASSES=0.5 CTX=128 MINI_HID=64 MINI_LAYERS=2 MINI_HEADS=2 BS=4 \
        "$PY" scripts/run_experiment.py; then
        pass "text tiny run"
    else
        fail "text tiny run"
    fi
fi

# ------------------------------- step 4: canonical table regeneration (temp dir)
say "step 4: canonical table regeneration into a temp root (no check mode exists)"
TMP_ROOT=$(mktemp -d)
mkdir -p "$TMP_ROOT/experiments"
if [ -f experiments/results_matrix.json ]; then
    cp experiments/results_matrix.json "$TMP_ROOT/experiments/"
    [ -f experiments/controller_current_canonical_v5.json ] \
        && cp experiments/controller_current_canonical_v5.json "$TMP_ROOT/experiments/"
    if [ -d outputs ]; then ln -s "$REPO_ROOT/outputs" "$TMP_ROOT/outputs"; else mkdir "$TMP_ROOT/outputs"; fi
    if OMNISELECT_ROOT="$TMP_ROOT" "$PY" scripts/canonical_paper_tables.py \
        && "$PY" - "$TMP_ROOT/experiments/canonical_tables.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("latex_legacy_main_table"), "legacy main table empty"
assert "latex_FINAL_main_table" in d, "FINAL table key missing"
print("regenerated canonical_tables.json is valid JSON with expected keys")
EOF
    then
        pass "canonical table regeneration"
        if [ -f experiments/canonical_tables.json ]; then
            if cmp -s "$TMP_ROOT/experiments/canonical_tables.json" experiments/canonical_tables.json; then
                printf 'NOTE  regenerated file is byte-identical to the committed one\n'
            else
                printf 'NOTE  regenerated file differs from the committed one - expected on a\n'
                printf '      clone without the full outputs/ tree (FINAL cells need the pubcore\n'
                printf '      run outputs, which ship in the release archive, not in git)\n'
            fi
        fi
    else
        fail "canonical table regeneration"
    fi
else
    skip "canonical table regeneration" "experiments/results_matrix.json absent"
fi
rm -rf "$TMP_ROOT"

# -------- step 4b: rebuild FINAL tables from the whitelisted results in git and
# prove numeric equality with the committed experiments/canonical_tables.json.
# A nonzero exit here is a FAIL (the FINAL four-arm tables must be rebuildable
# from results_canonical/ alone on a clean clone).
say "step 4b: FINAL-table rebuild from results_canonical/ whitelist (equality proof)"
if [ -d results_canonical ] && [ -f experiments/canonical_tables.json ]; then
    if "$PY" scripts/rebuild_canonical_from_whitelist.py; then
        pass "whitelist rebuild matches committed canonical_tables.json"
    else
        fail "whitelist rebuild matches committed canonical_tables.json"
    fi
else
    skip "whitelist rebuild" "results_canonical/ or experiments/canonical_tables.json absent"
fi

# ---------------------------------------------------------------------- report
say "summary"
if [ "$FAILS" -eq 0 ]; then
    printf 'ALL GREEN (failures: 0; SKIPs are acceptable on a fresh clone)\n'
    exit 0
else
    printf 'FAILURES: %s\n' "$FAILS"
    exit 1
fi
