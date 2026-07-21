#!/bin/sh
# gpu_gate.sh - STRICT GPU gate (audit 1910-2). Fail-closed by design:
#   - installs the exact audited cu128 stack (environment/constraints-cu128.txt);
#   - asserts torch.cuda.is_available() == True (exit 3 otherwise; NO CPU fallback);
#   - records torch / CUDA / driver versions;
#   - runs one tiny DLinear ETTh1 trial and asserts the runner actually used dev=cuda.
# The default clean-clone smoke (scripts/repro_smoke.sh) is CPU-scoped and never
# claims GPU coverage; only this gate may be cited as GPU evidence.
set -u

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT" || exit 1
VENV_DIR=${GPU_VENV_DIR:-.venv_gpu_gate}

echo "== gpu_gate step 0: venv + exact cu128 stack"
if [ ! -x "$VENV_DIR/bin/python" ]; then
    python3 -m venv "$VENV_DIR" || { echo "FAIL venv"; exit 1; }
fi
PY="$VENV_DIR/bin/python"
"$PY" -m pip install -q --upgrade pip || { echo "FAIL pip upgrade"; exit 1; }
"$PY" -m pip install -q torch==2.8.0 torchvision==0.23.0 \
    && "$PY" -m pip install -q -r requirements.txt -c environment/constraints-cu128.txt \
    && "$PY" -m pip install -q -e . pandas scikit-learn \
    || { echo "FAIL install (pinned 2.8.0 stack; constraints guard upgrades)"; exit 1; }
"$PY" -c "import torch; v=torch.__version__; assert v.startswith(\"2.8.0\"), v; print(\"torch\", v, \"cuda-build\", torch.version.cuda)" || { echo "FAIL version pin"; exit 1; }

echo "== gpu_gate step 1: fail-closed CUDA assertion + version record"
"$PY" - <<'EOF' || exit 3
import subprocess
import torch
ok = torch.cuda.is_available()
drv = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| driver", drv,
      "| cuda_available", ok)
assert ok, "torch.cuda.is_available() is False - GPU gate is fail-closed, refusing CPU fallback"
print("device0:", torch.cuda.get_device_name(0))
EOF
echo "PASS cuda_available=True"
echo "== resolved versions (cu128 lock)"
"$PY" -m pip freeze | grep -E "^(torch|torchvision|data-selection|numpy)" || true

echo "== gpu_gate step 2: tiny DLinear ETTh1 on cuda (runner must report dev=cuda)"
if [ ! -f data/processed/etth1.csv ]; then
    if [ "${GPU_GATE_ALLOW_DOWNLOAD:-0}" = "1" ]; then
        echo "data/processed/etth1.csv absent; the runner will download it from the pinned upstream (revision in docs/ARTIFACTS_INDEX.md) and the file SHA is recorded below"
    else
        echo "FAIL data/processed/etth1.csv absent and GPU_GATE_ALLOW_DOWNLOAD!=1 (raw data not in git)"; exit 4
    fi
fi
OUT=$(RUN_ID="gpu-gate-$(date +%Y%m%dT%H%M%S)" SEED=0 METHODS=random,auth_only,mmds_adapt \
    TS_DATASET=ETTh1 TS_MODEL=dlinear POOL_N=300 VAL_N=100 TEST_N=150 L=96 H=24 \
    NOISE_FRAC=0.4 BUDGET_FRAC=0.3 EPOCHS=5 PYTHONPATH=src \
    "$PY" scripts/run_timeseries_experiment.py 2>&1) || { echo "$OUT" | tail -5; echo "FAIL runner"; exit 5; }
echo "$OUT" | head -2
echo "$OUT" | grep -q "dev=cuda" || { echo "FAIL runner did not report dev=cuda"; exit 6; }
[ -f data/processed/etth1.csv ] && sha256sum data/processed/etth1.csv 2>/dev/null | sed "s/^/downloaded-data sha256: /"
echo "PASS tiny run on dev=cuda"
echo "GPU_GATE_OK"
