#!/usr/bin/env bash
# One-shot wrapper: select -> train -> eval.
#   bash run_mmdataselect/run.sh --config configs/experiments/demo_select.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${HERE}/run_pipeline.py" "$@"
