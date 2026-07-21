#!/usr/bin/env bash
# Scaffold a new dataset / modality arm from an existing arm runner.
# This copies a runner into scripts/ under a labeled name and prints the 4-step
# checklist. It does NOT wire anything automatically — the hard invariant (every
# baseline passed into the controller as a candidate) is yours to keep.
#
# Usage:  run_scripts/add_dataset.sh <new_name> [template_arm]
#   <new_name>       e.g. my_sensor   -> scripts/run_my_sensor_experiment.py
#   [template_arm]   vision|timeseries|tep|tabular  (default: timeseries)
set -euo pipefail
cd "$(dirname "$0")/.."

NEW="${1:?new dataset/modality name required, e.g. my_sensor}"
TPL="${2:-timeseries}"
SRC="scripts/run_${TPL}_experiment.py"
DST="scripts/run_${NEW}_experiment.py"

[ -f "$SRC" ] || { echo "template not found: $SRC" >&2; exit 2; }
[ -e "$DST" ] && { echo "already exists: $DST (refusing to overwrite)" >&2; exit 3; }

cp "$SRC" "$DST"
echo "created $DST  (from $SRC)"
cat <<EOF

Next steps (see docs/CONTRIBUTING_DATASETS.md for the full template):
  1. In $DST: replace the dataset loader and the controlled-noise recipe
     (NOISE_FRAC=0.40, per-record tags, RNG np.random.default_rng(seed+7)),
     and set the downstream model + primary metric for your modality.
  2. HARD INVARIANT: pass EVERY baseline into the controller as a candidate via
     AdaptiveController.select(..., extra_strategies=[fn(k)->indices, ...]).
     Copy the mmds_adapt block from the template verbatim. A baseline is a
     candidate, never a compare-only column.
  3. Emit results.json in the standard layout
     outputs/<arm>/<dataset>/run_id=...-<sorted tags>/seed_<N>/results.json
     with sel_sha12 / train_order_sha12, and dump the split manifest (SPLIT_EXPORT_DIR).
  4. Record source + pinned revision + SHA256 + noise recipe in
     docs/dataset_provenance.md and docs/ARTIFACTS_INDEX.md.

Then run it:
  run_scripts/run_single_arm.sh <arm> <dataset> 0     # once you register the arm name
  # or directly during development:
  SEED=0 METHODS=random,auth_only,mmds_adapt python $DST
EOF
