#!/bin/bash
# Rebuild the paper tables and figure statistics from published records, without training.
# Requires Python 3.10+ with numpy and scipy. Override PYTHON to select an environment.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
rm -rf work
mkdir -p work
cd work
for archive in ../records/*.tar.gz; do tar xzf "$archive"; done
S=../scripts
"$PYTHON" "$S/build_all_seed_results.py"
"$PYTHON" "$S/aggregate_partial.py" paper_main_stats.json paper_main_agg.json > /dev/null
"$PYTHON" "$S/aggregate_partial.py" paper_nocoop_stats.json paper_nocoop_agg.json > /dev/null
"$PYTHON" "$S/compute_table_method_ranks.py" > /dev/null
"$PYTHON" "$S/scale_up.py"
"$PYTHON" "$S/make_main_table.py" > /dev/null
"$PYTHON" "$S/paper_numbers.py" > /dev/null
"$PYTHON" "$S/adoption_numbers.py" > /dev/null
"$PYTHON" "$S/build_condition_results.py"
"$PYTHON" "$S/paired_v23.py"
"$PYTHON" "$S/compare_tables.py" ../tables .
