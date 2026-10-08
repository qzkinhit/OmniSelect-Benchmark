#!/bin/bash
# OmniSelect-Benchmark entry point. Delegates to run_omniselect/run.sh with all arguments.
#
#   bash run.sh setup                 install the package, run the CPU tests, verify the committed data checksums
#   bash run.sh smoke                 one small cell per track under protocol v2_2, then recompute and replay
#   bash run.sh local                 the cells of TRACK_CELLS at full size on this machine
#   bash run.sh queue FILE [FLAGS]    run a queue file (python -m run_omniselect.run_queue FILE FLAGS)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_omniselect/run.sh" "$@"
