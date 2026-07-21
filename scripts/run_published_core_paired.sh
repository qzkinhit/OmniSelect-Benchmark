#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/OmniSelect
RUN_ID=${RUN_ID:?RUN_ID is required}
LOG_A=/root/published_core_paired_${RUN_ID}_lane_a.log
LOG_B=/root/published_core_paired_${RUN_ID}_lane_b.log
MARKER=/root/PUBLISHED_CORE_PAIRED_MAIN_OK
rm -f "$MARKER"

RUN_ID=$RUN_ID bash scripts/run_published_core_paired_lane_a.sh >"$LOG_A" 2>&1 &
pid_a=$!
RUN_ID=$RUN_ID bash scripts/run_published_core_paired_lane_b.sh >"$LOG_B" 2>&1 &
pid_b=$!
printf 'lane_a_pid=%s lane_b_pid=%s run_id=%s\n' "$pid_a" "$pid_b" "$RUN_ID"

set +e
wait "$pid_a"; rc_a=$?
wait "$pid_b"; rc_b=$?
set -e
printf 'lane_a_exit=%s lane_b_exit=%s\n' "$rc_a" "$rc_b"
if [[ "$rc_a" -ne 0 || "$rc_b" -ne 0 ]]; then
  exit 1
fi

.venv/bin/python scripts/validate_published_core_paired.py \
  --run-id "$RUN_ID" --log "$LOG_A" --log "$LOG_B" --marker "$MARKER"
