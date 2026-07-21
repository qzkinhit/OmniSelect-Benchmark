#!/usr/bin/env bash
# post_audit2_prepare.sh -- audit 二 closeout staging (env snapshot + canonical-scope
# manifest + dual-end SHA-verified local backup + all-pass markers).
#
# WRITTEN 2026-07-17 during audit 二. NOT EXECUTED YET.
#
# ============================ IMPORTANT / DO NOT SKIP ============================
# * FINAL numbers and the final markers WAIT FOR THE QZ4 FREEZE: run this script only
#   AFTER /root/lane_text_qz4.done exists (QZ4 lane run_id text-qz4-20260717T0933,
#   PIDs 805023/805182, guard 806605) and its zip/quadmix_pub cells have been absorbed
#   into canonical_tables.json / master_coverage.json. Anything captured before the
#   freeze is a PRE-FREEZE snapshot and must not be marked final.
# * READ-ONLY on the server side except the two explicitly created outputs:
#   experiments/env_snapshot_postaudit2/ and experiments/POST_AUDIT2_manifest.sha256.
#   Never overwrite existing evidence dirs; never touch the QZ4 processes, code tree,
#   caches or configs while the lane is alive.
# * QZ3 history: the QZ3 attempt (text-qz3-20260716T2051) ended NOT_COMPLETED_BLOWUP,
#   evidence /root/qz3_blowup_evidence -- that dir is part of the manifest scope.
set -euo pipefail

SSH_ALIAS="omni"
R="/root/autodl-tmp/OmniSelect"                     # server repo
L="/Users/qianzekai/PycharmProjects/Paper2_OmniSelect"  # local repo
STAMP="$(date +%Y%m%dT%H%M%S)"
BACKUP="${L}/server_backup_postaudit2_${STAMP}"     # timestamped, never reused

# Refuse to run while QZ4 is alive / not frozen.
if ! ssh "$SSH_ALIAS" "test -f /root/lane_text_qz4.done"; then
  echo "ABORT: /root/lane_text_qz4.done not present -- QZ4 not frozen yet." >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# (a) Server environment snapshot -> experiments/env_snapshot_postaudit2/
#     (new dir; refuses to overwrite an existing snapshot)
# ---------------------------------------------------------------------------
ssh "$SSH_ALIAS" bash -s <<EOF
set -euo pipefail
SNAP="$R/experiments/env_snapshot_postaudit2"
if [ -e "\$SNAP" ]; then echo "env snapshot already captured at \$SNAP (kept, not overwritten)"; exit 0; fi
mkdir -p "\$SNAP"
"$R/.venv/bin/python" -m pip freeze > "\$SNAP/pip_freeze.txt" 2>/dev/null \
  || /root/miniconda3/bin/python -m pip freeze > "\$SNAP/pip_freeze.txt"
nvidia-smi > "\$SNAP/nvidia_smi.txt" 2>&1 || echo "nvidia-smi unavailable" > "\$SNAP/nvidia_smi.txt"
uname -a > "\$SNAP/uname.txt"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "\$SNAP/captured_utc.txt"
echo "env snapshot -> \$SNAP"
EOF

# ---------------------------------------------------------------------------
# (b0) ACTIVE GATE ASSERTIONS (audit 1156 item 二.5) - any failure aborts, no marker:
#   coverage top-level counts == mechanical cells recount; RUNNING=0; registry empty;
#   MISSING=0; server coverage SHA == local HEAD coverage SHA; stats source present.
# ---------------------------------------------------------------------------
LOCAL_COV_SHA=$(shasum -a 256 "$L/experiments/master_coverage.json" | awk '{print $1}')
ssh "$SSH_ALIAS" /root/miniconda3/bin/python - "$LOCAL_COV_SHA" <<'PYGATE'
import hashlib, json, sys
from collections import Counter
local_sha = sys.argv[1]
p = "/root/autodl-tmp/OmniSelect/experiments/master_coverage.json"
raw = open(p, "rb").read()
assert hashlib.sha256(raw).hexdigest() == local_sha, "server coverage SHA != local HEAD coverage SHA"
d = json.loads(raw)
sc = Counter(c.get("status") for f in d["cells"].values() for c in f.values())
gc = Counter(c.get("grade") for f in d["cells"].values() for c in f.values() if c.get("grade"))
rs = d["regrade_summary"]
assert rs["total_pass_cells"] == sc.get("PASS", 0), "summary PASS != cells"
assert rs["STRICT_PASS"] == gc.get("STRICT_PASS", 0), "summary STRICT != cells"
assert rs["PASS_WEAK"] == gc.get("PASS_WEAK", 0), "summary WEAK != cells"
assert sc.get("RUNNING", 0) == 0, "RUNNING cells remain"
assert sc.get("MISSING", 0) == 0, "MISSING cells remain"
assert not d["running_registry"]["entries"], "running_registry not empty"
import os
assert os.path.exists("/root/autodl-tmp/OmniSelect/experiments/text_controls_stats.json"), "stats source missing on server"
print("GATE ASSERTIONS ALL PASS: PASS=%d STRICT=%d WEAK=%d coverage-SHA bound to local HEAD" %
      (sc.get("PASS", 0), gc.get("STRICT_PASS", 0), gc.get("PASS_WEAK", 0)))
PYGATE

# ---------------------------------------------------------------------------
# (b) Server-side manifest of the canonical scope, per-file SHA256
#     -> experiments/POST_AUDIT2_manifest.sha256
#     Scope: canonical tables, every run_id= results.json, experiments/*.log,
#     ledgers + provenance docs, both manuscript PDFs+sources, runner + batch
#     scripts, config JSONs, raw-data SHA lists, caches inventory, QZ3 blowup
#     evidence, QZ4 done marker + log.
# ---------------------------------------------------------------------------
ssh "$SSH_ALIAS" bash -s <<EOF
set -euo pipefail
cd "$R"
MAN="$R/experiments/POST_AUDIT2_manifest.sha256"
if [ -e "\$MAN" ]; then echo "ABORT: \$MAN already exists (never overwrite)"; exit 3; fi
{
  # canonical tables + coverage/verdict/stats JSONs
  ls experiments/canonical_tables.json experiments/master_coverage.json \
     experiments/selection_manifest_verdicts.json experiments/text_controls_stats.json \
     experiments/published_core_paired_*.json experiments/controller_current_canonical_v5.json \
     experiments/results_matrix.json 2>/dev/null || true
  # every run_id= results.json (all arms/views)
  find outputs -type f -name results.json -path "*run_id=*"
  # experiment logs + ledgers + provenance docs
  find experiments -maxdepth 1 -type f -name "*.log"
  find docs -type f -name "*.md"
  # manuscripts live ONLY on the local machine - hashed in the LOCAL manifest
  # complement (stage c2), not here. Guard optional dirs so set -e survives.
  # runner + batch scripts, config JSONs
  find scripts -type f \( -name "*.py" -o -name "*.sh" \) || true
  [ -d configs ] && find configs -type f -name "*.json" || true
  # raw-data SHA lists (adjust to actual filenames)
  find . -maxdepth 3 -type f \( -name "*sha256*.txt" -o -name "*_sha_list*" \) -not -path "./.git/*" 2>/dev/null
  # lane markers + QZ3 blowup evidence (absolute paths outside the repo)
  ls /root/lane_text_qz4.done /root/qz3_blowup_evidence/* 2>/dev/null || true
} | sort -u | while read -r f; do sha256sum "\$f"; done > "\$MAN"
# caches inventory: names+sizes only (contents NEVER hashed nor pulled; do not touch)
find outputs -maxdepth 4 -type d -name "*cache*" -exec du -sh {} \; \
  > "$R/experiments/env_snapshot_postaudit2/caches_inventory.txt" 2>/dev/null || true
wc -l "\$MAN"
EOF

# ---------------------------------------------------------------------------
# (c) Timestamped local backup dir + scp pull (manifest-driven, never overwrite)
# ---------------------------------------------------------------------------
mkdir "$BACKUP"   # fails if it exists: never overwrite an evidence dir
scp -q "$SSH_ALIAS:$R/experiments/POST_AUDIT2_manifest.sha256" "$BACKUP/"
# pull exactly the manifested files, preserving relative layout
awk '{ $1=""; sub(/^ /,""); print }' "$BACKUP/POST_AUDIT2_manifest.sha256" | while read -r f; do
  case "$f" in
    /*) dest="$BACKUP/rootfs${f}" ;;      # absolute server paths (markers, blowup evidence)
    *)  dest="$BACKUP/repo/${f}" ;;
  esac
  mkdir -p "$(dirname "$dest")"
  scp -q "$SSH_ALIAS:$( [ "${f#/}" = "$f" ] && echo "$R/" )${f}" "$dest"
done
scp -qr "$SSH_ALIAS:$R/experiments/env_snapshot_postaudit2" "$BACKUP/repo/experiments/"

# (c2) LOCAL manifest complement: manuscripts (sources + PDFs) exist only locally.
LOCAL_MAN="$BACKUP/POST_AUDIT2_local_manifest.sha256"
( cd "$L" && find papers/mmdataselect/submissions -type f \
    \( -name "*.tex" -o -name "*.pdf" \) -exec shasum -a 256 {} \; ) > "$LOCAL_MAN"
echo "local manuscript manifest: $(wc -l < "$LOCAL_MAN") files"

# ---------------------------------------------------------------------------
# (d) Dual-end per-file SHA comparison -- ALL files must match
# ---------------------------------------------------------------------------
FAIL=0
while read -r sha f; do
  case "$f" in
    /*) local_f="$BACKUP/rootfs${f}" ;;
    *)  local_f="$BACKUP/repo/${f}" ;;
  esac
  local_sha=$(shasum -a 256 "$local_f" | awk '{print $1}')
  if [ "$sha" != "$local_sha" ]; then echo "SHA MISMATCH: $f"; FAIL=1; fi
done < "$BACKUP/POST_AUDIT2_manifest.sha256"

# ---------------------------------------------------------------------------
# (e) Markers ONLY on all-pass (both ends). No pass -> no marker, exit nonzero.
# ---------------------------------------------------------------------------
if [ "$FAIL" -eq 0 ]; then
  date -u +"%Y-%m-%dT%H:%M:%SZ ALL_SHA_MATCH manifest=$(wc -l < "$BACKUP/POST_AUDIT2_manifest.sha256") files backup=$BACKUP" \
    > "$BACKUP/POST_AUDIT2_BACKUP_VERIFIED_OK"
  ssh "$SSH_ALIAS" "date -u +'%Y-%m-%dT%H:%M:%SZ ALL_SHA_MATCH pulled_to=$BACKUP' > $R/experiments/POST_AUDIT2_BACKUP_VERIFIED_OK"
  echo "ALL PASS -- markers written (local $BACKUP/POST_AUDIT2_BACKUP_VERIFIED_OK, server experiments/POST_AUDIT2_BACKUP_VERIFIED_OK)"
else
  echo "SHA comparison FAILED -- no markers written; investigate before rerunning (use a NEW timestamped backup dir)." >&2
  exit 1
fi
