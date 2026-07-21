"""Dual-end SHA verification: every file in the server manifest must exist in the local
backup with an identical SHA256. Emits per-group pass/fail and a final verdict."""
import hashlib
import json
import os
import sys

BK = sys.argv[1]
manifest = json.load(open(os.path.join(BK, "manifest.json")))
SR = manifest["root"]  # /root/autodl-tmp/OmniSelect


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def localpath(server_path):
    if server_path.startswith(SR + "/"):
        return os.path.join(BK, "omni", server_path[len(SR) + 1:])
    if server_path.startswith("/root/"):
        return os.path.join(BK, "root", os.path.basename(server_path))
    return None


total = ok = missing = mismatch = 0
fails = []
for g, entries in manifest["groups"].items():
    gok = gmiss = gmis = 0
    for sp, meta in entries.items():
        total += 1
        lp = localpath(sp)
        if not lp or not os.path.isfile(lp):
            missing += 1
            gmiss += 1
            fails.append(("MISSING", sp))
            continue
        if sha256(lp) != meta["sha256"]:
            mismatch += 1
            gmis += 1
            fails.append(("MISMATCH", sp))
            continue
        ok += 1
        gok += 1
    print(f"  {g:20} ok={gok:4} missing={gmiss} mismatch={gmis}")

print(f"\nTOTAL={total} OK={ok} MISSING={missing} MISMATCH={mismatch}")
for kind, p in fails[:20]:
    print(f"  {kind}: {p}")
if missing == 0 and mismatch == 0 and total == manifest["total_files"]:
    print("BACKUP_VERIFIED_ALL_MATCH")
    sys.exit(0)
print("BACKUP_VERIFY_FAILED")
sys.exit(1)
