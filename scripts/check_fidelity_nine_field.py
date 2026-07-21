"""Mechanical gate for the baseline-fidelity nine-field anchor table (audit 1620).

baseline_fidelity_evidence_closure may read COMPLETE_WITH_DISCLOSED_TIERS only while
this check PASSes: the table in docs/baseline_fidelity_ledger.md §0.5 must have all
nine data fields non-empty for every baseline row, and the standing labels that must
never disappear are asserted verbatim.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "docs", "baseline_fidelity_ledger.md")
N_FIELDS = 9  # total columns; field (1) is the baseline/venue name itself

MANDATORY_LABELS = [
    "last-layer gradient-norm PROXY",           # GraNd
    "published-core transfer",                  # QuaDMix quadmix_pub
    "published-update transfer",                # DMF dmf_pub
    "PROTOCOL_INVALID_DUPLICATE_IDS",           # withdrawn style proxy
    "does NOT upgrade the local tier",          # CCS anchor vs local two tiers
    "strict_original_protocol_reproduction = NONE",
    "baseline_fidelity_evidence_closure = COMPLETE_WITH_DISCLOSED_TIERS",
]


def main():
    text = open(LEDGER).read()
    m = re.search(r"## 0\.5 .*?\n(.*?)\n## 0\.", text, re.S)
    if not m:
        print("FAIL: nine-field section not found")
        return 1
    rows = [l for l in m.group(1).splitlines()
            if l.startswith("|") and not set(l) <= set("|- ") and "baseline (venue)" not in l]
    bad = 0
    for l in rows:
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if len(cells) != N_FIELDS or any(not c for c in cells):
            print("FAIL row:", cells[0] if cells else l[:40],
                  "(cols=%d, empties=%d)" % (len(cells), sum(1 for c in cells if not c)))
            bad += 1
    for lab in MANDATORY_LABELS:
        if lab not in text:
            print("FAIL missing mandatory label:", lab)
            bad += 1
    print("rows checked:", len(rows))
    if bad == 0 and len(rows) >= 12:
        print("PASS: nine-field anchor table complete; closure flag COMPLETE_WITH_DISCLOSED_TIERS is supported")
        return 0
    print("FAIL: closure flag must be reverted to INCOMPLETE until fixed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
