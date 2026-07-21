"""Mechanical strict regrade of master_coverage.json PASS cells
(canonical parity gate). Run on the machine where logs/outputs live.

STRICT_PASS requires ALL of:
  1. every evidence log (brace patterns expanded) exists, contains a terminal marker
     (python_exit=0 / DONE / _OK), and has no Traceback/Killed/CUDA-OOM;
  2. json_path present, every expanded results.json exists and json.load succeeds;
  3. >= 3 seeds (families on the documented 2-seed flag list can never be STRICT);
  4. hash traceability: at least one JSON carries code_sha256_12 or config metadata.
Otherwise:
  RECOVERED_RESULT_ONLY  if no parseable JSON at all (log-recovered numbers);
  PASS_WEAK              for any other missing criterion.
Grades are written in place ("grade", "grade_reason") plus a "regrade_summary" block.
Non-PASS cells (N-A / MISSING / SUPERSEDED) are left untouched.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC = os.path.join(ROOT, "experiments", "master_coverage.json")
BAD = re.compile(r"Traceback \(most recent call last\)|(?<![A-Za-z])Killed|CUDA out of memory")
TERM = re.compile(r"python_exit=0|DONE|_OK")


def expand(pat):
    """Expand one {a,b,c} brace group; return [pat] if none."""
    m = re.search(r"\{([^{}]+)\}", pat)
    if not m:
        return [pat]
    out = []
    for alt in m.group(1).split(","):
        out.extend(expand(pat[:m.start()] + alt + pat[m.end():]))
    return out


def paths_of(field):
    if not field:
        return []
    items = field if isinstance(field, list) else [field]
    out = []
    for it in items:
        out.extend(expand(it))
    return out


_logcache = {}


def check_log(p):
    """(exists, has_terminal, bad_line_or_None)"""
    if p in _logcache:
        return _logcache[p]
    fp = os.path.join(ROOT, p)
    if not os.path.exists(fp):
        r = (False, False, None)
    else:
        txt = open(fp, errors="replace").read()
        bad = BAD.search(txt)
        r = (True, bool(TERM.search(txt)), bad.group(0) if bad else None)
    _logcache[p] = r
    return r


def main():
    d = json.load(open(MC))
    two_seed_families = set()
    for flag in d.get("seed_flags", []):
        for fam in d.get("cells", {}):
            if fam in str(flag):
                two_seed_families.add(fam)
    counts = {"STRICT_PASS": 0, "PASS_WEAK": 0, "RECOVERED_RESULT_ONLY": 0}
    for fam, methods in d.get("cells", {}).items():
        for meth, cell in methods.items():
            if cell.get("status") != "PASS":
                continue
            reasons = []
            # 1. logs
            logs = paths_of(cell.get("evidence_log"))
            if not logs:
                reasons.append("no evidence log listed")
            for lp in logs:
                ex, term, bad = check_log(lp)
                if not ex:
                    reasons.append("log missing: %s" % lp)
                elif bad:
                    reasons.append("log has %s: %s" % (bad, lp))
                elif not term:
                    reasons.append("no terminal marker in %s" % lp)
            # 2. JSONs
            jsons = paths_of(cell.get("json_path"))
            parseable = 0
            hash_ok = False
            for jp in jsons:
                fp = os.path.join(ROOT, jp)
                if not os.path.exists(fp):
                    reasons.append("json missing: %s" % jp)
                    continue
                try:
                    obj = json.load(open(fp))
                    parseable += 1
                    if isinstance(obj, dict) and ("code_sha256_12" in obj or "config" in obj):
                        hash_ok = True
                except Exception as e:
                    reasons.append("json unparseable: %s (%s)" % (jp, e))
            if not jsons:
                reasons.append("no canonical JSON (log-recovered only)")
            # 3. seeds
            if len(cell.get("seeds", [])) < 3 or fam in two_seed_families:
                reasons.append("fewer than 3 seeds (or documented 2-seed family)")
            # 4. hash
            if jsons and parseable and not hash_ok:
                reasons.append("no code/config hash metadata in JSON")
            if not reasons:
                grade = "STRICT_PASS"
            elif not jsons or parseable == 0:
                grade = "RECOVERED_RESULT_ONLY"
            else:
                grade = "PASS_WEAK"
            cell["grade"] = grade
            if reasons:
                cell["grade_reason"] = reasons
            counts[grade] += 1
    total = sum(counts.values())
    d["regrade_summary"] = {
        "date": "2026-07-16", "audit": "CANONICAL_PARITY_GATE item 3 (mechanical script)",
        "total_pass_cells": total, **counts,
        "strict_completion_rate": round(counts["STRICT_PASS"] / total, 3) if total else None,
        "rule": "only STRICT_PASS counts toward strict completion; weak evidence is disclosed, not counted"}
    json.dump(d, open(MC, "w"), indent=1)
    print(json.dumps(d["regrade_summary"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
