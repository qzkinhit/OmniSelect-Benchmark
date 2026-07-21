#!/usr/bin/env python3
"""Mechanical claim-evidence checker for the AAAI draft (audit 2100-4.4 / 2257-4.2).

Reads docs/claim_evidence_matrix.json, re-resolves every binding against the
mandated evidence files, compares each claimed number to the bound value, and
verifies that every quote exists at (or near, +/-3 lines) the stated line of
the current tex. Prints per-claim PASS/FAIL and a per-tier summary; exits
nonzero if any bound claim fails. Entries in the "unbound" array are printed
for transparency but are not failures (each carries a reason).

Number match rule (any one suffices):
  1. |bound - claimed| < 5e-4
  2. '%.<n>f' % bound == claimed string   (n = claimed decimal places)
  3. Decimal(str(bound)) quantized HALF_UP to the claim's precision == claim
  4. string-valued binding (e.g. a fmt cell): claimed string is a substring

This script only reads; it never writes to tex or evidence files.
"""

import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MATRIX_PATH = REPO / "docs" / "claim_evidence_matrix.json"
QUOTE_WINDOW = 3  # lines above/below the stated line searched for the quote

_json_cache = {}


def load_json(rel):
    if rel not in _json_cache:
        _json_cache[rel] = json.loads((REPO / rel).read_text(encoding="utf-8"))
    return _json_cache[rel]


def resolve_pointer(binding):
    """Resolve 'relative/file.json#/a/b/c' to the referenced value."""
    rel, _, pointer = binding.partition("#")
    obj = load_json(rel)
    for seg in pointer.strip("/").split("/"):
        if isinstance(obj, list):
            obj = obj[int(seg)]
        elif isinstance(obj, dict):
            if seg not in obj:
                raise KeyError(f"{binding}: segment '{seg}' not found")
            obj = obj[seg]
        else:
            raise KeyError(f"{binding}: cannot descend into {type(obj).__name__}")
    return obj


def resolve_binding(b):
    """Resolve a binding item: pointer string or {op:...} object."""
    if isinstance(b, str):
        return resolve_pointer(b)
    if isinstance(b, dict):
        op = b.get("op")
        if op == "mean_seeds":
            raw = resolve_pointer(b["path"])
            if isinstance(raw, dict):
                vals = [raw[k] for k in sorted(raw)]
            else:
                vals = list(raw)
            if not vals:
                raise ValueError(f"mean_seeds over empty container: {b['path']}")
            # deterministic decimal aggregation (audit 0048): binary float summation is
            # interpreter-dependent (Python 3.8 sums [1.0163,0.9986,1.0256] to
            # 1.0134999999999998, 3.12 to 1.0135). Decimal(str(v)) recovers each value's
            # shortest decimal representation exactly as written in the JSON, and the
            # Decimal sum/divide is exact, so every interpreter binds the same mean.
            dvals = [Decimal(str(v)) for v in vals]
            return sum(dvals, Decimal(0)) / Decimal(len(dvals))
        if op == "sub":
            a = resolve_binding(b["a"])
            bb = resolve_binding(b["b"])
            return Decimal(str(a)) - Decimal(str(bb))
        raise ValueError(f"unknown op: {op}")
    raise ValueError(f"bad binding: {b!r}")


def number_matches(claim_str, value):
    """Apply the match rules; returns (ok, detail)."""
    if isinstance(value, str):
        ok = claim_str in value
        return ok, f"string binding {value!r}"
    # decimal half-up FIRST (deterministic across interpreters, audit 0048); the
    # float tolerance and %-format paths below are fallbacks only.
    dv = value if isinstance(value, Decimal) else Decimal(str(value))
    try:
        q = dv.quantize(Decimal(claim_str), rounding=ROUND_HALF_UP)
        if q == Decimal(claim_str):
            return True, f"bound={dv} half-up rounds to {q}"
    except Exception:
        pass
    v = float(dv)
    c = float(claim_str)
    if abs(v - c) < 5e-4:
        return True, f"bound={v!r} |diff|<5e-4"
    dp = len(claim_str.split(".")[1]) if "." in claim_str else 0
    fmt = f"%.{dp}f" % v
    if fmt == claim_str or fmt.lstrip("0") == claim_str.lstrip("0"):
        return True, f"bound={v!r} formats to {fmt}"
    try:
        q2 = Decimal(str(v)).quantize(Decimal(claim_str), rounding=ROUND_HALF_UP)
        if q2 == Decimal(claim_str):
            return True, f"bound={v!r} half-up rounds to {q2}"
    except Exception:
        pass
    return False, f"bound={v!r} != claimed {claim_str}"


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def quote_found(file_rel, line_no, quote):
    lines = (REPO / file_rel).read_text(encoding="utf-8").splitlines()
    lo = max(0, line_no - 1 - QUOTE_WINDOW)
    hi = min(len(lines), line_no + QUOTE_WINDOW)
    window = norm(" ".join(lines[lo:hi]))
    return norm(quote) in window


def run_comparison(cmp_spec):
    t = cmp_spec["type"]
    if t == "substring":
        val = resolve_pointer(cmp_spec["path"])
        ok = cmp_spec["expect"] in str(val)
        return ok, f"substring '{cmp_spec['expect']}' in {val!r}"
    if t == "within":
        target = float(resolve_binding(cmp_spec["target"]))
        other = float(resolve_binding(cmp_spec["other"]))
        tol = cmp_spec["tol"]
        tol = float(resolve_binding(tol)) if isinstance(tol, str) else float(tol)
        ok = abs(target - other) <= tol + 1e-12
        return ok, f"|{target}-{other}|={abs(target-other):.4f} <= tol {tol}"
    target = float(resolve_binding(cmp_spec["target"]))
    among = [float(resolve_binding(p)) for p in cmp_spec.get("among", [])]
    lower_better = cmp_spec.get("direction", "higher") == "lower"
    if t == "better":
        than = float(resolve_binding(cmp_spec["than"]))
        ok = target < than if lower_better else target > than
        return ok, f"target={target} vs than={than} ({cmp_spec['direction']} better)"
    if t == "rank":
        ordered = sorted(among, reverse=not lower_better)
        # rank of target value within the listed set (1 = best)
        rank = 1 + sum(1 for v in ordered if (v < target if lower_better else v > target))
        ok = rank == cmp_spec["expect_rank"]
        return ok, f"target={target} rank={rank} expected={cmp_spec['expect_rank']} among {ordered}"
    if t == "not_worst":
        worst = max(among) if lower_better else min(among)
        ok = (target < worst) if lower_better else (target > worst)
        return ok, f"target={target} worst={worst}"
    raise ValueError(f"unknown comparison type: {t}")


def _selftest():
    """Fixed regression (audit 0048): the ETTm1 three-seed mean must bind 1.014 on
    EVERY interpreter (Python 3.8 binary sum yields 1.0134999999999998 -> 1.013)."""
    vals = [1.0163, 0.9986, 1.0256]
    dmean = sum((Decimal(str(v)) for v in vals), Decimal(0)) / Decimal(len(vals))
    assert dmean == Decimal("1.0135"), f"decimal mean drifted: {dmean}"
    ok, detail = number_matches("1.014", dmean)
    assert ok, f"regression: 1.0135 must half-up bind 1.014 ({detail})"
    ok2, _ = number_matches("1.013", dmean)
    assert not ok2, "regression: 1.0135 must NOT bind 1.013 under half-up"


def main():
    _selftest()
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    entries = matrix["entries"]
    unbound = matrix.get("unbound", [])

    failures = 0
    tier_counts = {}
    tier_fails = {}

    print(f"claim-evidence check: {len(entries)} bound entries, {len(unbound)} unbound")
    print("=" * 78)

    for e in entries:
        cid = e["claim_id"]
        tier = e.get("tier", "?")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        problems = []

        # 1. quote at/near stated line
        try:
            if not quote_found(e["file"], e["line"], e["quote"]):
                problems.append(f"quote not found near {e['file']}:{e['line']}")
        except FileNotFoundError:
            problems.append(f"tex file missing: {e['file']}")

        # 2. numbers vs bindings
        numbers = e.get("numbers", [])
        binding = e.get("binding", [])
        if isinstance(binding, (str, dict)):
            binding = [binding]
        if len(numbers) != len(binding):
            problems.append(f"numbers ({len(numbers)}) and binding ({len(binding)}) lengths differ")
        else:
            for n, b in zip(numbers, binding):
                try:
                    val = resolve_binding(b)
                except Exception as exc:
                    problems.append(f"binding unresolvable for {n}: {exc}")
                    continue
                ok, detail = number_matches(n, val)
                if not ok:
                    btxt = b if isinstance(b, str) else json.dumps(b)
                    problems.append(f"number {n}: {detail} [{btxt}]")

        # 3. comparisons
        for i, cmp_spec in enumerate(e.get("comparisons", [])):
            try:
                ok, detail = run_comparison(cmp_spec)
            except Exception as exc:
                ok, detail = False, f"error: {exc}"
            if not ok:
                problems.append(f"comparison[{i}] ({cmp_spec['type']}): {detail}")

        if problems:
            failures += 1
            tier_fails[tier] = tier_fails.get(tier, 0) + 1
            print(f"FAIL  [{tier:>17}] {cid}")
            for p in problems:
                print(f"      - {p}")
        else:
            nums = f" ({len(numbers)} numbers)" if numbers else ""
            cmps = f" ({len(e.get('comparisons', []))} comparisons)" if e.get("comparisons") else ""
            print(f"PASS  [{tier:>17}] {cid}{nums}{cmps}")

    print("=" * 78)
    for u in unbound:
        print(f"UNBOUND [{u.get('tier','?'):>15}] {u['claim_id']} @ {u['file']}:{u['line']}")
        print(f"        numbers={u.get('numbers', [])}")
        print(f"        reason: {u['reason']}")

    print("=" * 78)
    print("summary by tier (bound entries):")
    for tier in sorted(tier_counts):
        n = tier_counts[tier]
        f = tier_fails.get(tier, 0)
        print(f"  {tier:>17}: {n - f}/{n} PASS")
    print(f"total: {len(entries) - failures}/{len(entries)} PASS, {failures} FAIL, {len(unbound)} UNBOUND (documented)")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
