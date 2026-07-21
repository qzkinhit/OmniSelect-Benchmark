"""Build the machine-readable baseline x dataset results matrix from real experiment logs
(audit item 3: coverage must not be declared from filenames; every cell carries per-seed
values + source log + log SHA + metric + direction). Run on the server where logs live.

Verification-grade ledger extensions (audit item C):
  per view: protocol_type, applicability_na, expected_seeds, parse_counts,
            conflicts (same (method,seed) with DIFFERENT values), duplicate_sections
            (same seed anchor seen twice in one log), config_hash (sha256 of the
            SOURCES tuple repr), exit_evidence (terminal markers grepped from the log).
  top level: "__server_code_state__" = {"head": git HEAD, "diff_sha256": sha256 of the
            full `git diff` output (dirty-tree fingerprint)}. The key contains "__" on
            purpose so characteristic_metrics_v2.py's aux-view filter skips it unchanged.

Output: experiments/results_matrix.json
  {dataset: {"metric":..., "higher_is_better":..., "source_log":..., "log_sha256":...,
             "methods": {method: {seed: value}}, ...ledger fields...},
   "__server_code_state__": {...}}
Missing cells stay missing - honesty over completeness. Conflicts/duplicate_sections
must be empty lists for the matrix to be trusted; the script prints a loud warning and
exits nonzero if any view violates that.
"""
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")

# (log, dataset_key, seed-anchor regex, metric key regex, higher_is_better)
SOURCES = [
    # PRIMARY main views = the *_full_3seed.log runs (2026-07-04, newest complete
    # method set; the manuscripts' main/external table numbers trace here). The older
    # external_baselines_3seed.log (2026-06-30) views are kept below as __ext0630
    # SUPERSEDED provenance views (note: that older run had el2n==grand identical
    # selections; the 07-04 run separates them - GraNd proxy 0.3195 vs EL2N 0.2415).
    ("vision_full_3seed.log", "vision_cifar100",
     r"#+ SEED (\d) \(CIFAR-100", r"(?:test_acc|acc)=([0-9.]+)", True),
    ("timeseries_full_3seed.log", "time_etth1",
     r"#+ SEED (\d) \(ETTh1", r"MASE=([0-9.]+)", False),
    ("tep_full_3seed.log", "process_tep",
     r"#+ SEED (\d) \(TEP", r"(?:macroF1|F1)=([0-9.]+)", True),
    ("tabular_full_3seed.log", "tabular_electricity",
     r"#+ SEED (\d) \(electricity", r"auc=([0-9.]+)", True),
    ("external_baselines_3seed.log", "vision_cifar100__ext0630",
     r"#### VIS-cifar100 seed(\d) ####", r"(?:test_acc|acc)=([0-9.]+)", True),
    ("external_baselines_3seed.log", "time_etth1__ext0630",
     r"#### TS seed(\d) ####", r"MASE=([0-9.]+)", False),
    ("external_baselines_3seed.log", "process_tep__ext0630",
     r"#### TEP-mlp seed(\d) ####", r"(?:macroF1|F1)=([0-9.]+)", True),
    ("etth2_3seed.log", "time_etth2",
     r"#+ SEED (\d) \(ETTh2", r"MASE=([0-9.]+)", False),
    ("daisy_cstr_3seed.log", "time_daisy_cstr",
     r"#+ SEED (\d) \(DaISy CSTR", r"MASE=([0-9.]+)", False),
    ("daisy_steamgen_3seed.log", "time_daisy_steamgen",
     r"#+ SEED (\d) \(DaISy steamgen", r"MASE=([0-9.]+)", False),
    ("real_noise_cifar100n_3seed.log", "vision_cifar100n",
     r"#+ REALN SEED=(\d) #+", r"(?:test_acc|acc)=([0-9.]+)", True),
    ("semdedup_density_rerun_3seed.log", "vision_cifar100__semdedup_rerun",
     r"#+ SEED (\d) VISION #+", r"(?:test_acc|acc)=([0-9.]+)", True),
    ("semdedup_density_rerun_3seed.log", "process_tep__semdedup_rerun",
     r"#+ SEED (\d) TEP #+", r"(?:macroF1|F1)=([0-9.]+)", True),
    ("tabular_external_3seed.log", "tabular_electricity__ext0630",
     r"#### TAB-tabpfn seed(\d) ####", r"auc=([0-9.]+)", True),
] + [
    # chronos log holds five datasets; one view per dataset (dataset x base-model testbed)
    ("chronos_fm_3seed.log", "time_%s_chronos" % ds.lower(),
     r"##### CHRONOS ds=%s SEED=(\d) #####" % ds, r"MASE=([0-9.]+)", False)
    for ds in ("ETTh1", "ETTh2", "daisy_cstr", "daisy_steamgen")
] + [
    # ETTm1's chronos run lives in its own log (same anchor format)
    ("chronos_ettm1_3seed.log", "time_ettm1_chronos",
     r"##### CHRONOS ds=ETTm1 SEED=(\d) #####", r"MASE=([0-9.]+)", False),
    # ETTm1 x DLinear: previously-missing family, filled 2026-07-16 (3 seeds, RUN_ID
    # ettm1-dlinear, lane exit0)
    ("ettm1_dlinear_3seed.log", "time_ettm1",
     r"#+ ETTM1DL SEED=(\d)\b", r"MASE=([0-9.]+)", False),
]
# equal-budget selection methods only; excludes diagnostic reference rows (full/pool/...)
METHOD_WHITELIST = {
    "random", "coreset", "auth_only", "influence_only", "herding", "kcenter", "el2n",
    "grand", "ccs", "semdedup", "density", "quadmix", "dmf", "dsir", "zip", "entropy_law",
    "mmds_adapt",
}
METHOD_LINE = re.compile(r"^\s{2}(\w[\w()./-]*)\s+.*=")

PROTOCOL_TYPE = "unified-budget transfer (shared testbed)"
EXPECTED_SEEDS = 3
# EL2N/GraND/CCS are classification-error based scores; regression (MASE) views cannot
# host them by construction - recorded as explicit N/A, not as missing coverage.
TS_APPLICABILITY_NA = {
    "el2n": "classification-error based, not applicable to regression",
    "grand": "same",
    "ccs": "same",
}
EXIT_MARKER = re.compile(r"DONE|python_exit=|_OK")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def server_code_state():
    """git HEAD + sha256 of full `git diff` output (dirty-tree fingerprint)."""
    try:
        head = subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "-C", ROOT, "diff"])
        return {"head": head, "diff_sha256": hashlib.sha256(diff).hexdigest()}
    except Exception as e:  # not a git repo / git missing: record honestly
        return {"head": "NOT-CAPTURED (%s)" % e, "diff_sha256": "NOT-CAPTURED"}


def exit_evidence(log_path):
    """Terminal markers actually present in the log, or 'none-recorded'."""
    found = []
    for line in open(log_path, errors="replace"):
        if EXIT_MARKER.search(line):
            s = line.strip()
            if s not in found:
                found.append(s)
    return found if found else "none-recorded"


def parse(log_path, seed_re, metric_re):
    """Returns (values, parse_counts, conflicts, duplicate_sections).

    values: {method: {seed: value}}
    parse_counts: {method: {seed: n_occurrences_within_that_seed_section}}
    conflicts: same (method, seed) carrying DIFFERENT values (logs legitimately print
      each method twice per seed section - timing line + summary line with the SAME
      value; same value twice = OK, different values = CONFLICT).
    duplicate_sections: a seed anchor for the same seed appearing more than once.
    """
    values, counts, conflicts, dup_sections = {}, {}, [], []
    seen_seed_anchors = {}  # seed -> n anchors seen
    seed = None
    sre, mre = re.compile(seed_re), re.compile(metric_re)
    other_anchor = re.compile(r"^#")
    for lineno, line in enumerate(open(log_path, errors="replace"), 1):
        m = sre.search(line)
        if m:
            seed = int(m.group(1))
            seen_seed_anchors[seed] = seen_seed_anchors.get(seed, 0) + 1
            if seen_seed_anchors[seed] > 1:
                dup_sections.append({
                    "seed": str(seed), "line": lineno,
                    "detail": "seed anchor seen %d times" % seen_seed_anchors[seed]})
            continue
        if other_anchor.match(line) and not sre.search(line):
            # a different section header ends the current seed scope only if it looks like
            # another experiment anchor with 'SEED'/'seed' in it
            if re.search(r"[Ss][Ee][Ee][Dd]", line):
                seed = None
            continue
        if seed is None:
            continue
        ml = METHOD_LINE.match(line)
        if not ml:
            continue
        mv = mre.search(line)
        if not mv:
            continue
        method = ml.group(1)
        if method not in METHOD_WHITELIST:
            continue
        val, s = float(mv.group(1)), str(seed)
        prev = values.setdefault(method, {}).get(s)
        if prev is not None and prev != val:
            conflicts.append({
                "method": method, "seed": s, "line": lineno,
                "values": [prev, val],
                "detail": "same (method,seed) carries different values"})
        values[method][s] = val  # last occurrence wins (identical when no conflict)
        counts.setdefault(method, {})[s] = counts.get(method, {}).get(s, 0) + 1
    return values, counts, conflicts, dup_sections


def main():
    matrix = {"__server_code_state__": server_code_state()}
    trusted = True
    for src in SOURCES:
        log, ds, seed_re, metric_re, hib = src
        p = os.path.join(EXP, log)
        if not os.path.exists(p):
            print(f"[skip] {log} missing (dataset {ds})", file=sys.stderr)
            continue
        methods, counts, conflicts, dup_sections = parse(p, seed_re, metric_re)
        if not methods:
            print(f"[warn] {log}: no rows parsed for {ds}", file=sys.stderr)
            continue
        is_ts = ds.startswith("time_")  # regression (MASE) time-series views
        matrix[ds] = {
            "metric": metric_re, "higher_is_better": hib, "source_log": log,
            "log_sha256": sha256(p), "methods": methods,
            "protocol_type": PROTOCOL_TYPE,
            "applicability_na": dict(TS_APPLICABILITY_NA) if is_ts else {},
            "expected_seeds": EXPECTED_SEEDS,
            "parse_counts": counts,
            "conflicts": conflicts,
            "duplicate_sections": dup_sections,
            "config_hash": hashlib.sha256(repr(src).encode()).hexdigest(),
            "exit_evidence": exit_evidence(p),
        }
        n_full = sum(1 for m, sv in methods.items() if len(sv) >= EXPECTED_SEEDS)
        flag = ""
        if conflicts or dup_sections:
            trusted = False
            flag = "  <<< %d conflicts, %d duplicate sections" % (
                len(conflicts), len(dup_sections))
        print(f"[ok] {ds}: {len(methods)} methods, {n_full} with >={EXPECTED_SEEDS} seeds{flag}")
    out = os.path.join(EXP, "results_matrix.json")
    json.dump(matrix, open(out, "w"), indent=1, sort_keys=True)
    print(f"saved -> {out}")
    if not trusted:
        print("[FAIL] conflicts/duplicate_sections nonempty - matrix NOT trusted",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
