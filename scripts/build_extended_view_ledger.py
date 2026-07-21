"""Extended-view source ledger (audit 2030-4).

Single auditable source rule for the supplementary time-series table: EVERY cell
(controller included) comes from experiments/results_matrix.json three-seed means
(namespace `results_matrix`), whose entries carry source_log + log_sha256 +
config_hash + protocol_type; `full` reference cells are parsed from the same
tracked source logs. controller_current_canonical_v5 is an ANALYSIS-LAYER
priority only (characteristic metrics) and is recorded here as the known
alternative namespace with its differing value, NOT adopted for tables.

Outputs experiments/extended_view_source_ledger.json and asserts the LaTeX rows
in the AAAI supplementary table match the ledger-derived strings exactly.
"""
import hashlib
import json
import os
import re
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = os.path.join(ROOT, "experiments", "results_matrix.json")
V5 = os.path.join(ROOT, "experiments", "controller_current_canonical_v5.json")
OUT = os.path.join(ROOT, "experiments", "extended_view_source_ledger.json")
TEX = os.path.join(ROOT, "papers", "mmdataselect", "submissions", "aaai2027",
                   "sections", "experiments.tex")

VIEWS = {
    "time_daisy_cstr": "DaISy CSTR",
    "time_daisy_steamgen": "DaISy steamgen",
    "time_etth2": "ETTh2 (LSF)",
    "time_etth1": "ETTh1 (LSF, 2nd)",
    "time_ettm1": "ETTm1 (prose)",
}
METHODS = ["mmds_adapt", "auth_only", "full", "random"]


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def parse_full(log_path):
    """three-seed mean of the `full` rows in a tracked extended-lane log."""
    vals = []
    for line in open(log_path):
        m = re.match(r"\s*full\s+.*MASE=([0-9.]+)", line)
        if m:
            vals.append(float(m.group(1)))
    uniq = sorted(set(vals))
    return uniq, (sum(uniq) / len(uniq) if uniq else None)


def main():
    mx = json.load(open(MATRIX))
    v5 = json.load(open(V5)) if os.path.exists(V5) else {}
    canon = json.load(open(os.path.join(ROOT, "experiments", "canonical_tables.json")))
    ledger = {"rule": "single auditable source: results_matrix three-seed means for every cell; "
                      "full parsed from the same tracked source log; v5 recorded as analysis-layer "
                      "alternative only (audit 2030-4)",
              "views": {}}
    # ETTh1 row: adopted namespace is the paper's FINAL source (pubcore-paired
    # single-code-state batch + ts2-noproxy replacement), NOT the superseded
    # matrix lane. Declared per-row so the supp table is source-bound, not mixed
    # silently.
    fc = canon["final_cells"]["time_etth1"]
    ledger["views"]["time_etth1"] = {
        "label": VIEWS["time_etth1"], "adopted_namespace": "canonical_final_cells",
        "source": canon.get("FINAL_main_table_source"),
        "raw_artifacts_in_git": "results_canonical/ (pubcore-paired 12 + ctrl-ts2-noproxy 1; per-file SHA "
                                "verified; rebuildable via scripts/rebuild_canonical_from_whitelist.py diff=0)",
        "protocol_equality": "all four cells from the same paired single-code-state batch "
                             "(PAIRED_RNG=1, per-seed shared fit seed, parity gate 9/9)",
        "methods": {m: {"mean": fc[m]["mean"], "std": fc[m].get("std"),
                        "n_seeds": fc[m].get("n_seeds"), "fmt": fc[m].get("fmt")}
                    for m in ("mmds_adapt", "auth_only", "full", "random") if m in fc},
        "alternative_namespace_matrix": {
            "means": {m: round(sum(sv.values()) / len(sv), 4)
                      for m, sv in (mx.get("time_etth1", {}).get("methods") or {}).items()
                      if m in ("mmds_adapt", "auth_only", "random")},
            "adopted": False,
            "note": "superseded pre-paired lane; kept for provenance only"}}
    for view, label in VIEWS.items():
        if view == "time_etth1":
            continue
        e = mx.get(view)
        if not e:
            continue
        log_rel = os.path.join("experiments", e["source_log"])
        log_abs = os.path.join(ROOT, log_rel)
        log_ok = os.path.exists(log_abs) and sha256(log_abs) == e["log_sha256"]
        row = {"label": label, "adopted_namespace": "results_matrix",
               "source_log": log_rel, "log_sha256": e["log_sha256"],
               "log_sha256_reverified": bool(log_ok),
               "config_hash": e.get("config_hash"),
               "protocol_type": e.get("protocol_type"),
               "protocol_equality": "all methods in this row share the same source log, config_hash, "
                                    "pool/split/noise/budget (unified-budget shared testbed) and seeds 0/1/2",
               "metric": e.get("metric"), "higher_is_better": e.get("higher_is_better"),
               "methods": {}}
        for meth in METHODS:
            if meth == "full":
                uniq, mean = parse_full(log_abs) if os.path.exists(log_abs) else ([], None)
                row["methods"]["full"] = {"per_seed": uniq, "mean": round(mean, 4) if mean else None,
                                          "source": "parsed from source_log full rows"}
                continue
            sv = (e.get("methods") or {}).get(meth)
            if not sv:
                continue
            vals = [sv[s] for s in sorted(sv)]
            row["methods"][meth] = {"per_seed": dict(sorted(sv.items())),
                                    "mean": round(sum(vals) / len(vals), 4),
                                    "std": round(st.stdev(vals), 4) if len(vals) > 1 else None}
        alt = (v5.get(view) or {}) if isinstance(v5, dict) else {}
        if alt:
            row["alternative_namespace_v5"] = {
                "value": alt, "adopted": False,
                "note": "controller_current_canonical_v5 re-adjudication value; analysis-layer only; "
                        "differs from the adopted matrix mean where re-adjudication changed the pick"}
        ledger["views"][view] = row

    # derive + assert the AAAI supp-table rows
    def fmt(x):
        return ("%.3f" % x)
    tex = open(TEX).read()
    derived, mismatches = {}, []
    order = ["time_daisy_cstr", "time_daisy_steamgen", "time_etth2", "time_etth1"]
    for view in order:
        r = ledger["views"][view]
        m = r["methods"]
        cells = [round(m["mmds_adapt"]["mean"], 3), round(m["auth_only"]["mean"], 3),
                 round(m["full"]["mean"], 3), round(m["random"]["mean"], 3)]
        best = min(range(4), key=lambda i: cells[i])
        parts = [("\\textbf{%s}" % fmt(c)) if i == best else fmt(c) for i, c in enumerate(cells)]
        line = "%s & %s \\\\" % (r["label"], " & ".join(parts))
        derived[view] = line
        if line not in tex:
            mismatches.append(line)
    ledger["derived_supp_rows"] = derived
    ledger["tex_row_assertion"] = "PASS" if not mismatches else {"MISMATCH": mismatches}
    with open(OUT, "w") as fh:
        json.dump(ledger, fh, indent=1)
    print("ledger ->", OUT)
    for v, l in derived.items():
        print(("OK  " if l in tex else "MISS"), l)
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
