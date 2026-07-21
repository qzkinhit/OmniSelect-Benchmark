"""SUPERSEDED - legacy 4-dataset, 3-seed-mean protocol, kept for audit-trail
provenance only. Its output lists auth_only/dmf/kcenter/semdedup as ordinary comparison
baselines, which the current protocol excludes as the controller's own portfolio
candidates. Do NOT cite this script's numbers in the manuscript. The current
headline source is scripts/build_canonical_seed0.py -> experiments/canonical_tables_seed0.json
(9 datasets, seed 0; see docs/REPRODUCIBILITY.md section 0/6).

--- original docstring below, describes the legacy pipeline ---

Single machine-readable canonical source for BOTH paper manuscripts
(CODEX_AUDIT_FINAL_GATE_2 item 7 / FINAL_GATE item 9).

Reads experiments/results_matrix.json (verification-grade ledger, per-seed values from
real logs) and emits experiments/canonical_tables.json holding mean+-std for every
(view, method) cell, plus ready-to-paste LaTeX rows for the main and external tables.
Both manuscripts' tables MUST be regenerated from this file - hand-maintained numbers
are forbidden. Conflicts between this source and the current .tex values are the
backfill worklist (includes the known 0.324/0.319, 0.428/0.434, 0.871/0.854 conflicts
and the 4 alerts from docs/baseline_fidelity_ledger.md).

TEP detection columns (calibrated FDR@FAR/balacc/AUPRC) are appended from the
tep-calib lane outputs when present (outputs/tep/*/tep-calib*/seed_*/results.json).
"""
import glob
import json
import os
import re
import statistics as st

ROOT = os.environ.get("OMNISELECT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = os.path.join(ROOT, "experiments", "results_matrix.json")
V5 = os.path.join(ROOT, "experiments", "controller_current_canonical_v5.json")
OUT = os.path.join(ROOT, "experiments", "canonical_tables.json")

# paper table layout: column order per manuscript table (view keys from the matrix)
MAIN_COLS = ["vision_cifar100", "time_etth1", "process_tep", "tabular_electricity"]
# method -> display name (paper row labels); order = table row order
MAIN_ROWS = [
    ("random", "Random"), ("coreset", "Coreset (k-center greedy)"),
    ("auth_only", "Authenticity-only"), ("influence_only", "Influence-only"),
    ("dmf", "Dynamic fusion (DMF)"), ("mmds_adapt", "Controller (ours)"),
]
EXT_ROWS = [
    ("herding", "Herding"), ("kcenter", "k-center"), ("el2n", "EL2N"),
    ("grand", "GraNd (proxy)"), ("ccs", "CCS"), ("semdedup", "SemDeDup"),
    # quadmix style proxy WITHDRAWN (PROTOCOL_INVALID_DUPLICATE_IDS, see
    # docs/quadmix_styleproxy_invalidation.md); quadmix_pub is the sole representative
    ("density", "Density"), ("quadmix_pub", "QuaDMix (published-core transfer)"),
]


def mstd(vals):
    if not vals:
        return None
    m = sum(vals) / len(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": round(m, 4), "std": round(s, 4), "n_seeds": len(vals),
            "fmt": "%.3f$\\pm$%.3f" % (m, s)}


def tep_calib_cells():
    """Aggregate calibrated TEP detection metrics per method across tep-calib seeds."""
    rows = {}
    # tep-calib2 = the dedicated large-normal-sample calibration split (final); the
    # earlier tep-calib run (small V1-half calibration) is superseded provenance
    for p in sorted(glob.glob(os.path.join(ROOT, "outputs", "tep", "*", "*tep-calib2*", "seed_*", "results.json"))):
        seed = os.path.basename(os.path.dirname(p)).split("_")[-1]
        loaded = json.load(open(p))
        if isinstance(loaded, dict):  # _trial_dump wrapper
            loaded = loaded["results"]
        for r in loaded:
            c = r.get("calib") or {}
            if "fdr@far5" not in c:
                continue
            rows.setdefault(r["method"], {}).setdefault(seed, c)
    out = {}
    for m, per_seed in rows.items():
        agg = {}
        for k in ("fdr@far5", "far@far5", "balacc@far5", "auroc", "auprc"):
            vals = [c[k] for c in per_seed.values() if k in c]
            if vals:
                agg[k] = mstd(vals)
        agg["n_seeds"] = len(per_seed)
        out[m] = agg
    return out


PUBCORE_GLOB = os.path.join(ROOT, "outputs", "*", "*", "run_id=pubcore-paired-*", "seed_*", "results.json")
# Tab-AICL transparent standalone rows: same frozen protocol + code state as the pubcore
# batch (verified: runner/baseline SHAs unchanged; seed1 hybrid parity MATCH vs the
# pubcore controller manifest). Tabular view only.
TABAICL_GLOB = os.path.join(ROOT, "outputs", "tabular", "*", "run_id=tabaicl-standalone-*", "seed_*", "results.json")
ARM2VIEW = {"vision": "vision_cifar100", "timeseries": "time_etth1",
            "tep": "process_tep", "tabular": "tabular_electricity"}
FIDELITY_LABEL = {"quadmix": "style proxy - PROTOCOL_INVALID_DUPLICATE_IDS, withdrawn",
                  "dmf": "dynamic-fusion proxy",
                  "quadmix_pub": "published-core transfer", "dmf_pub": "published-update transfer",
                  "grand": "last-layer proxy", "ccs": "local implementation (EL2N-binned)",
                  "tabpfn_coreset": "Tab-AICL transfer (standalone)",
                  "tabpfn_margin": "Tab-AICL transfer (standalone)",
                  "tabpfn_hybrid": "Tab-AICL transfer (standalone, parity-matched to controller pick)"}

# ts-seed2 controller replacement: the pubcore timeseries seed2 mmds_adapt cell had
# chosen=quadmix (style proxy, PROTOCOL_INVALID_DUPLICATE_IDS); it is replaced by the
# targeted no-proxy controller re-run (run_id ctrl-ts2-noproxy-*).
TS2_NOPROXY_GLOB = os.path.join(ROOT, "outputs", "timeseries", "*",
                                "run_id=ctrl-ts2-noproxy-*", "seed_2", "results.json")
TS2_NOTE = ("chosen=quadmix cell PROTOCOL_INVALID, replaced by ctrl-ts2-noproxy re-run "
            "(1.0284), docs/quadmix_styleproxy_invalidation.md")


def ts_seed2_replacement():
    """Return (mase, run_id) of the mmds_adapt row in the NEWEST ctrl-ts2-noproxy
    seed_2 results.json, or (None, None) if absent."""
    paths = sorted(glob.glob(TS2_NOPROXY_GLOB), key=os.path.getmtime)
    if not paths:
        return None, None
    d = json.load(open(paths[-1]))
    rid = (d.get("config") or {}).get("run_id")
    if not rid:
        m = re.search(r"run_id=(ctrl-ts2-noproxy-[0-9T]+)", paths[-1])
        rid = m.group(1) if m else None
    for r in d["results"]:
        if r["method"] == "mmds_adapt" and r.get("mase") is not None:
            return float(r["mase"]), rid
    return None, None


def pubcore_cells():
    """FINAL main-table source: the unified same-code-state paired batch
    (PUBLISHED_CORE_PAIRED_MAIN_OK; independent parity re-check 9/9 MATCH). Every cell
    from one RUN_ID, one code state, per-seed shared fit_seed, sel_sha12 recorded."""
    per = {}  # view -> method -> {seed: value}
    run_id = None
    for p in sorted(glob.glob(PUBCORE_GLOB)):
        d = json.load(open(p))
        view = ARM2VIEW.get(d["arm"])
        if view is None:
            continue
        run_id = (d.get("config") or {}).get("run_id") or run_id
        metric = {"vision_cifar100": "acc", "time_etth1": "mase",
                  "process_tep": "f1", "tabular_electricity": "auc"}[view]
        for r in d["results"]:
            v = r.get(metric)
            if v is not None:
                per.setdefault(view, {}).setdefault(r["method"], {})[str(d["seed"])] = float(v)
    # ts-seed2 controller replacement (see TS2_NOTE): override the timeseries
    # mmds_adapt seed '2' value with the no-proxy re-run before aggregating.
    ts2_val, ts2_rid = ts_seed2_replacement()
    ts2_applied = (ts2_val is not None
                   and "time_etth1" in per and "mmds_adapt" in per["time_etth1"])
    if ts2_applied:
        per["time_etth1"]["mmds_adapt"]["2"] = ts2_val
    cells = {}
    for view, methods in per.items():
        cells[view] = {}
        for m, sv in methods.items():
            c = mstd(list(sv.values()))
            c["source"] = "pubcore-paired (same-code-state)"
            if m in FIDELITY_LABEL:
                c["fidelity_label"] = FIDELITY_LABEL[m]
            if ts2_applied and view == "time_etth1" and m == "mmds_adapt":
                c["ts_seed2_replacement"] = TS2_NOTE
                c["ts_seed2_replacement_run_id"] = ts2_rid
            cells[view][m] = c
    return cells


def tabaicl_transparency():
    """SUPPLEMENTARY transparency table - NOT main-table rows (ZIP_REUSE_AND_TABAICL_SCOPE
    items 5-6). Tab-AICL is withdrawn from numeric head-to-head per
    docs/frozen_baseline_fidelity_gates.md (iterative in-context support selection is a
    different problem from one-shot budgeted selection; related-work only). These rows
    exist solely to prove that the controller's portfolio candidates are transparent:
    the pubcore seed1 controller pick (tabpfn_hybrid) matches its standalone row
    bit-for-bit (sel_sha12 c05525f026e1, auc 0.8568). Wording: 'Tab-AICL transfer',
    never original-protocol reproduction."""
    rows = {}
    for p in sorted(glob.glob(TABAICL_GLOB)):
        d = json.load(open(p))
        for r in d["results"]:
            v = r.get("auc")
            if v is not None:
                rows.setdefault(r["method"], {})[str(d["seed"])] = float(v)
    out = {}
    for m, sv in rows.items():
        c = mstd(list(sv.values()))
        c["fidelity_label"] = FIDELITY_LABEL.get(m, "Tab-AICL transfer")
        out[m] = c
    if out:
        out["__parity_evidence__"] = ("pubcore seed1 controller chosen=tabpfn_hybrid "
                                      "sel_sha12=c05525f026e1 auc=0.8568 == standalone row (MATCH)")
    return out


def v5_controller_cells():
    """Controller (mmds_adapt) rows take PRIORITY from the adopted re-validation
    canonical (controller_current_canonical_v5, scope=current-formal-implementation,
    fingerprint-keyed replays of 2026-07-15) over any older log-derived value. This
    resolves the 0.424 / 0.428 / 0.434 vision-controller conflict to the v5 numbers."""
    if not os.path.exists(V5):
        return {}
    d = json.load(open(V5))
    out = {}
    pat = re.compile(r"run_id=ctrlv5-([a-z]+)-([A-Za-z0-9_]+?)-s(\d)-")
    for r in d.get("rows", []):
        ad = r.get("artifact_detail", {})
        if (ad.get("artifact_config") or {}).get("drop"):
            continue
        v = ad.get("new_test")
        m = pat.search(r.get("artifact", ""))
        if v is None or not m:
            continue
        kind, ds, seed = m.group(1), m.group(2), m.group(3)
        view = {("vision", "base"): "vision_cifar100", ("tep", "base"): "process_tep",
                ("tabular", "base"): "tabular_electricity"}.get((kind, ds))
        if view is None and kind == "ts":
            view = "time_%s" % ds.lower()
        elif view is None and kind == "chronos":
            view = "time_%s_chronos" % ds.lower()
        if view:
            out.setdefault(view, {})[seed] = float(v)
    return out


def main():
    mx = json.load(open(MATRIX))
    cells = {}
    for view, e in mx.items():
        if view.startswith("__"):
            continue
        cells[view] = {m: mstd(list(sv.values())) for m, sv in e["methods"].items()}
    v5_overridden = []
    for view, sv in v5_controller_cells().items():
        if view in cells and len(sv) >= 3:
            cell = mstd(list(sv.values()))
            cell["source"] = "controller_current_canonical_v5"
            cells[view]["mmds_adapt"] = cell
            v5_overridden.append(view)

    def latex_rows(rows, cols, src):
        lines = []
        for key, label in rows:
            vals = []
            for c in cols:
                cell = src.get(c, {}).get(key)
                vals.append(cell["fmt"] if cell else "--")
            lines.append("%s & %s \\\\" % (label, " & ".join(vals)))
        return lines

    final_cells = pubcore_cells()
    FINAL_MAIN = MAIN_ROWS + [("quadmix_pub", "QuaDMix (published-core transfer)"),
                              ("dmf_pub", "DMF (published-update transfer)")]
    out = {
        "FINAL_main_table_source": "pubcore-paired-20260716T1754 (single code state, PUBLISHED_CORE_PAIRED_MAIN_OK, independent parity re-check 9/9 MATCH on controller-picked named baselines, per-seed shared fit_seed, sel_sha12 per row; baseline_impl dea8ee64)",
        "final_cells": final_cells,
        "latex_FINAL_main_table": latex_rows(FINAL_MAIN, MAIN_COLS, final_cells),
        "latex_FINAL_external_table": latex_rows(EXT_ROWS, MAIN_COLS, final_cells),
        "fidelity_labels": FIDELITY_LABEL,
        "tabaicl_transparency_supplement": tabaicl_transparency(),
        "legacy_source": "experiments/results_matrix.json (per-seed, log-SHA verified) - PROVENANCE ONLY",
        "legacy_controller_priority": "mmds_adapt legacy cells overridden from controller_current_canonical_v5 on: %s" % ", ".join(sorted(v5_overridden)),
        "rule": "both manuscripts regenerate tables from latex_FINAL_* only; no hand-maintained numbers; legacy cells are provenance, never for the main table",
        "legacy_code_state_caveat": "legacy baseline rows parsed from 2026-07-04 full_3seed logs; legacy controller rows from 2026-07-15 v5 replays. MIXED CODE STATES - superseded by the FINAL pubcore-paired source above; 'sanctioned adoption' wording retracted.",
        "cells": cells,
        "latex_legacy_main_table": latex_rows(MAIN_ROWS, MAIN_COLS, cells),
        "latex_legacy_external_table": latex_rows(EXT_ROWS, MAIN_COLS, cells),
        "tep_calibrated_detection": tep_calib_cells(),
        "all_views_available": sorted(k for k in cells),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("legacy views: %d, FINAL views: %d, tep-calib methods: %d" %
          (len(cells), len(final_cells), len(out["tep_calibrated_detection"])))
    print("--- FINAL main table (pubcore-paired, same code state) ---")
    for l in out["latex_FINAL_main_table"]:
        print(" ", l)
    print("--- FINAL external table ---")
    for l in out["latex_FINAL_external_table"]:
        print(" ", l)
    print("saved -> %s" % OUT)


if __name__ == "__main__":
    main()
