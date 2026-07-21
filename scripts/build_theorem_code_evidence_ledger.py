#!/usr/bin/env python3
"""Audit 2257-5: machine-readable theorem-to-code-to-evidence ledger (READ-ONLY).

Reads ONLY existing artifacts:
  - results_canonical/**/results.json  (adapt_manifest leaderboards, configs)
  - experiments/selection_manifest_verdicts.json (replay-parity gate evidence)
Writes ONLY:
  - experiments/theorem_code_evidence_ledger.json

No method changes, no reruns, no edits to tex/canonical/ledger/results files.

Theorem-4 numeric probe: for every canonical cell with a stored adapt_manifest,
replay the controller's regularized-switch decision from the stored leaderboard
(margin = switch_margin_frac * |best_ref gain|, switch_margin_frac default 0.015,
src/mmdataselect/fusion/adaptive.py lines 79/335-336), then replace the empirical
margin with the theory-style Hoeffding radius tau = 2*eps,
eps = R * sqrt(ln(2 m / delta) / (2 n_v)), delta = 0.05, m = |leaderboard|,
n_v = |V2 adjudication half| (runners split VAL_N in half: _v2 = perm[len//2:]).
R = 1 for metrics bounded in [0,1] (vision top-1 acc; tabular AUC; TEP macro-F1,
both with the paper's own scope caveats), and for timeseries neg-MASE (unbounded)
R = leaderboard value range as an explicitly-labelled crude proxy.
Cells without the needed leaderboard are recorded NOT_RECONSTRUCTIBLE.
"""
import glob
import json
import math
import os

REPO = "/Users/qianzekai/PycharmProjects/Paper2_OmniSelect"
DELTA = 0.05
SWITCH_MARGIN_FRAC = 0.015  # adaptive.py default; ADAPT_MARGIN env override not recorded in artifacts

# V2 adjudication sample size per arm = VAL_N // 2 (runners: _v2 = _vperm[len//2:]).
# VAL_N source: vision recorded in results.json config (val_n=800); tabular/tep/timeseries
# use the runner defaults (2500 / 2000 / 1000) because the canonical results.json configs
# do not record VAL_N -- flagged per cell as val_n_source=runner_default_not_recorded.
ARM_VAL = {
    "vision": {"val_n_total": 800, "val_n_source": "results.json config val_n=800 (recorded)",
               "metric": "linear-head top-1 accuracy on V2", "bounded": True,
               "bound_note": "per-sample 0/1 loss, mean in [0,1]; Hoeffding applies directly. "
                             "Leaderboard granularity 1/400 = 0.0025 confirms n_v=400."},
    "tabular": {"val_n_total": 2500, "val_n_source": "runner default VAL_N=2500 (scripts/run_tabular_experiment.py:42), not recorded in results.json config",
                "metric": "TabPFN ROC AUC on V2", "bounded": True,
                "bound_note": "AUC in [0,1] but a paired U-statistic, not a mean of iid per-sample terms; "
                              "paper scope remark (method.tex L171): concentration governed by min(pos,neg). "
                              "Hoeffding radius with n_v=1250 is a crude OPTIMISTIC proxy."},
    "tep": {"val_n_total": 2000, "val_n_source": "runner default VAL_N=2000 (scripts/run_tep_experiment.py:38), not recorded in results.json config",
            "metric": "MLP macro-F1 on V2", "bounded": True,
            "bound_note": "macro-F1 in [0,1] but a nonlinear function of class counts; paper scope remark "
                          "(method.tex L171): same-form bound with larger constants. Radius is a crude proxy."},
    "timeseries": {"val_n_total": 1000, "val_n_source": "runner default VAL_N=1000 (scripts/run_timeseries_experiment.py:34), not recorded in results.json config",
                   "metric": "DLinear negative MASE on V2 windows", "bounded": False,
                   "bound_note": "MASE is UNBOUNDED (NOT_DIRECTLY_BOUNDED): no natural [0,1] range. "
                                 "R is set to the observed leaderboard value range as an explicitly crude proxy; "
                                 "additionally V2 windows overlap, so effective n < 500 and the radius is optimistic "
                                 "on both counts (paper scope remark: truncation + disjoint-window effective n)."},
    "experiment": {"val_n_total": None, "val_n_source": "text arm: no controller run, no adapt_manifest",
                   "metric": "perplexity / lm-eval (text arm)", "bounded": False, "bound_note": "controller not run on this arm"},
}

NON_REF_PREFIXES = ("fuse ", "vote_ensemble", "grpo_policy")  # adaptive.py: fusions + self-improving = is_ref False


def is_ref(name: str) -> bool:
    return not any(name.startswith(p) for p in NON_REF_PREFIXES)


def probe_cell(path: str) -> dict:
    d = json.load(open(path))
    arm = d.get("arm")
    rel = os.path.relpath(path, REPO)
    parts = rel.split(os.sep)
    cell = {
        "cell": rel,
        "arm": arm,
        "dataset": d.get("dataset"),
        "run_id": next((p.split("run_id=")[1].split("-")[0:3] and p for p in parts if p.startswith("run_id=")), None),
        "seed": d.get("seed"),
    }
    am = d.get("adapt_manifest")
    meta = ARM_VAL.get(arm, {})
    if not am or not am.get("leaderboard"):
        cell["status"] = "NOT_RECONSTRUCTIBLE"
        cell["reason"] = ("no adapt_manifest stored (text arm does not run the adaptive controller; "
                         "methods are quadmix_pub/zip only)" if arm == "experiment"
                         else "adapt_manifest or leaderboard missing")
        return cell
    lb = [(str(n), float(g)) for n, g in am["leaderboard"]]
    chosen = am.get("chosen", {})
    m = len(lb)
    n_v = meta["val_n_total"] // 2 if meta.get("val_n_total") else None
    refs = [(n, g) for n, g in lb if is_ref(n)]
    overall_name, overall_gain = max(lb, key=lambda t: t[1])
    best_ref_name, best_ref_gain = max(refs, key=lambda t: t[1])
    gap = overall_gain - best_ref_gain
    # --- replay of the DEPLOYED empirical rule (adaptive.py L335-336) ---
    margin_emp = SWITCH_MARGIN_FRAC * abs(best_ref_gain)
    switch_emp = (not is_ref(overall_name)) and (overall_gain > best_ref_gain + margin_emp)
    emp_chosen = overall_name if (is_ref(overall_name) or switch_emp) else best_ref_name
    replay_ok = (emp_chosen == chosen.get("strategy")) and (switch_emp == bool(chosen.get("switched")))
    # --- theory-style rule (Theorem 4: tau = 2*eps, switch iff g_f >= g_r + tau) ---
    if meta.get("bounded"):
        R = 1.0
        r_src = "metric bounded in [0,1]"
    else:
        R = max(g for _, g in lb) - min(g for _, g in lb)
        r_src = "NOT_DIRECTLY_BOUNDED: leaderboard value range used as crude proxy (stated)"
    eps = R * math.sqrt(math.log(2.0 * m / DELTA) / (2.0 * n_v))
    tau = 2.0 * eps
    switch_th = (not is_ref(overall_name)) and (overall_gain >= best_ref_gain + tau)
    th_chosen = overall_name if (is_ref(overall_name) or switch_th) else best_ref_name
    cell.update({
        "status": "PROBED",
        "m_candidates": m,
        "val_n_total": meta["val_n_total"],
        "n_v_adjudication_V2": n_v,
        "val_n_source": meta["val_n_source"],
        "metric": meta["metric"],
        "metric_bounded": meta["bounded"],
        "bound_note": meta["bound_note"],
        "range_R": round(R, 6),
        "range_R_source": r_src,
        "best_ref": {"name": best_ref_name, "gain": best_ref_gain},
        "overall_best": {"name": overall_name, "gain": overall_gain, "is_reference": is_ref(overall_name)},
        "gap_overall_minus_best_ref": round(gap, 6),
        "stored_decision": {"strategy": chosen.get("strategy"), "switched": chosen.get("switched"),
                            "val_gain": chosen.get("val_gain"), "kappa_hat": chosen.get("kappa_hat")},
        "empirical_rule_replay": {"margin": round(margin_emp, 6), "switch": switch_emp,
                                  "chosen": emp_chosen, "matches_stored": replay_ok},
        "theory_rule": {"delta": DELTA, "hoeffding_eps": round(eps, 6), "tau_2eps": round(tau, 6),
                        "switch": switch_th, "chosen": th_chosen},
        "decision_changed": th_chosen != chosen.get("strategy"),
    })
    return cell


def main():
    paths = sorted(glob.glob(os.path.join(REPO, "results_canonical", "**", "results.json"), recursive=True))
    cells = [probe_cell(p) for p in paths]
    probed = [c for c in cells if c["status"] == "PROBED"]
    changed = [c for c in probed if c["decision_changed"]]
    not_rec = [c for c in cells if c["status"] == "NOT_RECONSTRUCTIBLE"]
    replay_bad = [c for c in probed if not c["empirical_rule_replay"]["matches_stored"]]

    probe_summary = {
        "cells_total": len(cells),
        "cells_probed": len(probed),
        "cells_not_reconstructible": len(not_rec),
        "not_reconstructible_cells": [c["cell"] for c in not_rec],
        "empirical_rule_replay_matches_stored": f"{len(probed) - len(replay_bad)}/{len(probed)}",
        "replay_mismatch_cells": [c["cell"] for c in replay_bad],
        "decisions_changed_under_theory_rule": len(changed),
        "changed_cells": [{"cell": c["cell"], "stored": c["stored_decision"]["strategy"],
                           "theory": c["theory_rule"]["chosen"],
                           "gap": c["gap_overall_minus_best_ref"], "tau_2eps": c["theory_rule"]["tau_2eps"]}
                          for c in changed],
        "modalities_without_natural_bound": {
            "timeseries": "neg-MASE unbounded; leaderboard-range proxy used (crude, stated); overlapping "
                          "windows further shrink effective n_v",
            "text(experiment)": "controller not run; no adapt_manifest; NOT_RECONSTRUCTIBLE",
        },
        "modalities_with_caveated_bounds": {
            "tabular": "AUC is a U-statistic, Hoeffding-on-mean is a proxy (paper scope remark)",
            "tep": "macro-F1 nonlinear in counts, same-form bound with larger constants (paper scope remark)",
        },
        "reading": ("The theory-calibrated tau=2*eps radius is one order of magnitude larger than the "
                    "deployed empirical margin (e.g. vision: tau~0.19 vs margin~0.007). It never creates a "
                    "new switch; its only effect is to CANCEL the switches the empirical rule made, i.e. "
                    "the controller would fall back to the best reference in every probed cell."),
        "recommendation": ("Keep method-v1's empirical fixed-fraction margin as the DEPLOYED rule: it is what "
                           "produced every canonical number and is scoped in the paper as an empirical threshold "
                           "explicitly not covered by Theorem 4 (method.tex L166/L171, fig_voting caption). A "
                           "theory-calibrated switch (tau=2*eps, or the paired-difference radius the paper says "
                           "is the relevant smaller quantity) would change controller decisions in the cells "
                           "listed above and therefore constitutes a NEW frozen method version requiring fresh "
                           "runs and re-freezing; it is NOT adopted here. This probe is read-only evidence."),
    }

    entries = build_entries()
    ledger = {
        "_meta": {
            "audit": "2257-5 theorem-to-code-to-evidence ledger",
            "generated_by": "scripts/build_theorem_code_evidence_ledger.py (read-only probe)",
            "read_only": True,
            "delta": DELTA,
            "switch_margin_frac_assumed": SWITCH_MARGIN_FRAC,
            "inputs": ["results_canonical/**/results.json (16 files)",
                       "experiments/selection_manifest_verdicts.json",
                       "papers/mmdataselect/submissions/aaai2027/sections/method.tex (statements, read)",
                       "papers/mmdataselect/submissions/aaai2027/sections/appendix.tex (proofs, read)",
                       "src/mmdataselect/fusion/adaptive.py",
                       "scripts/run_{vision,tabular,tep,timeseries}_experiment.py"],
            "honesty_notes": [
                "VAL_N for tabular/tep/timeseries is the runner default; the canonical results.json configs do not "
                "record it (only vision records val_n=800). If a canonical run overrode VAL_N via env, n_v here is wrong; "
                "no artifact contradicts the defaults (vision leaderboard granularity 1/400 confirms its half).",
                "switch_margin_frac assumed at its code default 0.015; the ADAPT_MARGIN env override is not recorded in "
                "artifacts. The empirical-rule replay matching the stored decision in all probed cells is the evidence "
                "the default was in force.",
                "reference/non-reference classification of leaderboard entries is reconstructed from candidate-name "
                "prefixes exactly as adaptive.py assigns is_ref (fuse*/vote_ensemble*/grpo_policy* are non-reference).",
                "Statuses cover only what the named artifacts exercise. Nothing here claims universal optimality, "
                "one-click reproduction, or clean-clone verification of SKIPped arms (TEP/text/vision per RC v7 scope).",
            ],
        },
        "entries": entries,
        "theorem4_probe": {"summary": probe_summary, "cells": cells},
    }
    out = os.path.join(REPO, "experiments", "theorem_code_evidence_ledger.json")
    with open(out, "w") as f:
        json.dump(ledger, f, indent=1, ensure_ascii=False)
    print("wrote", out)
    print(json.dumps(probe_summary, indent=1))


def build_entries():
    """Per-theorem ledger entries. Statements summarized from method.tex/appendix.tex; code refs are
    file:symbol; statuses are honest: VERIFIED only where a mechanical gate exercises the claim."""
    adaptive = "src/mmdataselect/fusion/adaptive.py"
    runners = ["scripts/run_vision_experiment.py", "scripts/run_tabular_experiment.py",
               "scripts/run_tep_experiment.py", "scripts/run_timeseries_experiment.py"]
    return [
        {
            "id": "Proposition 1",
            "statement_summary": "Complete coverage and orthogonal identifiability of the three channels "
                                 "(authenticity/influence/coverage) over the five modalities: every record is "
                                 "characterized by at least one channel, extreme policies delimit a reachable "
                                 "region, and channel effect directions are modality-invariant.",
            "assumptions": ["channels consume only content/embedding/structural flags of the unified record, not the modality value",
                            "per-record decisions independent (keep/resample/downweight/drop)"],
            "code_functions": ["src/mmdataselect/signals/* (three channel scorers)",
                               "src/mmdataselect/datatypes.py:UnifiedRecord",
                               adaptive + ":AdaptiveController (modality passed only via the gain callback)"],
            "config_fields": [],
            "experiment_gates": ["all four controller arms reuse one AdaptiveController with only the gain callback swapped "
                                 "(runners); channel-drop ablation experiments/channel_drop_ablation_3seed.log exercises "
                                 "per-channel contribution"],
            "status": "THEORY_ONLY",
            "notes": "Structural/design claim with a paper proof; orthogonal identifiability is not machine-checked by any "
                     "artifact. The code does instantiate the claimed decoupling (one controller, five callbacks).",
        },
        {
            "id": "Theorem 2",
            "statement_summary": "Budget-constrained selection (decision form) is NP-complete via reduction from "
                                 "budgeted set-function maximization; motivates approximate solvers.",
            "assumptions": ["general non-monotone non-submodular set function", "non-uniform token costs"],
            "code_functions": ["src/mmdataselect/selectors/budget_select.py:BudgetSelector (greedy / Gumbel-Top-K approximate solver)",
                               adaptive + ":AdaptiveController._fusion (budgeted Top-K / BudgetSelector calls)"],
            "config_fields": ["lam_grid (diversity strength)", "budget fraction per runner"],
            "experiment_gates": ["no experiment can exercise NP-completeness; the (1-1/e) submodular-case guarantee is "
                                 "cited, not measured"],
            "status": "THEORY_ONLY",
            "notes": "Self-contained complexity proof (method.tex L94). Code implements the approximation the theorem "
                     "motivates; nothing to verify empirically.",
        },
        {
            "id": "Theorem 3",
            "statement_summary": "Identification lower bound: any score-only policy (never queries downstream response) "
                                 "suffers worst-case excess risk >= kappa/2 on a constructed environment pair with "
                                 "pointwise-identical observables; validation-driven selection escapes conditionally "
                                 "(under Thm 4 premises, excess <= best candidate + 2*eps).",
            "assumptions": ["response strength kappa in (0,1]", "constructed environment pair, observables agree pointwise",
                            "escape clause needs the portfolio to contain the zero-excess policy (realizability) and kappa >= 4*eps"],
            "code_functions": [adaptive + ":AdaptiveController.select (queries held_out_gain = downstream response)",
                               "runners: extras include ('random', ...) and ('auth_bottom', reversed-authenticity Bottom-K)"],
            "config_fields": [],
            "experiment_gates": ["flip instances in canonical artifacts stand in for the construction (CIFAR-100 vs CIFAR-10 "
                                 "best-channel flip, TabPFN vs XGBoost direction flip; experiments/results_matrix.json and "
                                 "canonical_tables.json cells)"],
            "status": "THEORY_ONLY",
            "notes": "The bound itself is a construction, not runnable. PREMISE-GROUNDING GAP (flag for text fix, tex not "
                     "editable in this audit): method.tex L135 and appendix L63 say the portfolio contains a reversed "
                     "candidate PER CHANNEL; the code implements only the authenticity-reversed candidate auth_bottom "
                     "plus random (all four runners); no influence- or coverage-reversed candidates exist, and the fusion "
                     "grid has no negative weights. Realizability is therefore grounded for the authenticity direction only.",
        },
        {
            "id": "Proposition 2",
            "statement_summary": "Construction optimality: controller returns argmax of held-out gain over a portfolio "
                                 "containing every reference baseline, hence >= every baseline on validation; transfers "
                                 "to test up to generalization error; with no switch it returns the best reference's "
                                 "subset bit-for-bit.",
            "assumptions": ["portfolio contains every reference baseline (extras lists in runners)",
                            "i.i.d. validation/test with sufficient samples for the transfer clause"],
            "code_functions": [adaptive + ":AdaptiveController.select L327-346 (scored argmax, refs, best_ref, regularized fallback)",
                               "runners: extra_strategies lists passing each compared baseline into the portfolio"],
            "config_fields": ["switch_margin_frac / ADAPT_MARGIN (fallback margin)", "n_val_repeats"],
            "experiment_gates": ["adapt_manifest leaderboard+chosen stored in 13/16 canonical cells",
                                 "experiments/selection_manifest_verdicts.json: mmds_adapt REPLAY_VERIFIED on vision/tep/tabular "
                                 "seeds 0-2, timeseries seeds 0-1 REPLAY_VERIFIED_SELECTION_ONLY, timeseries seed 2 "
                                 "HASH_ONLY_NOT_REPLAYED (duplicates under no-replacement protocol)",
                                 "this ledger's decision replay: empirical rule reproduces the stored chosen strategy and "
                                 "switched flag in every probed cell (see theorem4_probe.summary)"],
            "status": "VERIFIED",
            "notes": "argmax+regularized-fallback is directly and exactly implemented; validation-side (i) is mechanical. "
                     "The test-side transfer (ii) is a statistical claim, checked only in the multi-seed 'never worst' "
                     "sense reported in the experiments section, not per-cell. Timeseries seed 2 replay gap is inherited "
                     "from the manifest gate and disclosed there.",
        },
        {
            "id": "Theorem 4",
            "statement_summary": "Finite-sample guarantee and calibrated switch: with eps = sqrt(ln(2m/delta)/(2 n_v)), "
                                 "uniform convergence, selection guarantee G(chosen) >= max G - 2*eps, and a calibrated "
                                 "switch tau = 2*eps that is safe (never switches to truly worse) and sensitive "
                                 "(true advantage >= 4*eps always switches).",
            "assumptions": ["per-sample metric bounded in [0,1], mean form", "validation i.i.d. with test",
                            "portfolio constructed independently of the adjudication validation half",
                            "statements conditional on the candidate pool"],
            "code_functions": [adaptive + ":AdaptiveController.__init__ L79 (switch_margin_frac, ADAPT_MARGIN env)",
                               adaptive + ":AdaptiveController.select L333-336 (margin = switch_margin_frac * |best_ref gain|; "
                               "switch iff overall > best_ref + margin)",
                               adaptive + ":select L166-199 (V1/V2 protocol: construction on V1, adjudication on V2, candidate "
                               "list frozen before V2 -- implements the independence premise)",
                               "runners: gain/gain1 (V2/V1 callbacks), _vperm half-split"],
            "config_fields": ["ADAPT_MARGIN (default 0.015)", "VAL_N per runner (800/2500/2000/1000 defaults)", "delta not a code parameter"],
            "experiment_gates": ["theorem4_probe in this ledger (13 probed cells, per-cell Hoeffding radius vs empirical margin)",
                                 "V1/V2 protocol is code-enforced (self-improving candidates disabled without a V1 callback)"],
            "status": "MISMATCH-DISCLOSED",
            "notes": "The code does NOT implement tau = 2*eps: the deployed margin is a fixed fraction (0.015) of the best "
                     "reference gain, e.g. vision ~0.007 vs theory tau ~0.19 at n_v=400, m=26. The papers already scope "
                     "this: method.tex L166 ('the current fixed-fraction margin remains an empirical threshold and is not "
                     "covered directly by the theorem'), L171 third scope remark, and the fig_voting caption. The "
                     "independence premise (portfolio frozen before V2) IS implemented. Probe finding: the theory radius "
                     "never adds a switch and cancels the empirical switches (see theorem4_probe.summary.changed_cells).",
        },
        {
            "id": "Theorem 5",
            "statement_summary": "Selection gain decomposition under Assumption A (first-order response model "
                                 "R(S)=R0+kappa*rho(S)+eta*d(S)): gain over random = kappa*(rho_bar-rho_sigma) - "
                                 "eta*(d_sigma-d_bar); corollaries scaling / flip threshold / oracle ceiling and "
                                 "conversion / comparative statics.",
            "assumptions": ["Assumption A: linear first-order response around zero-harm full-coverage (testable modeling assumption)",
                            "kappa,eta properties of learner+task, not data",
                            "comparative-statics corollary additionally needs monotone detector effectiveness"],
            "code_functions": [adaptive + ":select L337-345 (kappa_hat = best-candidate gain minus random gain, the deployable "
                               "empirical signature; stored in every adapt_manifest chosen)"],
            "config_fields": [],
            "experiment_gates": ["kappa_hat stored per cell: tabular 0.005-0.011 (robust-FM near-zero regime) vs "
                                 "tep 0.087-0.116, vision 0.058-0.090, timeseries 0.061-0.118 -- direction matches the "
                                 "scaling corollary",
                                 "noise-ratio ablation (experiments/noise_ratio_ablation_3seed.log) and oracle rows in "
                                 "canonical results exercise the ceiling corollary qualitatively"],
            "status": "THEORY_ONLY",
            "notes": "Assumption A is never fitted or falsified mechanically in the artifacts; only the kappa_hat signature "
                     "is implemented and recorded. The decomposition itself (rho, d accounting) has no code counterpart.",
        },
        {
            "id": "Theorem 6",
            "statement_summary": "Dominance of measured allocation over any fixed weight under the channel-linear model: "
                                 "expected-utility gap = N_B * (1-cos theta_bar) * sqrt(beta* Sigma beta*) * M(B); zero iff "
                                 "sensitivities homogeneous at the budget margin (the gap equals the flip).",
            "assumptions": ["channel-linear data model delta_f(x) = <beta*(f,x), phi(x)> + noise, E[noise|phi]=0",
                            "Sigma_phi positive definite", "near-Gaussian budget margins"],
            "code_functions": [adaptive + ":select L252-282 (measured-weight fusion: coordinate ascent from V1-best grid "
                               "point, probes on V1 only, 'fuse learned ...' candidate added to portfolio)"],
            "config_fields": ["policy_search / ADAPT_GRPO (optional continuous search variant, default off)"],
            "experiment_gates": ["'fuse learned w=...' candidates appear in stored leaderboards (e.g. vision seed 0) and are "
                                 "adjudicated on V2 like every candidate"],
            "status": "THEORY_ONLY",
            "notes": "The MECHANISM (measured allocation) is implemented and exercised; the closed-form gap identity is "
                     "never computed on data (no beta*, Sigma_phi, or cos theta_bar estimation exists in the code). "
                     "Honest split: mechanism implemented, quantitative identity theory-only.",
        },
        {
            "id": "Proposition 3",
            "statement_summary": "Superposition beats picking one: least-squares projection of delta onto the three "
                                 "channels correlates with true delta at least as well as the best single channel, with "
                                 "strict improvement equal to the residual partial correlation; fused-proxy ranking has "
                                 "regret <= best single channel.",
            "assumptions": ["least-squares projection well-defined", "reference set grows for the convergence clause"],
            "code_functions": [adaptive + ":_default_grid/_W3 L38-46 (discrete weight simplex)",
                               adaptive + ":AdaptiveController._fusion (weighted fusion candidates)"],
            "config_fields": ["weight_grid (runners pass reduced grids on tabular/tep)"],
            "experiment_gates": ["fusion candidates outrank single channels on some stored leaderboards (e.g. timeseries "
                                 "pubcore seed 0: chosen 'fuse w=(0.5, 0, 0.5) q=0.0 lam=0.0' beat all references)"],
            "status": "THEORY_ONLY",
            "notes": "Paper itself frames the grid+validation as a DISCRETE APPROXIMATION of the projection (method.tex "
                     "L204); no exact least-squares fit exists in code, so the proposition is not directly implemented.",
        },
        {
            "id": "Proposition 5",
            "statement_summary": "Vote-ensemble strict improvement under sufficient conditions: near-disjoint harmful "
                                 "mis-selections (complementarity), common two-candidate core >= budget, balanced weights "
                                 "=> harmful rate of the vote set is zero and (with Assumption A) downstream strictly "
                                 "exceeds the best single member; degrades non-gracefully when conditions break.",
            "assumptions": ["mechanism complementarity (near-disjoint harmful sets)", "core size >= budget",
                            "balanced vote weights", "Assumption A for the downstream clause"],
            "code_functions": [adaptive + ":select L209-217 (vote_ensemble(top3): V1-top-3, gain-margin-over-random weights, top-k by votes)",
                               adaptive + ":select L225-229 (METHOD_V2 diversity-regularized variant, env-gated default off)",
                               adaptive + ":select L235-249 (METHOD_V2C complementarity-gated variant, env-gated default off)"],
            "config_fields": ["METHOD_V2", "METHOD_V2_LAM", "METHOD_V2C", "METHOD_V2C_THR (default 0.35)"],
            "experiment_gates": ["tep seed 1: vote_ensemble(top3) adopted by the switch (stored chosen, switched=true)",
                                 "vision seed 0: vote_ensemble tops the leaderboard (0.48) but the margin rule keeps dmf_pub "
                                 "-- the adjudicated-not-unconditional design the proposition motivates",
                                 "measured complementarity gamma_hat=8.9% on the vision arm (method.tex L150)"],
            "status": "THEORY_ONLY",
            "notes": "Mechanism implemented and exercised (one canonical adoption); the sufficient CONDITIONS are not "
                     "machine-checked in the canonical path (the complementarity check exists only behind METHOD_V2C, "
                     "default off, outside the frozen method).",
        },
    ]


if __name__ == "__main__":
    main()
