"""DEPRECATED DRAFT - DO NOT USE FOR PAPER NUMBERS (CODEX_AUDIT_20260716_1230 item 4).

Retracted defects of this v1 draft:
  1. Inputs are hand-copied rounded paper means, not per-seed canonical JSONs.
  2. Raw metric deltas (acc/F1/AUC/MASE) are dimensionally incompatible; negating MASE
     is not normalization, so cross-modal averaging here is not meaningful.
  3. OSRR uses a post-hoc TEST-set oracle with an arbitrary 0.003 tolerance (test leakage).
  4. "WCR" is min gain-over-random, not a regret; the "60x" positive/negative ratio claim
     is mathematically meaningless and is RETRACTED.
The redesigned version (characteristic_metrics_v2) must: read per-seed raw JSONs, use a
dimensionless utility (per-dataset rank percentile / standardized effect vs random),
define regret = oracle_utility - method_utility (lower better), use a VALIDATION oracle or
paired-bootstrap statistically-top-tier recovery, report bootstrap 95% CI + seed variance +
normalization sensitivity, and be disclosed as secondary/exploratory (post-hoc, not
preregistered). Standard task metrics stay primary.

Original draft below, kept for the record. Running it now hard-fails.
"""
raise SystemExit("DEPRECATED: retracted by CODEX_AUDIT_20260716_1230; use characteristic_metrics_v2 when built")

_DOC_V1 = """Characteristic metrics for OmniSelect (analogous to UniClean's REDR / DemandClean's
model-tolerance): principled, method-property metrics the voting-based controller is
designed to be high on. NOT gerrymandered - each measures the paper's actual thesis and
is computable for every method from the same per-(modality,strategy) test table.

Metrics
  AG  Adaptivity Gain: a method's mean test gain over random, MINUS the best single fixed
      strategy's mean gain when that fixed strategy is applied UNIFORMLY across modalities.
      A fixed method must commit to one signal; the controller adapts per modality. AG>0
      exactly where the best signal flips across modalities.
  OSRR Oracle-Signal Recovery Rate: fraction of modalities where the controller's picked
      strategy matches the post-hoc oracle-best fixed strategy (the one with the highest
      test on that modality). Measures 'measurement necessity' fulfilment.
  WCR Worst-case Cross-modal Regret: for each method, its worst single-modality gain over
      random across all modalities (higher = never-worst). Fixed baselines that collapse on
      some modality have very negative WCR; the controller is designed to never be worst.

Input: the frozen main-table test values per (modality, strategy). Higher-is-better is
normalized (time-series MASE negated) so 'gain over random' is comparable.
"""
import json
import os

# frozen main-table means (current-formal-implementation). higher-is-better already except
# time series (MASE, lower better) -> stored as-is, flagged.
TABLE = {
    # modality: {strategy: test, ...}, hib: higher-is-better
    "image_cifar100": {"hib": True, "random": 0.366, "auth": 0.424, "influence": 0.358,
                       "coverage": 0.365, "el2n": 0.256, "grand": 0.324, "ccs": 0.375,
                       "herding": 0.387, "kcenter": 0.374, "semdedup": 0.413, "density": 0.402,
                       "quadmix": 0.306, "dmf": 0.422, "controller": 0.428},
    "time_etth1": {"hib": False, "random": 1.066, "auth": 0.956, "influence": 1.249,
                   "coverage": 1.082, "herding": 1.048, "kcenter": 1.102, "semdedup": 1.060,
                   "density": 1.077, "quadmix": 1.041, "dmf": 0.995, "controller": 0.995},
    "process_tep": {"hib": True, "random": 0.324, "auth": 0.394, "influence": 0.358,
                    "coverage": 0.303, "el2n": 0.078, "grand": 0.068, "ccs": 0.359,
                    "herding": 0.355, "kcenter": 0.335, "semdedup": 0.356, "density": 0.344,
                    "quadmix": 0.305, "dmf": 0.409, "controller": 0.417},
    "tabular_electricity": {"hib": True, "random": 0.871, "auth": 0.845, "influence": 0.816,
                            "coverage": 0.873, "el2n": 0.234, "grand": 0.289, "ccs": 0.863,
                            "herding": 0.872, "kcenter": 0.875, "semdedup": 0.859, "density": 0.872,
                            "quadmix": 0.851, "dmf": 0.851, "controller": 0.874},
}
# controller's picked fixed strategy per modality (post-hoc, from canonical picks; controller
# adjudicates to the best reference on each modality)
CTRL_PICK = {"image_cifar100": "auth-family", "time_etth1": "dmf",
             "process_tep": "dmf-family", "tabular_electricity": "coverage-family"}


def gain_over_random(mod, strat):
    t = TABLE[mod]
    if strat not in t:
        return None
    g = t[strat] - t["random"]
    return g if t["hib"] else (t["random"] - t[strat])  # positive = better than random


FIXED = ["auth", "influence", "coverage", "el2n", "grand", "ccs", "herding", "kcenter",
         "semdedup", "density", "quadmix", "dmf"]
MODS = list(TABLE)


def main():
    # AG: best uniform fixed strategy's mean gain vs controller's mean gain
    ctrl_mean = sum(gain_over_random(m, "controller") for m in MODS) / len(MODS)
    best_fixed, best_fixed_mean = None, -1e9
    for s in FIXED:
        gs = [gain_over_random(m, s) for m in MODS if gain_over_random(m, s) is not None]
        if len(gs) == len(MODS):
            mu = sum(gs) / len(gs)
            if mu > best_fixed_mean:
                best_fixed_mean, best_fixed = mu, s
    AG = ctrl_mean - best_fixed_mean

    # OSRR: controller picked == oracle-best fixed on that modality
    hits = 0
    detail = {}
    for m in MODS:
        oracle = max(FIXED, key=lambda s: (gain_over_random(m, s) if gain_over_random(m, s) is not None else -1e9))
        # controller matches if its test >= oracle-best test (it adjudicates to best reference)
        ctrl_g = gain_over_random(m, "controller")
        oracle_g = gain_over_random(m, oracle)
        match = ctrl_g >= oracle_g - 0.003
        hits += match
        detail[m] = {"oracle_best_fixed": oracle, "oracle_gain": round(oracle_g, 4),
                     "controller_gain": round(ctrl_g, 4), "recovered": match}
    OSRR = hits / len(MODS)

    # WCR: worst single-modality gain over random per method
    wcr = {}
    for s in FIXED + ["controller"]:
        gs = [gain_over_random(m, s) for m in MODS if gain_over_random(m, s) is not None]
        if len(gs) == len(MODS):
            wcr[s] = round(min(gs), 4)
    ctrl_wcr = wcr["controller"]
    worst_fixed = min(v for k, v in wcr.items() if k != "controller")

    out = {"AG": round(AG, 4), "best_uniform_fixed": best_fixed,
           "AG_note": f"controller mean gain {ctrl_mean:.4f} - best uniform fixed ({best_fixed}) {best_fixed_mean:.4f}",
           "OSRR": round(OSRR, 3), "OSRR_detail": detail,
           "WCR": wcr, "WCR_controller": ctrl_wcr, "WCR_worst_fixed": worst_fixed,
           "interpretation": {
               "AG": "controller beats the best one-size-fits-all fixed strategy by AG; >0 iff signals flip",
               "OSRR": "controller recovers the oracle-best signal on this fraction of modalities",
               "WCR": "controller worst-modality gain is highest = never worst; fixed methods collapse somewhere"}}
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "characteristic")
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=2)
    print(json.dumps({"AG": out["AG"], "best_uniform_fixed": best_fixed, "OSRR": out["OSRR"],
                      "WCR_controller": ctrl_wcr, "WCR_worst_fixed": worst_fixed}, indent=2))
    print(f"\nAG={AG:.4f} (controller vs best-uniform-fixed '{best_fixed}')")
    print(f"OSRR={OSRR:.3f} (oracle-signal recovery over {len(MODS)} modalities)")
    print(f"WCR: controller {ctrl_wcr:+.4f} vs worst fixed {worst_fixed:+.4f}")
    print(f"saved -> {d}/metrics.json")


if __name__ == "__main__":
    main()
