# Method-v2 mechanical diagnosis — conclusions (methodv2_diag_20260717)

Audit 1450 item 3. All numbers in `diagnosis.json` (same dir). Sources: selreplay selection
manifests (`outputs/*/*/run_id=selreplay-*/seed_{0,1,2}/results.json`), pubcore-paired results +
adapt_manifest leaderboards (`run_id=pubcore-paired-20260716T1754-*`), `experiments/
canonical_tables.json` final_cells, `experiments/controller_current_canonical_v5.json`,
`experiments/text_controls_stats.json`, `outputs/experiment/run_id=text-qz4-20260717T0933-*`.
CPU-only, read-only on all existing artifacts; outputs written only to this new dir.

Tag reconstruction was replicated standalone (`default_rng(seed+7)`, first-permutation partition,
arm-specific split) and VERIFIED on every arm x seed: every recomputed high-fraction matches the
recorded `clean%` of the sha12-matched pubcore row to 3 decimals (0 mismatches). Purity numbers
below are therefore trusted.

## Per-arm verdicts

### tabular (electricity, TabPFN, AUC) — verdict: no-signal-separation + seed-noise. NO actionable gap.
- Saturated by rule: full-random gain 0.0182 <= 2*std(random)=0.022.
- Purity is uncorrelated with test AUC: controller selected-set purity 0.5711 < random 0.6033,
  yet controller 0.8734 > random 0.8671; auth_only with the highest purity (0.712) scores 0.8433
  (worst tier). TabPFN is robust to the injected noise; geometry/diversity dominates.
- Controller vs strongest fixed (kcenter 0.8753): -0.0019, within pooled std 0.0201.
- Val probe is NOT the problem: leaderboard val-test rho 0.947 (candidates are simply near-tied).
- Jaccard(ctrl): kcenter 0.685, dmf_pub 0.386, auth_only 0.296, random 0.331 — controller sits in
  the geometry family, not the cleaning family, and that is the right neighborhood here.

### tep (tep21, MLP, F1) — verdict: seed-noise at ceiling (controller-near-ceiling). NO actionable gap.
- Controller 0.4079 >= full 0.4057; gap to strongest fixed dmf_pub 0.409 is -0.0011 vs pooled
  std 0.0062. Leaderboard rho 0.886. Purity 0.797 near auth_only ceiling 0.838.
- Jaccard(ctrl, dmf_pub)=0.804, (ctrl, auth_only)=0.676 — selections already largely shared.

### timeseries ETTh1 (DLinear, MASE) — verdict: real-mechanism-gap. THE ONLY actionable arm.
- Drift hypothesis SUPPORTED: leaderboard val-test Spearman mean 0.6202 (per-seed
  0.7212 / 0.6848 / 0.4545) vs 0.886-0.947 on tep/vision/tabular — materially lower.
  (Caveat: the v5-rows timeseries spearman of -0.84 is a sign convention — new_val is
  negative-MASE val_gain — and indicates alignment across configs, not drift.)
- Mechanism: raw argmax over a near-tied, drifting val leaderboard picks a different strategy
  every seed: seed0 "fuse w=(0.5,0,0.5)" (switched from best_ref dmf) -> 0.9675; seed1 dmf_pub ->
  0.9475; seed2 quadmix -> 1.0075 (canonical cell 1.0284 after the protocol-invalid style-proxy
  replacement) vs auth_only 0.9461. Worst pick coincides with the worst per-seed rho (0.4545).
- Cost: controller 0.9811+-0.0421 vs auth_only 0.9501+-0.0053 — 0.031 MASE mean gap and 8x std.
  Formally within pooled std 0.0424, but that pooled std is dominated by the controller's own
  pick instability, which is exactly the mechanism.
- Purity: ctrl 0.850 vs auth_only 0.889, dmf_pub 0.923 — mild, secondary to the pick problem.

### vision (cifar100, budget 0.5) — verdict: undertraining-or-saturation (quantity-dominated). NO actionable selection-quality gap.
- "Open-headroom" by rule (full-random 0.1575; full-ctrl 0.1148) but full trains on 4000 samples
  vs the budget's 2000 — a quantity effect, not selection quality.
- The controller is already at the purity ceiling: 0.801 vs auth_only 0.8013; residual flips
  0.0075 (~eliminated). In the vision recipe only "flip" actually perturbs data (dup/hard are
  tag-only markers), so no purity headroom remains.
- Jaccard(ctrl, auth_only)=0.919, (ctrl, dmf_pub)=0.929 — selections near-identical already.
- Ctrl vs strongest fixed (auth_only 0.426): -0.002, within pooled std 0.0096. Leaderboard rho 0.916.

### text — verdict: no-signal-separation under the current finetune regime. NOT a candidate target.
- Old claims (docs/code_signal_diagnosis.md, old stratified from-scratch regime): code loses to
  DMF by ~1.2%; general loses to random (2472 vs 2338 = +5.7% in the doc's table; the task's
  "~4.2%" figure is NOT-CAPTURED in any current doc).
- Current per-domain ppl (text-qz4 20260717 x3 seeds vs text-controls same code state):
  mmds_adapt vs random: code +0.024%, general +0.012%, image -0.194%, math -0.171%,
  table +1.706%. vs noselect: code +0.396%, general -0.115%, image +1.046%, math +0.166%,
  table +4.945%. The old gaps do NOT persist — everything is compressed to ~ties (table slightly
  worse than random/noselect).
- dmf / dmf_pub / dsir per-domain ppl rows do not exist in the current batches
  (controls = base/random/noselect; qz4 = quadmix_pub/zip/mmds_adapt): NOT-CAPTURED — the 1.2%
  code-vs-DMF gap cannot be re-measured on the current code state.

## Ranked mechanism hypotheses (what a candidate could exploit)

1. **ETTh1 val-probe drift x near-tie argmax (strongest evidence).** The val leaderboard ranks
   strategies at rho 0.45-0.72 against test on ETTh1 (0.89-0.95 elsewhere) and the top candidates
   are near-tied, so the raw argmax flips per seed and occasionally lands a bad pick (seed2:
   1.0284 vs auth_only 0.9461). Candidate lever: tie-aware adjudication (deviate from a stable
   reference only when the paired val margin exceeds noise, else keep the reference) and/or a
   drift-robust val probe (blocked/rolling validation). Target: recover ~0.031 MASE mean and cut
   the controller std ~8x. This is the only mean-metric win available.
2. **Purity is saturated wherever it matters.** Controller purity already sits at the auth_only
   ceiling on vision (0.801 vs 0.8013) and near it on tep/ts; on tabular purity anti-correlates
   with the metric. Any candidate whose mechanism is "filter noise better" has no headroom on any
   arm; remaining full-vs-controller gaps are quantity (vision) or model-robustness (tabular)
   effects. Candidates should NOT target tabular/tep/vision/text.
3. **Cross-arm, the only other improvement space is pick-stability (variance), not mean.** On all
   four arms ctrl-vs-best is within pooled seed noise and the pick differs per seed while test
   barely moves (consistent with docs/method_v2_diagnostic.md). A stability mechanism is
   defensible and metric-neutral everywhere except ETTh1, where it doubles as hypothesis 1.

## NOT-CAPTURED register
- The "general-domain ~4.2% gap vs random" figure: absent from every current doc; closest
  captured figure is +5.7% (code_signal_diagnosis.md old table).
- dmf/dmf_pub/dsir per-domain text ppl under the current code state (rows absent from
  text-controls-20260717T0026 and text-qz4-20260717T0933).
- docs/method_v2_diagnostic.md lives in the local repo and /root/autodl-tmp/OmniSelect_v2/docs/,
  not in the main server repo docs/ (content identical in substance; used the local copy).
- v5 rows give only 3 tabular points (single config), so the tabular v5-rows spearman (0.5, n=3)
  is uninformative; the leaderboard rho (0.947) is the authoritative tabular val-test number.
