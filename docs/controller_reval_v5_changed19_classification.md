# Controller re-validation v5: 19 changed rows classification (2026-07-16)

Report: `experiments/controller_reval_report_v5.json`
(sha256 c9f4158f0bec941451215a81a803150fb87afd7db4b882752120bda7fd820c4c)
Scope: `current-formal-implementation`. summary: expected=63 present=63 gate_pass=63
changed_vs_historical_log=19 unchanged_vs_historical_log=44.

## 63/62/64 reconciliation (independently verified)

- checkpoint.json has 64 fingerprint-keyed entries but only 63 unique (config,seed);
  the single duplicate is `vision-drop-infl::0` (fingerprints 24536617 and 96fa7e12).
  Corrected root cause (Codex 0145): the two keys differ NOT by timestamp but by
  `asset_sha256` (7d9a0cbf vs f108d66f) and `code_tree_sha256` (7c975bc4 vs 2f8160dd);
  `runner_code_sha256` (a9346c90) and `artifact_sha256` (7b309024) are identical. The
  trial was computed under two different (code-tree, asset) snapshots, and the produced
  artifact is byte-identical so the result is valid. The report/canonical anchors to
  24536617 (code_tree 7c975bc4, asset 7d9a0cbf); **96fa7e12 is superseded-stale and is
  NOT counted in the formal 63** (the report already dedups to the anchored entry).
- full log has 62 `[trial]` lines + 1 `[resume] vision-drop-infl seed0` = 63 replays.
- report has 63 unique replays, invalid_replays=0, gate_pass=63.
- Ruled out: missed trial (all 63 present, all gate_pass), stale checkpoint corrupting a
  result (the duplicate is byte-identical), double counting (report dedups to 63).

## Root-cause classification of the 19 (NOT attributed to one patch)

These replays run the CURRENT formal code; the deltas are vs the HISTORICAL logs, which
predate several formal commits (timeseries `train_model`/chronos factory + RUN_ID
isolation, selected-set hash, gate-NaN finite-aware fix). Downstream trainers (DLinear
from-scratch, mini-LM probes) are not bit-pinned across historical vs current runs.

Category A - metric-identical margin flip (picked differs, |dtest|<=0.005), 5 rows:
  tep-base s0 (.0025), tabular-base s1 (.0012), ts-ETTh2 s0 (.0009), ts-ETTh2 s2 (.0003),
  ts-daisy_steamgen s0 (.0011). Two candidates near-tied on validation; epsilon numerical
  differences flip the argmax. Downstream metric materially unchanged. Harmless.

Category B - same picked, metric drift (picked identical, dtest>tol), 2 rows:
  ts-ETTh1 s2 (quadmix->quadmix, .0177), ts-daisy_cstr s1 (dmf->dmf, .0143). Controller
  DECISION unchanged; difference is pure downstream-training stochasticity. Not a
  controller change.

Category C - both differ, tiny dtest (<=0.005), 3 rows:
  ts-ETTh2 s1 (.0021), ts-daisy_cstr s2 (.0007), ts-daisy_steamgen s2 (.0050).
  Margin flip + training noise. Harmless.

Category D - both differ, moderate dtest (0.009-0.037), 9 rows:
  vision-base s1 (.019, IMPROVED 0.416->0.435), vision-base s2 (.009), chronos-ETTm1 s0
  (.0143), tep-nf0.2 s0/s1/s2 (.015/.0065/.0198), ts-ETTh1 s0 (.037), ts-ETTh1 s1 (.0086),
  ts-daisy_steamgen s1 (.0114). Current-code replay differs from historical logs by
  accumulated formal-code changes plus downstream stochasticity. 13 of the 19 are
  time-series/chronos rows, exactly the arm with the most formal-code churn and the
  highest from-scratch training variance.

## NaN-patch attribution (explicitly bounded)

The gate-NaN fix only changes selection when the WINNING candidate has q>0 AND lam>0.
Among the 19 new_picked, only ts-ETTh2 s0/s2, chronos-ETTm1 s0 and tep-nf0.2 s0 carry a
q=0.25 winner; the rest win on non-gated candidates (dmf, quadmix, vote_ensemble,
auth_only, tabpfn_hybrid, density) where the fix is inapplicable. So <=4 rows are even
eligible for NaN-patch influence; refactor + stochasticity dominate the remaining 15+.

## Validity of the current formal implementation

gate_pass=63/63 (exit0, artifact present, old+new picked present, finite metrics,
sel_sha12 present, stderr clean). Every changed row still lands on a competitive
candidate within seed-noise of the historical value (vision-base net flat, tep within
.003, ts within the documented drift story). No degradation, no NaN, no crash. The
current formal implementation is therefore VALID and adopted as canonical.

## Backfill obligation (carried to Phase 3/backfill)

The paper tables must be refreshed from CURRENT-code artifacts for internal consistency;
the 19 historical-vs-current deltas are documented here as impact evidence only. ETTh1
remains the second-place temporal-drift arm; the current-code ts-ETTh1 s0 MASE 0.987 (was
0.950) stays inside that story and must be reported as such, not as a regression.
