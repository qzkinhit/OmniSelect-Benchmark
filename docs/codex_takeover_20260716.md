# Codex server takeover handoff, 2026-07-16

## Scope

Claude reached its usage limit around 15:06 local time. Codex took over only the
two approved baseline-coverage gaps below. Existing successful artifacts were not
overwritten. The ongoing geometric-coreset lane was not interrupted or modified.

## 1. ETTm1 x DLinear unified-budget coverage

- Status: PASS, three seeds, 13 rows per seed including the full-data reference.
- Run ID: `codex-ettm1-dlinear-paired-20260716T1532`
- Server marker: `/root/ETTM1_DLINEAR_PAIRED_OK`
- Server log: `/root/ettm1_dlinear_paired_codex-ettm1-dlinear-paired-20260716T1532.log`
- Data SHA256: `093cc4efd56a6bf68fb20cc93a2a79a4fbb06f02c8f4e7e5efa5520cc68afce6`
- Runner SHA256: `0a7288462d2e447961ad443cf9f618fdddc457dab44418e4931f2e5ddb1c8aad`
- Pairing helper SHA256: `b0cba7c898a2ad463dd46397734255add92cfb0c288b1e3bcc9d8d85b17c8f5c`
- Output root: `outputs/timeseries/ETTm1/run_id=codex-ettm1-dlinear-paired-20260716T1532-*`
- Mean MASE across seeds: OmniSelect controller 1.0234, auth-only 1.0524,
  DMF 1.0786, full 1.0995, fixed OmniSelect 1.1026, QuaDMix 1.1165,
  random 1.2289.
- Claim limit: this closes the paper's ETTm1 unified-budget coverage gap. It is
  not original-paper numerical reproduction evidence for every named baseline.

Pairing instrumentation records selected-ID SHA, selection-dependent fit seed,
and training-order SHA. When the controller selects an exact named baseline, the
validator requires exact selection, seed, order, and metric parity. In this run
all three controller choices were composite fusion strategies, so the named-row
parity clause was not applicable.

## 2. QuaDMix text transfer

- Status: PASS, three seeds, five-domain PPL and four lm-eval tasks.
- Run ID: `codex-text-quadmix-20260716T1550`
- Server marker: `/root/TEXT_QUADMIX_TRANSFER_OK`
- Server log: `/root/text_quadmix_codex-text-quadmix-20260716T1550.log`
- Pool SHA256: `84e174dbb097288c6b4473af2af8d6cb46a0b00a000b6534545725e53f9939c5`
- Heldout SHA256: `1e4a45c9c959995a3c10c840dc2a8b84ce33bd82ba17682c8f50c3b4b3a1e785`
- Runner SHA256: `06eb3433cb18552425925fc75f670f9faa7137838bc1e5cf3f4eebbe84fa6c42`
- Output root: `outputs/experiment/run_id=codex-text-quadmix-20260716T1550-*`
- Geometric-mean PPL by seed: 11.341, 11.352, 11.351.
- Claim limit: local QuaDMix-style quality and diversity implementation under
  the paper's fixed stratified token budget. This is not the original 570B-token
  QuaDMix reproduction. The result is worse than the existing random and DSIR
  rows, so no dominance claim is allowed.

The first isolated smoke test stalled because the Hugging Face client attempted
an unreachable network endpoint. It was stopped without affecting any formal
artifact. The batch now sources `/root/omni_env.sh` and uses the existing local
cache in offline mode. The formal three-seed run exited cleanly.

## 3. Coverage-ledger correction

The current `docs/master_coverage_matrix.md` incorrectly marks EL2N as missing
for both CIFAR-10 full original-protocol and ImageNet-100 reduced-scale. Direct
server inspection shows all six canonical JSON files already contain an `el2n`
row:

- `outputs/original_protocol/cifar10_full/seed_{0,1,2}/results.json`
- `outputs/original_protocol/imagenet100/seed_{0,1,2}/results.json`

Do not rerun those cells. Regrade the matrix from the JSON evidence instead.

## 4. Local backup

The 17 new result, log, marker, runner, validator, and helper files were copied to:

`server_backup_codex_takeover_20260716T1557/`

An rsync checksum dry run plus direct marker and log SHA comparison returned
`DUAL_SHA_OK files=17`.

## 5. Still running

The pre-existing geometric-coreset curve remains under `/root/lane_geom.sh`.
Keep 0.1 and 0.3 exited zero. Keep 0.5 started at 14:38 and was still in the
CPU selection phase at 15:56. Its Python PID at that time was 297774. Do not
restart completed keep fractions. Wait for `/root/lane_geom.done`, then require
the independent watcher verdict before adopting the curve.

## 6. Required next steps

1. Validate and back up the geometric-coreset result when its watcher finishes.
2. Rebuild the master coverage matrix, correcting the two false EL2N gaps and
   adding the ETTm1 and QuaDMix evidence above.
3. Keep ZIP text as cited-only or run it later on a cheaper CPU-oriented instance.
   The current 25k implementation spends hours in CPU compression selection and
   is a poor use of the RTX 6000D.
4. Rebuild the post-takeover artifact manifest, environment snapshot, canonical
   tables, and timestamped local backup. Verify every file by SHA before changing
   instances.
5. Preserve the fidelity labels. Industrial original protocols for QuaDMix,
   Density, SemDeDup, and DSIR were not reproduced by these runs.

## Server files changed by Codex

- `scripts/run_timeseries_experiment.py`
- `src/mmdataselect/utils/pairing.py`
- `scripts/validate_timeseries_paired.py`
- `scripts/run_ettm1_dlinear_paired.sh`
- `scripts/run_experiment.py`
- `scripts/validate_text_quadmix.py`
- `scripts/run_text_quadmix_3seed.sh`

Pre-Codex runner copies are preserved under
`/root/codex_takeover_20260716T1530/` and
`/root/codex_takeover_20260716T1545/`. Do not use `git add -A` on the server.

## 7. Current-code paired main-table rerun

Audit after the first ETTm1 batch found that its paired fit seed depended on the
selected-id hash. That guarantees parity for identical selections, but does not
give different methods the same initialization. The ETTm1 result therefore remains
coverage evidence and is not adopted as the final fairness-controlled row.

The pairing rule was corrected so every method in one paper seed receives the same
initialization. Selected integer ids are sorted before fitting. Every row records
the selected-id SHA, fit seed, and actual training-order SHA. Pool, validation, and
test fingerprints are stored in `pairing_manifest`. If the controller selects a
named standalone method, the fail-closed validator requires exact selected-id and
test-metric parity.

The current-code main-table batch started at 16:14 with run id
`codex-current-paired-20260716T1614`. It runs the 16 numerical classification rows
on CIFAR-100, TEP-21, and electricity, plus the 13 applicable time-series rows on
ETTh1, all over seeds 0/1/2. The strict marker is
`/root/CURRENT_CODE_PAIRED_MAIN_OK`. No result may be adopted until that marker is
created by `scripts/validate_current_code_paired.py`.
