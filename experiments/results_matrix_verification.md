# results_matrix.json spot-check verification

Date: 2026-07-16. Matrix regenerated on server (`ssh omni`, repo `/root/autodl-tmp/OmniSelect`)
by `scripts/build_results_matrix.py` (verification-grade ledger version), then copied back to
local `experiments/results_matrix.json`.

## Server code state (recorded in matrix under `__server_code_state__`)

- head: `0561b95286052577d33d634c7603553b3d322232` (from `git -C /root/autodl-tmp/OmniSelect rev-parse HEAD`)
- diff_sha256: `ec1782a8e49f09f034179acb3176f739ae9d5c8d461aed7b427d57f64802b36a` (sha256 of full `git diff` output, dirty-tree fingerprint)

Builder run output on server: all 14 views parsed, `conflicts` and `duplicate_sections` are
empty lists for every view (script exits nonzero otherwise; exit code was 0). One pre-existing
warn unchanged from the old script: `chronos_fm_3seed.log: no rows parsed for time_ettm1_chronos`
(that log contains no ETTm1 sections; ETTm1 chronos lives in a separate log not in SOURCES).

## Spot-check method (independent of the parser)

For each view: seed 0, method `mmds_adapt` (present in all 14 views). Section extracted with
awk (literal-string anchor match, section ends at the next `#`-header containing seed/SEED,
same scoping rule the logs use), then plain grep. Run on the server in
`/root/autodl-tmp/OmniSelect/experiments`:

```sh
awk -v a='<ANCHOR>' 'index($0,a){f=1;next} f&&/^#/&&/[Ss][Ee][Ee][Dd]/{f=0} f' <LOG> \
  | grep -E '^  mmds_adapt ' \
  | grep -oE '(<METRIC_KEYS>)=[0-9.]+' | sed 's/^[^=]*=//' | sort -u
```

Each log prints every method twice per seed section (a timing line and a summary line with the
same value); `sort -u` returning exactly ONE value per view additionally confirms the two
occurrences agree (matching the matrix's empty `conflicts`).

Per-view `<LOG>` / `<ANCHOR>` / `<METRIC_KEYS>` substitutions:

| view | LOG | ANCHOR | METRIC_KEYS |
|---|---|---|---|
| vision_cifar100 | external_baselines_3seed.log | `#### VIS-cifar100 seed0 ####` | `test_acc\|acc` |
| time_etth1 | external_baselines_3seed.log | `#### TS seed0 ####` | `MASE` |
| process_tep | external_baselines_3seed.log | `#### TEP-mlp seed0 ####` | `macroF1\|F1` |
| time_etth2 | etth2_3seed.log | `SEED 0 (ETTh2` | `MASE` |
| time_daisy_cstr | daisy_cstr_3seed.log | `SEED 0 (DaISy CSTR` | `MASE` |
| time_daisy_steamgen | daisy_steamgen_3seed.log | `SEED 0 (DaISy steamgen` | `MASE` |
| vision_cifar100n | real_noise_cifar100n_3seed.log | `REALN SEED=0` | `test_acc\|acc` |
| vision_cifar100__semdedup_rerun | semdedup_density_rerun_3seed.log | `SEED 0 VISION` | `test_acc\|acc` |
| process_tep__semdedup_rerun | semdedup_density_rerun_3seed.log | `SEED 0 TEP` | `macroF1\|F1` |
| tabular_electricity | tabular_external_3seed.log | `#### TAB-tabpfn seed0 ####` | `auc` |
| time_etth1_chronos | chronos_fm_3seed.log | `CHRONOS ds=ETTh1 SEED=0` | `MASE` |
| time_etth2_chronos | chronos_fm_3seed.log | `CHRONOS ds=ETTh2 SEED=0` | `MASE` |
| time_daisy_cstr_chronos | chronos_fm_3seed.log | `CHRONOS ds=daisy_cstr SEED=0` | `MASE` |
| time_daisy_steamgen_chronos | chronos_fm_3seed.log | `CHRONOS ds=daisy_steamgen SEED=0` | `MASE` |

## Spot-check rows (all 14 views, seed 0, method mmds_adapt)

| view | method | seed | raw-grep value | matrix value | result |
|---|---|---|---|---|---|
| vision_cifar100 | mmds_adapt | 0 | 0.4285 | 0.4285 | MATCH |
| time_etth1 | mmds_adapt | 0 | 0.9691 | 0.9691 | MATCH |
| process_tep | mmds_adapt | 0 | 0.3843 | 0.3843 | MATCH |
| time_etth2 | mmds_adapt | 0 | 0.6230 | 0.623 | MATCH |
| time_daisy_cstr | mmds_adapt | 0 | 1.0041 | 1.0041 | MATCH |
| time_daisy_steamgen | mmds_adapt | 0 | 0.8348 | 0.8348 | MATCH |
| vision_cifar100n | mmds_adapt | 0 | 0.3885 | 0.3885 | MATCH |
| vision_cifar100__semdedup_rerun | mmds_adapt | 0 | 0.4330 | 0.433 | MATCH |
| process_tep__semdedup_rerun | mmds_adapt | 0 | 0.4138 | 0.4138 | MATCH |
| tabular_electricity | mmds_adapt | 0 | 0.8568 | 0.8568 | MATCH |
| time_etth1_chronos | mmds_adapt | 0 | 0.8689 | 0.8689 | MATCH |
| time_etth2_chronos | mmds_adapt | 0 | 0.5821 | 0.5821 | MATCH |
| time_daisy_cstr_chronos | mmds_adapt | 0 | 0.9235 | 0.9235 | MATCH |
| time_daisy_steamgen_chronos | mmds_adapt | 0 | 0.7091 | 0.7091 | MATCH |

14/14 MATCH (0.6230 vs 0.623 and 0.4330 vs 0.433 are the same float; the matrix stores
JSON numbers without trailing zeros). No parser fixes were needed.

## Ledger fields added per view (backward compatible; characteristic_metrics_v2.py re-run on server unchanged, regenerated experiments/characteristic_metrics_v2.json successfully)

- `protocol_type`: "unified-budget transfer (shared testbed)" (all views)
- `applicability_na`: for the 8 time-series (MASE) views, `{el2n, grand, ccs}` marked
  "classification-error based, not applicable to regression"; `{}` elsewhere
- `expected_seeds`: 3; `parse_counts`: per (method, seed) occurrence counts within that seed section
- `conflicts` / `duplicate_sections`: empty lists in ALL 14 views (matrix trusted)
- `config_hash`: sha256 of the SOURCES tuple repr for the view
- `exit_evidence`: terminal-marker lines grepped from the source log (`DONE|python_exit=|_OK`):
  - `#### EXT3 DONE ####`: vision_cifar100, time_etth1, process_tep
  - `#### TAB DONE ####`: tabular_electricity
  - `CHRONOS_LANE_DONE`: all 4 chronos views
  - `none-recorded` (honest, no marker in log): time_etth2, time_daisy_cstr,
    time_daisy_steamgen, vision_cifar100n, both `__semdedup_rerun` views
- `__server_code_state__` (top level): key deliberately contains `__` so the consumer's
  existing aux-view filter (`if "__" in ds: continue`) skips it without any code change.
