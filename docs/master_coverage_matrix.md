# Master baseline x dataset coverage matrix (2026-07-16, REBUILD)

Audit item 二.1: mechanical coverage ledger rebuilt from REAL server artifacts (JSONs under `outputs/`,
`experiments/*.log`, `/root/*_OK` markers) on `omni:/root/autodl-tmp/OmniSelect`, 2026-07-16 evening.
Machine-readable twin: `experiments/master_coverage.json`. Nothing was rerun; the live QZ3 lane was not touched.
Supersedes the first-pass matrix of the same date (stale MISSING cells, stale 35% strict rate).

Code/config hash source: `experiments/results_matrix.json` `__server_code_state__` (head `0561b952`,
diff_sha256 `a0dc8900...`); live working-tree diff at rebuild time `a254017e...` (drift while QZ3 runs);
per-run JSON `code_sha256`/`code_sha256_12` fields pin each result.

Legend: `P3` = PASS 3-seed STRICT (per-seed exit evidence + parseable JSON); `P3w` = PASS_WEAK (canonical JSON
present, chain incomplete or disclosed structural failure); `P3r` = RECOVERED_RESULT_ONLY (numbers only in log
final table, log re-verified present); `P1*`/`P1*r` = 1-seed BY DESIGN qualitative curve; `P3q` = 3-seed but
reduced-scale QUALITATIVE tier only; `na` = NOT_APPLICABLE with per-cell algorithmic input-requirement reason
(see JSON); `RUN` = lane RUNNING on server (never pre-PASS); `MISS` = MISSING (none remain).

Coverage definition: **'all baselines covered' = every APPLICABLE cell PASS or RUNNING-with-live-evidence, and
every NOT_APPLICABLE cell carries an explicit reason.** Current state: 252/254 applicable cells complete; 2 RUNNING (QZ3), 0 MISSING.

Family key: TXT=text STRATIFY=1 main | GMIX=global-mix RegMix proxy | V100=CIFAR-100 | V100N=CIFAR-100N |
H1/H2/M1/CS/SG + d/c = ETTh1/ETTh2/ETTm1/DaISy-CSTR/DaISy-steamgen x DLinear/Chronos | TEP=unified testbed |
CAL2=TEP calib2 (dedicated held-out calibration split, 3x20) | TAB=electricity TabPFN | C10F=CIFAR-10 45k/160ep |
C10K=keep-sweep fidelity curve | C10G=geom coreset curve (COMPLETE, local qualitative) | IN100=ImageNet-100
reduced-scale qualitative. Off-table families (see JSON): `tabaicl_transfer` (supplementary transparency,
'Tab-AICL transfer' wording), `current_code_paired_batch` + `published_core_paired_batch` (controller
adjudication, both PASS).

| method | TXT | GMIX | V100 | V100N | H1d | H2d | M1d | CSd | SGd | H1c | H2c | M1c | CSc | SGc | TEP | CAL2 | TAB | C10F | C10K | C10G | IN100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `herding` | na | na | P3r | P3r | P3r | P3r | P3 | P3r | P3r | P3 | P3 | P3w | P3 | P3 | P3r | P3 | P3r | na | na | P1*r | na |
| `kcenter` | na | na | P3r | P3r | P3r | P3r | P3 | P3r | P3r | P3 | P3 | P3w | P3 | P3 | P3r | P3 | P3r | na | na | P1*r | na |
| `el2n` | na | na | P3r | P3r | na | na | na | na | na | na | na | na | na | na | P3r | P3 | P3r | P3 | P1*r | P1*r | P3q |
| `grand` | na | na | P3r | P3r | na | na | na | na | na | na | na | na | na | na | P3r | P3w | P3r | P3 | P1*r | P1*r | P3q |
| `ccs` | na | na | P3r | P3r | na | na | na | na | na | na | na | na | na | na | P3r | P3 | P3r | P3 | P1*r | P1*r | P3q |
| `semdedup` | na | na | P3r | P3r | P3r | P3r | P3 | P3r | P3r | P3 | P3 | P3w | P3 | P3 | P3r | P3 | P3r | na | na | na | na |
| `density` | na | na | P3r | P3r | P3r | P3r | P3 | P3r | P3r | P3 | P3 | P3w | P3 | P3 | P3r | P3 | P3r | na | na | na | na |
| `quadmix` | P3 | na | P3r | P3r | P3r | P3r | P3 | P3r | P3r | P3 | P3 | P3w | P3 | P3 | P3r | P3 | P3r | na | na | na | na |
| `dmf` | P3 | na | P3r | P3r | P3r | P3r | P3 | P3r | P3r | P3 | P3 | P3w | P3 | P3 | P3r | P3 | P3r | na | na | na | na |
| `dsir` | P3 | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na |
| `random` | P3 | P3 | P3r | P3r | P3r | P3r | P3 | P3r | P3r | P3 | P3 | P3w | P3 | P3 | P3r | P3 | P3r | P3 | P1*r | P1*r | P3q |
| `coreset` | na | na | P3r | P3r | P3r | P3r | P3 | P3r | P3r | P3 | P3 | P3w | P3 | P3 | P3r | P3 | P3r | na | na | na | na |
| `auth_only` | na | na | P3r | P3r | P3r | P3r | P3 | P3r | P3r | P3 | P3 | P3w | P3 | P3 | P3r | P3 | P3r | P3 | P1*r | P1*r | P3q |
| `influence_only` | na | na | P3r | P3r | P3r | P3r | P3 | P3r | P3r | P3 | P3 | P3w | P3 | P3 | P3r | P3 | P3r | na | na | na | na |
| `mmdataselect` | P3 | na | P3r | P3r | P3r | P3r | P3 | P3r | P3r | P3 | P3 | P3w | P3 | P3 | P3r | P3 | P3r | na | na | na | na |
| `mmds_adapt` | P3 | P3 | P3w | P3w | P3w | P3w | P3 | P3w | P3w | P3 | P3 | P3w | P3 | P3 | P3w | P3 | P3w | P3 | P1*r | P1*r | P3q |
| `zip` | RUN | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na |
| `noselect` | P3 | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na |
| `balance` | P3 | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na |
| `if_mates` | P3 | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na |
| `perpcorr` | P3 | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na |
| `quality_ppl` | P3 | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na |
| `regmix` | na | P3 | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na | na |
| `d4` | na | na | P3w | na | na | na | na | na | na | na | na | na | na | na | P3r | P3 | P3w | na | na | na | na |
| `dsdm` | na | na | P3w | na | na | na | na | na | na | na | na | na | na | na | P3r | P3 | P3w | na | na | na | na |
| `quadmix_pub` | RUN | na | na | na | na | na | P3 | na | na | na | na | na | na | na | na | P3 | na | na | na | na | na |
| `dmf_pub` | na | na | na | na | na | na | P3 | na | na | na | na | na | na | na | na | P3 | na | na | na | na | na |

## Rebuild corrections (verified on server, vs first-pass matrix)

1. **ETTm1 x DLinear is COMPLETE, not MISSING**: two run_ids (`ettm1-dlinear`, `codex-ettm1-dlinear-paired-20260716T1532`),
   `experiments/ettm1_dlinear_3seed.log` with `ETTM1DL SEED={0,1,2} python_exit=0`, per-seed results.json parse-verified,
   markers `/root/lane_ettm1_dlinear.done` + `/root/ETTM1_DLINEAR_PAIRED_OK` (runner/pairing/data/log sha256).
2. **CIFAR-10 full has all six rows (random/el2n/grand/ccs/auth/ctrl) at s0/1/2**: el2n rows exist at line 7 of every
   repro log (test 0.8784/0.8589/0.8470) -- the first-pass "no el2n row" claim was wrong; plus independent fill-in
   validation `experiments/el2n_cifar45k_fillin.json` (mean_test 0.8501, runner_sha256 `5de6da4c...`) +
   `/root/EL2N45K_VALIDATED_OK`. Fill-in used fresh score inits, values differ; both sources disclosed.
3. **ImageNet-100 s0/1/2 six methods incl. el2n** (rows verified in all three logs) -- entire family is
   **reduced-scale qualitative ONLY** (in_res=112, 40ep; config scale_note says so).
4. **geom curve COMPLETE**: keeps 0.1/0.3/0.5 all `python_exit=0`, `/root/lane_geom.done` + `/root/GEOM_LOCAL_CURVE_OK`;
   **local qualitative curve ONLY**, 1-seed by design; does not upgrade herding/kcenter fidelity tier.
5. **Current-code paired + published-core paired batches PASS**: `/root/CURRENT_CODE_PAIRED_MAIN_OK` and
   `/root/PUBLISHED_CORE_PAIRED_MAIN_OK`, 4 arms x 3 seeds each, per-row code_sha256_12; pubcore pins
   baseline_impl_sha256 `dea8ee64...`.
6. **TEP calib2 3x20 PASS** (`tep-calib2`, exit0 x3, 20 rows/seed, `/root/TEP_CALIB2_VALIDATED_OK`); **GraNd kept as
   n=2** -- seed2 structurally degenerate (test FDR=1.0000, FAR=1.0000, saturated threshold), 1/3 failure disclosed,
   never hidden or imputed.
7. **Tab-AICL standalone 3x3 PASS as 'Tab-AICL transfer', supplementary transparency table ONLY** (protocol mismatch:
   iterative active learning vs one-shot budgeted selection, BASELINE_FIDELITY_AUDIT.md line 24). Seed1 parity vs the
   earlier in-lane rows: coreset 0.8623 vs 0.8626, hybrid 0.8568 vs 0.8565, margin 0.8155 vs 0.8287 (drift disclosed).
8. **QuaDMix text transfer PASS** (`codex-text-quadmix-20260716T1550`, `/root/TEXT_QUADMIX_TRANSFER_OK` with full hash
   chain; claim_limit: local QuaDMix-style under the fixed stratified token budget, not the 570B original).
9. **QZ3 lane (quadmix_pub + zip x text) = RUNNING** (`text-qz3-20260716T2051`, PIDs 372096/372241 live, no done/failed
   marker) -- never pre-PASS; two earlier attempts (TEXTQZ2, scratch-proxy abort) superseded.
10. **D4/DsDm and family-specific methods = NOT_APPLICABLE with algorithmic input-requirement reasons** (embedding-space
    requirement for D4, datamodel requirement for DsDm, text-token inputs for text-lane methods); they PASS where the
    required inputs exist (vision CIFAR-100, tabular, TEP calib2).

## Family evidence (log + JSON + exit-code)


### text_stratify1_main
- Text lane, 5 textualized domains (code/general/image/math/table PPL), STRATIFY=1, infl=pplq, SmolLM2 finetune, lm-eval (ARC-e/c, HellaSwag, OBQA); lmeval field present in every results.json row
- protocol tier: **main-quantitative (unified-budget testbed)**
- seeds: 0,1,2
- logs: `experiments/text_scaleup_seed0.log`; `experiments/text_scaleup_seed1.log`; `experiments/text_scaleup_seed2.log`
- json: `outputs/experiment/stratify=1-infl=pplq-train=finetune-lmeval=1/seed_0/results.json`; `outputs/experiment/stratify=1-infl=pplq-train=finetune-lmeval=1/seed_1/results.json`; `outputs/experiment/stratify=1-infl=pplq-train=finetune-lmeval=1/seed_2/results.json`
- exit evidence: SEED=<s> python_exit=0 in each log
- note: quadmix cell filled by the QuaDMix text-transfer lane (run_id codex-text-quadmix-20260716T1550, /root/TEXT_QUADMIX_TRANSFER_OK); zip + quadmix_pub cells owned by the LIVE QZ3 lane (run_id text-qz3-20260716T2051) -- RUNNING, never pre-PASS

### text_globalmix_proxy
- Global-mix RegMix proxy lane (STRATIFY=0, infl=pplq, finetune, lm-eval). PROXY only, never enters STRATIFY main table (frozen gate Tier 4)
- protocol tier: **proxy-only (never enters STRATIFY main table)**
- seeds: 0,1,2
- logs: `experiments/text_globalmix_regmix_seed0.log`; `experiments/text_globalmix_regmix_seed1.log`; `experiments/text_globalmix_regmix_seed2.log`
- json: `outputs/experiment/stratify=0-infl=pplq-train=finetune-lmeval=1/seed_0/results.json`; `outputs/experiment/stratify=0-infl=pplq-train=finetune-lmeval=1/seed_1/results.json`; `outputs/experiment/stratify=0-infl=pplq-train=finetune-lmeval=1/seed_2/results.json`
- exit evidence: SEED=<s> python_exit=0 in each log; /root/lane_globalmix.done marker present

### vision_cifar100
- Vision CIFAR-100, frozen CLIP + linear head, unified-budget testbed (budget=0.5, noise_frac=0.4, pool=4000, val_n=800)
- protocol tier: **main-quantitative (unified-budget testbed)**
- seeds: 0,1,2
- logs: `experiments/vision_full_3seed.log`; `experiments/external_baselines_3seed.log`; `experiments/semdedup_density_rerun_3seed.log`; `experiments/vision_newbaselines_3seed.log`
- json: `outputs/vision/uoft-cs_cifar100/run_id=ctrlv5-vision-base-s{0,1,2}-*/seed_*/results.json (controller canonical)`; `outputs/vision/uoft-cs_cifar100/newbaselines-recovered/seed_{0,1,2}/results.json`
- exit evidence: external lane '#### EXT3 DONE ####' (results_matrix.json); vision_full '=== VISION 3-seed 完成 ==='; rerun '=== rerun done ==='; newbaselines none-recorded

### vision_cifar100n
- Vision CIFAR-100N real human-label noise (vis_noise=real), same testbed
- protocol tier: **main-quantitative (unified-budget testbed)**
- seeds: 0,1,2
- logs: `experiments/real_noise_cifar100n_3seed.log`
- json: `outputs/vision/uoft-cs_cifar100/run_id=ctrlv5-vision-realnoise-s{0,1,2}-*/seed_*/results.json (controller canonical)`
- exit evidence: '=== real noise done ===' (exit code none-recorded in results_matrix.json)

### time_etth1_dlinear
- Time series etth1, DLinear from scratch, unified-budget testbed (L=96 H=24 budget=0.3 noise=0.4 pool=3000)
- protocol tier: **main-quantitative (unified-budget testbed)**
- seeds: 0,1,2
- logs: `experiments/external_baselines_3seed.log`
- json: `outputs/timeseries/*/run_id=ctrlv5-ts-*-s{0,1,2}-*/results.json (controller canonical, ETTh1/ETTh2/daisy_cstr/daisy_steamgen)`
- exit evidence: #### EXT3 DONE #### + '=== TIMESERIES 3-seed 完成 ==='

### time_etth2_dlinear
- Time series etth2, DLinear from scratch, unified-budget testbed (L=96 H=24 budget=0.3 noise=0.4 pool=3000)
- protocol tier: **main-quantitative (unified-budget testbed)**
- seeds: 0,1,2
- logs: `experiments/etth2_3seed.log`
- json: `outputs/timeseries/*/run_id=ctrlv5-ts-*-s{0,1,2}-*/results.json (controller canonical, ETTh1/ETTh2/daisy_cstr/daisy_steamgen)`
- exit evidence: end-of-log results present; exit code none-recorded (results_matrix.json)

### time_daisy_cstr_dlinear
- Time series daisy_cstr, DLinear from scratch, unified-budget testbed (L=96 H=24 budget=0.3 noise=0.4 pool=3000)
- protocol tier: **main-quantitative (unified-budget testbed)**
- seeds: 0,1,2
- logs: `experiments/daisy_cstr_3seed.log`
- json: `outputs/timeseries/*/run_id=ctrlv5-ts-*-s{0,1,2}-*/results.json (controller canonical, ETTh1/ETTh2/daisy_cstr/daisy_steamgen)`
- exit evidence: end-of-log results present; exit code none-recorded (results_matrix.json)

### time_daisy_steamgen_dlinear
- Time series daisy_steamgen, DLinear from scratch, unified-budget testbed (L=96 H=24 budget=0.3 noise=0.4 pool=3000)
- protocol tier: **main-quantitative (unified-budget testbed)**
- seeds: 0,1,2
- logs: `experiments/daisy_steamgen_3seed.log`
- json: `outputs/timeseries/*/run_id=ctrlv5-ts-*-s{0,1,2}-*/results.json (controller canonical, ETTh1/ETTh2/daisy_cstr/daisy_steamgen)`
- exit evidence: end-of-log results present; exit code none-recorded (results_matrix.json)

### time_ettm1_dlinear
- Time series ETTm1, DLinear from scratch, unified-budget testbed (L=96 H=24 budget=0.3 noise=0.4 pool=3000). REBUILD CORRECTION: lane now EXISTS -- two run_ids on server
- protocol tier: **main-quantitative (unified-budget testbed)**
- run_ids: `ettm1-dlinear (paired_rng=0)`; `codex-ettm1-dlinear-paired-20260716T1532 (paired_rng=1)`
- seeds: 0,1,2
- logs: `experiments/ettm1_dlinear_3seed.log`
- json: `outputs/timeseries/ETTm1/run_id=ettm1-dlinear-H=24-L=96-budget=0.3-model=dlinear-noise=0.4-paired_rng=0-pool=3000/seed_{0,1,2}/results.json (15 method rows each, parse verified)`; `outputs/timeseries/ETTm1/run_id=codex-ettm1-dlinear-paired-20260716T1532-H=24-L=96-budget=0.3-model=dlinear-noise=0.4-paired_rng=1-pool=3000/seed_{0,1,2}/results.json (13 method rows each, parse verified)`
- exit evidence: ETTM1DL SEED={0,1,2} python_exit=0 (log lines 38/76/114) + /root/lane_ettm1_dlinear.done (ETTM1DL_DONE) + /root/ETTM1_DLINEAR_PAIRED_OK (status PASS; runner_sha256 0a728846..., pairing_sha256 b0cba7c8..., data_sha256 093cc4ef..., log_sha256 a5ebfae1..., per-seed json_sha256 recorded)
- note: closes the former MISSING family; docs/full_paper_coverage_ledger.md line 26 (ETTm1 chronos + DLinear) is now artifact-supported. Marker claim_limit: unified-budget coverage only, not original-paper fidelity evidence per named baseline.

### time_etth1_chronos
- Time series ETTh1, Chronos foundation model arm, unified-budget testbed
- protocol tier: **main-quantitative (unified-budget testbed)**
- seeds: 0,1,2
- logs: `experiments/chronos_fm_3seed.log`
- json: `outputs/timeseries/ETTh1/model=chronos-recovered/seed_{0,1,2}/results.json`; `outputs/timeseries/ETTh1/run_id=ctrlv5-chronos-ETTh1-s{0,1,2}-*/results.json (controller canonical)`
- exit evidence: CHRONOS_LANE_DONE (recorded in results_matrix.json)

### time_etth2_chronos
- Time series ETTh2, Chronos foundation model arm, unified-budget testbed
- protocol tier: **main-quantitative (unified-budget testbed)**
- seeds: 0,1,2
- logs: `experiments/chronos_fm_3seed.log`
- json: `outputs/timeseries/ETTh2/model=chronos-recovered/seed_{0,1,2}/results.json`; `outputs/timeseries/ETTh2/run_id=ctrlv5-chronos-ETTh2-s{0,1,2}-*/results.json (controller canonical)`
- exit evidence: CHRONOS_LANE_DONE (recorded in results_matrix.json)

### time_daisy_cstr_chronos
- Time series daisy_cstr, Chronos foundation model arm, unified-budget testbed
- protocol tier: **main-quantitative (unified-budget testbed)**
- seeds: 0,1,2
- logs: `experiments/chronos_fm_3seed.log`
- json: `outputs/timeseries/daisy_cstr/model=chronos-recovered/seed_{0,1,2}/results.json`; `outputs/timeseries/daisy_cstr/run_id=ctrlv5-chronos-daisy_cstr-s{0,1,2}-*/results.json (controller canonical)`
- exit evidence: CHRONOS_LANE_DONE (recorded in results_matrix.json)

### time_daisy_steamgen_chronos
- Time series daisy_steamgen, Chronos foundation model arm, unified-budget testbed
- protocol tier: **main-quantitative (unified-budget testbed)**
- seeds: 0,1,2
- logs: `experiments/chronos_fm_3seed.log`
- json: `outputs/timeseries/daisy_steamgen/model=chronos-recovered/seed_{0,1,2}/results.json`; `outputs/timeseries/daisy_steamgen/run_id=ctrlv5-chronos-daisy_steamgen-s{0,1,2}-*/results.json (controller canonical)`
- exit evidence: CHRONOS_LANE_DONE (recorded in results_matrix.json)

### time_ettm1_chronos
- Time series ETTm1, Chronos foundation model arm, unified-budget testbed
- protocol tier: **main-quantitative (unified-budget testbed)**
- seeds: 0,1,2
- logs: `experiments/chronos_ettm1_3seed.log`
- json: `outputs/timeseries/ETTm1/model=chronos-recovered/seed_{0,1,2}/results.json`; `outputs/timeseries/ETTm1/run_id=ctrlv5-chronos-ETTm1-s{0,1,2}-*/results.json (controller canonical)`
- exit evidence: CHRONOS_ETTM1_DONE (recorded in results_matrix.json)

### process_tep
- Process TEP (tep21), MLP fault classifier, unified-budget testbed (budget=0.3, noise_frac=0.4, pool=4000); FDR lanes are companion evidence
- protocol tier: **main-quantitative (unified-budget testbed)**
- seeds: 0,1,2
- logs: `experiments/tep_full_3seed.log`; `experiments/external_baselines_3seed.log`; `experiments/tep_fdr_full_3seed.log`; `experiments/tep_calibrated_fdr_3seed.log`; `experiments/semdedup_density_rerun_3seed.log`
- json: `outputs/tep/tep21/run_id=ctrlv5-tep-base-s{0,1,2}-*/results.json (controller canonical)`; `outputs/tep/tep21/fdr-uncalibrated-recovered/seed_{0,1,2}/results.json`; `outputs/tep/tep21/calib=1-*/seed_{0,1,2}/results.json`
- exit evidence: '#### EXT3 DONE ####' + '=== TEP 3-seed 完成 ===' + CPU_LANES_DONE (fdr) + TEP_CALIB_EXIT0/python_exit=0 (calibrated)

### tabular_electricity
- Tabular electricity, TabPFN-v2, unified-budget testbed (budget=0.5, noise_frac=0.4, pool=3000)
- protocol tier: **main-quantitative (unified-budget testbed)**
- seeds: 0,1,2
- logs: `experiments/tabular_full_3seed.log`; `experiments/tabular_external_3seed.log`; `experiments/tabular_newbaselines_3seed.log`; `experiments/semdedup_density_rerun_3seed.log`
- json: `outputs/tabular/electricity/run_id=ctrlv5-tabular-base-s{0,1,2}-*/results.json (controller canonical)`; `outputs/tabular/electricity/newbaselines-recovered/seed_{0,1,2}/results.json`
- exit evidence: '#### TAB DONE ####' (results_matrix.json) + '=== TABULAR 3-seed 完成 ==='

### cifar10_original_protocol_full
- CIFAR-10 original-protocol reproduction: 45k pool, ResNet-18 from scratch, 160ep, keep=0.3, SCORE_RUNS=3 score inits
- protocol tier: **original-protocol fidelity reproduction (quantitative)**
- seeds: 0,1,2
- logs: `experiments/cifar10_full_original_repro_seed0.log`; `experiments/cifar10_full_original_repro_seed1.log`; `experiments/cifar10_full_original_repro_seed2.log`
- json: `outputs/original_protocol/cifar10_full/seed_0/results.json`; `outputs/original_protocol/cifar10_full/seed_1/results.json`; `outputs/original_protocol/cifar10_full/seed_2/results.json`
- exit evidence: SEED=<s> python_exit=0 in each log; /root/lane_cifar.done marker present
- note: supersedes experiments/deepcore_original_protocol.log (early n=6000 run); REBUILD CORRECTION: el2n rows DO exist in all three repro logs (line 7) and in the per-seed results.json -- the first-pass matrix claim "no el2n row" was factually wrong

### cifar10_fidelity_curve_keep_sweep
- CIFAR-10 original-dataset fidelity curve, keep in {0.1,0.2,0.3,0.5,0.8}, ResNet-18 from scratch, SEED=0 only with SCORE_RUNS=3 score inits (qualitative-reduced-scale by design, FIDELITY_CURVE_OK gate)
- protocol tier: **QUALITATIVE CURVE ONLY (1-seed by design)**
- seeds: 0
- SEED FLAG: 1-seed BY DESIGN (qualitative curve-shape evidence, not a numeric table); flagged per audit rule
- logs: `experiments/fidelity_curve_cifar10_keep0.1.log`; `experiments/fidelity_curve_cifar10_keep0.2.log`; `experiments/fidelity_curve_cifar10_keep0.3.log`; `experiments/fidelity_curve_cifar10_keep0.5.log`; `experiments/fidelity_curve_cifar10_keep0.8.log`
- exit evidence: KEEP=<k> python_exit=0 in each of the 5 logs

### cifar10_geom_curve
- CIFAR-10 geometric-coreset curve (GEOM_CORESETS=1 adds herding/kcenter to the original-protocol script), keep in {0.1,0.3,0.5}, SEED=0, SCORE_RUNS=3. LANE COMPLETE (was RUNNING in the first-pass matrix).
- protocol tier: **LOCAL QUALITATIVE CURVE ONLY (1-seed by design; per baseline_fidelity_ledger.md completion does NOT auto-upgrade herding/kcenter fidelity tier)**
- seeds: 0
- SEED FLAG: 1-seed BY DESIGN (qualitative curve-shape evidence, not a numeric table)
- logs: `experiments/fidelity_geom_cifar10_keep0.1.log`; `experiments/fidelity_geom_cifar10_keep0.3.log`; `experiments/fidelity_geom_cifar10_keep0.5.log`
- exit evidence: KEEP={0.1,0.3,0.5} python_exit=0 (all three logs; 8 method rows per keep verified) + /root/lane_geom.done (GEOM_DONE) + /root/GEOM_LOCAL_CURVE_OK

### imagenet100_reduced_scale
- ImageNet-100 @112px/40ep reduced-scale qualitative protocol, keep=0.3 (n=36000), ResNet from scratch
- protocol tier: **REDUCED-SCALE QUALITATIVE ONLY: config scale_note='reduced-scale qualitative validation' (in_res=112, train_epoch=40, keep=0.3, n=36000); the six method rows (random/el2n/grand/ccs/auth/ctrl) may be cited as qualitative ordering evidence only, never as full-scale ImageNet numbers**
- seeds: 0,1,2
- logs: `experiments/imagenet100_protocol_seed0.log`; `experiments/imagenet100_protocol_seed1.log`; `experiments/imagenet100_protocol_seed2.log`
- json: `outputs/original_protocol/imagenet100/seed_0/results.json`; `outputs/original_protocol/imagenet100/seed_1/results.json`; `outputs/original_protocol/imagenet100/seed_2/results.json`
- exit evidence: SEED=<s> python_exit=0 in each log
- note: supersedes experiments/imagenet100_protocol.log (early single run)

### tep_calibrated2
- TEP calib2: calibrated operating point with a DEDICATED HELD-OUT CALIBRATION SPLIT (all te indices beyond val+test; disjoint from validation/V1/V2 and test; pre-registered); primary operating point = validation-calibrated 5% FAR target, threshold frozen pre-test. 3 seeds x 20 method rows.
- protocol tier: **calibrated-FDR companion lane (quantitative)**
- run_ids: `tep-calib2`
- seeds: 0,1,2
- logs: `experiments/tep_calibrated2_3seed.log`
- json: `outputs/tep/tep21/run_id=tep-calib2-budget=0.3-model=mlp-noise_frac=0.4-paired_rng=0-pool=4000/seed_{0,1,2}/results.json (20 rows each, parse verified)`
- exit evidence: TEPCALIB2 SEED={0,1,2} python_exit=0 (log lines 529/1058/1586) + /root/TEP_CALIB2_VALIDATED_OK (TEP_CALIB2_VALIDATED) + /root/lane_tep_calib2.done (TEP_CALIB2_DONE)
- note: companion to (not superseding) the earlier calib lane (experiments/tep_calibrated_3seed.log, /root/TEP_CALIB_VALIDATED_OK). GraNd seed2 structural failure disclosed in its cell.

### tabaicl_transfer
- Tab-AICL transfer: the three Tab-AICL rules (TabPFN-Coreset/Margin/Hybrid) transplanted onto the unified-budget tabular testbed (electricity, pool=3000, budget=0.5, noise_frac=0.4). SUPPLEMENTARY TRANSPARENCY TABLE ONLY -- Tab-AICL is iterative active learning (per-round new labels), a different problem from one-shot budgeted selection (experiments/BASELINE_FIDELITY_AUDIT.md line 24), so it is NEVER cited as a head-to-head main-table comparison; paper wording must be 'Tab-AICL transfer'.
- protocol tier: **SUPPLEMENTARY TRANSPARENCY ONLY**
- run_ids: `tabaicl-standalone-20260716T2014`
- seeds: 0,1,2
- logs: `experiments/tabaicl_standalone_3seed.log`
- json: `outputs/tabular/electricity/run_id=tabaicl-standalone-20260716T2014-budget=0.5-model=tabpfn-noise_frac=0.4-paired_rng=1-pool=3000/seed_{0,1,2}/results.json (code_sha256_12 d2b21debe8c5, baseline_impl_sha256 dea8ee64..., fidelity_mode published-core-unified-protocol-v1)`
- exit evidence: TABAICL SEED={0,1,2} python_exit=0 + /root/lane_tabaicl.done (tabaicl-standalone-20260716T2014)
- parity check vs earlier in-lane tabpfn_* rows in experiments/tabular_full_3seed.log (lines 43-45/89-91/135-137):
  - seed0: coreset 0.8758 vs 0.8757 | hybrid 0.8673 vs 0.8680 | margin 0.8218 vs 0.8159
  - seed1: coreset 0.8623 vs 0.8626 | hybrid 0.8568 vs 0.8565 | margin 0.8155 vs 0.8287
  - seed2: coreset 0.8877 vs 0.8877 (exact) | hybrid 0.8735 vs 0.8700 | margin 0.8599 vs 0.8644
  - verdict: coreset/hybrid reproduce within <=0.004 AUC (seed1 within 0.0003); margin drifts up to 0.0132 (seed1) -- disclosed, not hidden

### current_code_paired_batch
- Current-code paired batch: controller (ours row) re-run under paired_rng=1 across all four arms (vision CIFAR-100 / TEP / tabular electricity / timeseries ETTh1) x 3 seeds, current formal implementation.
- protocol tier: **paired-batch controller adjudication (quantitative)**
- run_ids: `codex-current-paired-20260716T1614`
- seeds: 0,1,2
- logs: `experiments/current_code_paired_codex-current-paired-20260716T1614.json (report)`
- json: `outputs/{vision/uoft-cs_cifar100,tep/tep21,tabular/electricity,timeseries/ETTh1}/run_id=codex-current-paired-20260716T1614-*/seed_{0,1,2}/results.json (per-row code_sha256_12 + fit_seed recorded in report)`
- exit evidence: /root/CURRENT_CODE_PAIRED_MAIN_OK (status PASS); report arms verified: 4 arms x seeds {0,1,2}

### published_core_paired_batch
- Published-core paired batch: same four arms x 3 seeds under the published-core faithful baseline implementations (fidelity_mode published-core-unified-protocol-v1).
- protocol tier: **paired-batch controller adjudication (quantitative)**
- run_ids: `pubcore-paired-20260716T1754`
- seeds: 0,1,2
- logs: `experiments/published_core_paired_pubcore-paired-20260716T1754.json (report)`
- json: `outputs/{vision/uoft-cs_cifar100,tep/tep21,tabular/electricity,timeseries/ETTh1}/run_id=pubcore-paired-20260716T1754-*/seed_{0,1,2}/results.json`
- exit evidence: /root/PUBLISHED_CORE_PAIRED_MAIN_OK (status PASS, baseline_impl_sha256 dea8ee64236126d0bcfc6c758231a901a64cd06d66664d9e98249eb21f502d1c); report arms verified: 4 arms x seeds {0,1,2}

## Superseded registry

- **external_baselines_3seed.log / tabular_external_3seed.log old grand==el2n degenerate rows (ICLR-draft numbers 0.256/0.076/0.214)** -> *_full_3seed.log independent last-layer GraNd proxy (AAAI numbers 0.324/0.068/0.289); baseline_fidelity_ledger.md section 4.1; ICLR draft must be backfilled
- **vision/TS/TEP/tabular semdedup+density values inside *_full_3seed.log (e.g. vision semdedup mean 0.321)** -> experiments/semdedup_density_rerun_3seed.log (fidelity-corrected implementations); ledger section 4.4
- **experiments/text_controller_2seed.log per-method PPL numbers (dsir 1914 vs random 2131, 2 seeds)** -> 3-seed text_scaleup lane results.json (pending paper backfill, ledger section 4.3); the controller-pick evidence itself ('both seeds picked dsir') remains 2-seed - legitimately-2-seed FLAG
- **experiments/deepcore_original_protocol.log (early CIFAR-10 n=6000 original-protocol run)** -> experiments/fidelity_curve_cifar10_keep*.log + cifar10_full_original_repro_seed{0,1,2}.log
- **experiments/imagenet100_protocol.log (early single run)** -> experiments/imagenet100_protocol_seed{0,1,2}.log
- **experiments/etth1_recentval_3seed.log** -> not a matrix cell: documented negative result (docs/negative_result_recent_val.md), recent-val split rejected
- **experiments/text_quadmix_zip_3seed_SCRATCH_PROXY_ABORTED_20260716T1927.log (scratch-proxy attempt, aborted by design)** -> QZ3 lane experiments/text_qz3_cacheaware_3seed.log (run_id text-qz3-20260716T2051, RUNNING)
- **experiments/text_quadmixpub_zip_3seed.log (TEXTQZ2 run_id text-qz2-20260716T1945; no python_exit lines, incomplete)** -> QZ3 lane experiments/text_qz3_cacheaware_3seed.log (run_id text-qz3-20260716T2051, RUNNING)

## Seed / honesty flags

- 2-seed family (legitimate, documented): text controller-pick evidence in experiments/text_controller_2seed.log; per-method numbers superseded by 3-seed scaleup (ledger 4.3)
- 1-seed families (legitimate BY DESIGN, qualitative curve-shape only): cifar10_fidelity_curve_keep_sweep and cifar10_geom_curve (both COMPLETE, python_exit=0 per keep)
- RESOLVED 2026-07-16: docs/full_paper_coverage_ledger.md line 26 (ETTm1 chronos + DLinear) is now artifact-supported (time_ettm1_dlinear family, two run_ids)
- RESOLVED 2026-07-16: EL2N-on-ImageNet wording is now supported -- el2n rows exist in all three imagenet100 logs (test 0.3192/0.3288/0.3078 vs random ~0.58), qualitative tier only
- TEP calib2 GraNd: 1/3 structural failure (seed2 FDR=FAR=1.0000); aggregate kept as n=2, disclosed never imputed
- Tab-AICL transfer margin seed1 parity drift 0.0132 AUC vs earlier in-lane row (0.8155 vs 0.8287) -- disclosed; coreset/hybrid reproduce within <=0.004
- exit-code 'none-recorded' (log ends with well-formed final table, no exit marker) remains true for: etth2/daisy_cstr/daisy_steamgen DLinear logs, semdedup_density_rerun, real_noise_cifar100n, vision/tabular newbaselines logs -- acceptable, noted; all these logs re-verified PRESENT on server

## Action list

| # | cell | class | detail |
|---|---|---|---|
| 1 | zip + quadmix_pub x text_stratify1_main | **RUNNING (QZ3, do not touch)** | run_id text-qz3-20260716T2051, PIDs 372096/372241; absorb after /root/lane_text_qz3.done; never pre-PASS |
| 2 | (none MISSING) | **closed** | all previously-MISSING cells resolved by real artifacts on 2026-07-16: ettm1-dlinear lane, el2n cifar10-full + imagenet100 rows, geom curve, QuaDMix text transfer; zip reclassified RUNNING |

## Grade summary (rebuild, 2026-07-16 (rebuild))

| grade | count | meaning |
|---|---|---|
| STRICT_PASS | 111 | per-seed exit evidence + parseable per-seed JSON + full seed set |
| PASS_WEAK | 25 | canonical JSON present, chain incomplete / disclosed structural failure (reason per cell) |
| RECOVERED_RESULT_ONLY | 116 | numbers only in log final table (log re-verified present); no per-method canonical JSON |

Totals: 252 PASS cells, 2 RUNNING, 0 MISSING, 321 NOT_APPLICABLE
(of 575 cells across 24 families).

**Strict completion rate = 111/252 = 44.0%** (was 71/203 = 35% in the
first-pass matrix; the gain comes from real artifacts landing -- ettm1-dlinear lane, el2n rows, TEP calib2, Tab-AICL
transfer, QuaDMix transfer, paired batches -- not from regrading leniency; the 116 RECOVERED cells keep their tier,
only their stale "log missing" reasons were corrected to "log present, no per-method canonical JSON").
Only STRICT_PASS counts toward strict completion. Honesty over completeness.
