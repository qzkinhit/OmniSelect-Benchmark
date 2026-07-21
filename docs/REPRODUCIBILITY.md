# Reproducibility guide

This document gives the per-table / per-figure recipes for the OmniSelect paper. All
paths are repo-relative. All commands run from the repo root. Numbers in the
manuscripts are **never hand-maintained**: the headline table is regenerated from
`experiments/canonical_tables_seed0.json` (see "Canonical table regeneration" below).

**Headline result**: OmniSelect ranks first-or-tied against 11 standard baselines
(random, coreset, herding, EL2N, GraNd, CCS, Density, QuaDMix-published-
transfer, DMF-published-transfer, influence-only, fixed-weight fusion) on 9 datasets:
CIFAR-100, CIFAR-100N, ETTh1, ETTm1, ETTh2, TEP21, Electricity, DaISy
CSTR, DaISy steamgen. Strict first on 5, tied for first on 4, never beaten. Four
methods (auth_only, dmf-proxy, kcenter, semdedup) are the controller's own portfolio candidates
and are excluded from the 11-baseline comparison by definition, not by outcome. They
still run inside `src/` and `baselines/`. Regenerate this table with
`python scripts/build_canonical_seed0.py`. Every number comes straight out of a
`results.json` row under `results_canonical/`, no hand-typed numbers.

Sections 2-3 below additionally document an earlier four-arm protocol (4 core
datasets, mean±std over repeated runs) kept for provenance; the 9-dataset table
above is the paper's headline result.

## 0. Environment

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e . && pip install -r requirements.txt
pip install -e ".[train,eval,dev]"     # torch/transformers/datasets/lm-eval/pytest
```

Server lock used for the full-scale runs: `environment/pip_freeze_server_vgpu.txt`
(Python 3.12 cp312, `torch 2.8.0+cu128`, single NVIDIA vGPU-32GB class card, CUDA
12.8). Small lanes also run on Apple MPS / CPU (device auto-selected `cuda → mps →
cpu`).

## 1. RUN_ID convention and output layout

Every arm runner writes one atomic JSON per (run, seed):

```
outputs/<arm>/<dataset>/run_id=<RUN_ID>-<sorted config tags>/seed_<SEED>/results.json
```

e.g. `outputs/vision/uoft-cs_cifar100/run_id=<RUN_ID>-budget=0.5-noise_frac=0.4-paired_rng=1-pool=4000-val_n=800-vis_noise=inject/seed_0/results.json`.
The payload records the full config, `code_sha256_12`, `baseline_impl_sha256`,
`fidelity_mode`, the published-core protocol constants when active, the per-method
results (with `sel_sha12` selection fingerprints and `train_order_sha12` training-order
fingerprints), and the pairing manifest. The text runner writes
`outputs/experiment/<tags>/seed_<SEED>/results.json` with tags like
`stratify=1-infl=pplq-train=finetune-lmeval=1`.

Named batches referenced below:

- `run_id=pubcore-paired-20260716T1754` (the FINAL main/external-table batch): all
  four arms, one code state, seeds 0/1/2, `PAIRED_RNG=1`, per-seed shared `fit_seed`,
  validated by `scripts/validate_published_core_paired.py`
  (marker `PUBLISHED_CORE_PAIRED_MAIN_OK`, independent parity re-check 9/9 MATCH).
- `run_id=text-qz4-20260717T0933`: the text-lane zip/quadmix_pub completion run.
- `run_id=split-export-20260717`: the select-only replay that exported the split-ID
  manifests and matched the pubcore batch `arrays_sha256` exactly (12/12).

## 2. Main table + external-baselines table (four arms)

Source: `experiments/canonical_tables.json` → `latex_FINAL_main_table` (rows: Random,
Coreset (k-center greedy), Authenticity-only, Influence-only, Dynamic fusion (DMF),
Controller (ours), QuaDMix (published-core transfer), DMF (published-update transfer))
and `latex_FINAL_external_table` (Herding, k-center, EL2N, GraNd (proxy), CCS,
SemDeDup, Density, QuaDMix (published-core transfer)). Columns in order:
`vision_cifar100`, `time_etth1`, `process_tep`, `tabular_electricity`.

Shared protocol for all four arms (the "pubcore" lane config, from
`scripts/run_published_core_paired_lane_a.sh` / `_lane_b.sh`):

```bash
export RUN_ID=<your-run-id> PAIRED_RNG=1 FIDELITY_MODE=published-core-unified-protocol-v1
export ADAPT_GRPO=0 ADAPT_MARGIN=0.015 ADAPT_SH=0 ROBUST_VAL=0 VAL_NOISE=0
METHODS=full,random,coreset,auth_only,influence_only,mmdataselect,herding,kcenter,el2n,grand,ccs,semdedup,density,quadmix,quadmix_pub,dmf,dmf_pub,mmds_adapt
# (time-series lane drops el2n,grand,ccs from METHODS: not applicable there)
```

Note: `quadmix` (the style proxy) is part of the lane METHODS for pairing parity but
its results are INVALID_WITHDRAWN (duplicate selected ids,
`docs/quadmix_styleproxy_invalidation.md`) and never enter any table. `quadmix_pub`
is the sole QuaDMix representative.

Seeds: `for seed in 0 1 2` around each command. Pool composition everywhere: 60%
clean + 40% controlled noise (`NOISE_FRAC=0.4`), tagged per record. Candidate,
clean-reference, and held-out splits are disjoint, with equal budget across methods.

### 2.1 Vision: CIFAR-100, frozen CLIP + linear probe, top-1 (higher is better)

```bash
SEED=$seed METHODS=$METHODS VIS_DATASET=uoft-cs/cifar100 VIS_ENCODER=openai/clip-vit-base-patch32 \
  POOL_N=4000 VAL_N=800 TEST_N=2000 NOISE_FRAC=0.4 BUDGET_FRAC=0.5 KNN=15 \
  LAM=0.5 AUTH_Q=0.25 W_INFL=0.5 VIS_NOISE=inject \
  python scripts/run_vision_experiment.py
```

Budget is 50% of a 4000-image pool. Noise kinds are label-flip, near-duplicate, and
hard-ambiguous. The controller adjudicates on the independent clean `VAL_N=800` split.

### 2.2 Time series: ETTh1, DLinear from scratch, MASE (lower is better)

```bash
SEED=$seed METHODS=$METHODS_TS TS_DATASET=ETTh1 TS_MODEL=dlinear \
  POOL_N=3000 VAL_N=1000 TEST_N=1500 L=96 H=24 NOISE_FRAC=0.4 BUDGET_FRAC=0.3 \
  KNN=15 EPOCHS=60 LAM=0.5 AUTH_Q=0.25 W_INFL=0.5 TS_VAL_MODE=full \
  python scripts/run_timeseries_experiment.py
```

Selection is over training windows (input length 96, horizon 24, OT channel). Noise
kinds are corrupt, flat, shuffle, and near-duplicate. `TS_MODEL=chronos` swaps in the
fine-tuned Chronos-Bolt-tiny foundation model (the `*_chronos` views).

### 2.3 Process industry: TEP, MLP fault classifier, macro-F1 (higher is better)

```bash
SEED=$seed METHODS=$METHODS MODEL=mlp N_FAULTS=21 \
  POOL_N=4000 VAL_N=2000 TEST_N=3000 NOISE_FRAC=0.4 BUDGET_FRAC=0.3 KNN=15 \
  LAM=0.5 AUTH_Q=0.25 W_INFL=0.5 TEP_CALIB=0 \
  python scripts/run_tep_experiment.py
```

52 process variables, normal + faults 1–21 (22 classes), held-out test from the
`d*_te.dat` files. The calibrated-detection view (`tep_calibrated2`: FDR@FAR5,
balanced accuracy, AUPRC) comes from the `tep-calib2` lane (`TEP_CALIB` enabled with a
dedicated large-normal-sample calibration split). `scripts/canonical_paper_tables.py`
aggregates it from `outputs/tep/*/*tep-calib2*/seed_*/results.json`, but this glob currently
matches nothing in the active `outputs/` tree (the `tep-calib2` data only exists in older
server backup snapshots), so this view is not reproducible from the current checkout.

### 2.4 Tabular: OpenML electricity, TabPFN-v2 support set, ROC-AUC (higher is better)

```bash
SEED=$seed METHODS=$METHODS TAB_DATASET=electricity MODEL=tabpfn \
  POOL_N=3000 VAL_N=2500 TEST_N=2000 NOISE_FRAC=0.4 BUDGET_FRAC=0.5 KNN=15 \
  LAM=0.5 AUTH_Q=0.25 W_INFL=0.5 \
  python scripts/run_tabular_experiment.py
```

Selection = TabPFN's in-context support set (no gradient training). Noise kinds are
label-flip, feature-corruption, and near-duplicate.

## 3. The other 5 headline datasets (CIFAR-100N, ETTh2/ETTm1, DaISy CSTR/steamgen)

These 5 datasets complete the 9-dataset headline table in `canonical_tables_seed0.json`
(see the "Headline result" paragraph above, not a numbered section). They use the same runners as section 2 with `TS_DATASET` ∈ {ETTh2, ETTm1,
daisy_cstr, daisy_steamgen}, `TS_MODEL=dlinear` (the Chronos-model runs under the same
`TS_DATASET` values are a separate downstream-model comparison, not part of this
table), and `VIS_DATASET=uoft-cs/cifar100 VIS_NOISE=real` for CIFAR-100N (same base
images as CIFAR-100, real human-annotated labels from Wei et al. 2022 instead of
injected noise. The two are disambiguated by `config.vis_noise`, not by dataset name,
since both runs carry `dataset="uoft-cs/cifar100"`). Committed results:
`results_canonical/timeseries/{ETTh2,ETTm1,daisy_cstr,daisy_steamgen}/` and the
`vis_noise=real` run under `results_canonical/vision/uoft-cs_cifar100/`.
(`results_canonical/vision/uoft-cs_cifar10/` is a separate CLIP-linear-probe run not among
the 9 headline datasets; it is not part of this table.) The older 20-view log-parsed ledger
(`experiments/results_matrix.json`, 3-seed mean, built by
`scripts/build_results_matrix.py`) is legacy provenance for an earlier protocol
iteration and is no longer the source of any manuscript number.

## 4. Text lane (five textualized domains, SmolLM2-135M)

Runner: `scripts/run_experiment.py`. Pool: `data/processed/qpool_train.jsonl`
(25,000 records, 5,000 per domain general/math/code/image-proxy/table-proxy, 60/40
high/low quality, heldout 2,000 (see `docs/dataset_provenance.md`). Main
configuration (matches the ledgered output path
`outputs/experiment/stratify=1-infl=pplq-train=finetune-lmeval=1/seed_{0,1,2}/results.json`):

```bash
SEED=$seed STRATIFY=1 INFL_KIND=pplq TRAIN_MODE=finetune LM_EVAL=1 \
  REF_MODEL=HuggingFaceTB/SmolLM2-135M \
  python scripts/run_experiment.py
```

Stratified per-domain token budget (`BUDGET_FRAC` default 0.5). Metrics = per-domain
held-out perplexity + lm-eval tasks (`LMEVAL_TASKS` default
`arc_easy,arc_challenge,hellaswag,openbookqa`). Evidence:
`experiments/master_coverage.json` cells `text_stratify1_main` /
`text_globalmix_proxy`. **Scope limit:** the zip / quadmix_pub cells are graded
PASS_WEAK. The text lane's runtime split/noise/init fingerprints were NOT_CAPTURED at
run time and are never re-derived post hoc. No general-model text claims are made, and
none may be added.

## 5. Fidelity / anchor lanes (not main-table)

- `cifar10_original_protocol_full`: from-scratch ResNet-18 on CIFAR-10
  (`baselines/deepcore_original/run_original_protocol.py`,
  `experiments/deepcore_original_protocol.log`): reproduces the original-paper
  *relative order* (CCS > random > EL2N/GraNd at high pruning), qualitative tier only.
- `cifar10_fidelity_curve_keep_sweep`, `cifar10_geom_curve`: 1-seed BY DESIGN,
  qualitative curve shape only.
- `imagenet100_reduced_scale`: qualitative tier.
- `cifar10_ccs_anchor`: official released-implementation CCS anchor
  (`docs/ccs_anchor_protocol.md`, `experiments/ccs_anchor_canonical.json`), its own
  tier. It does NOT upgrade the local EL2N-binned CCS and does not change
  `reproduction_verified = NONE`.
- DSIR official alignment: `experiments/dsir_official_fidelity.log` (Spearman 0.8553,
  top-50% overlap 0.8546 against the official `data-selection` package).

## 6. Canonical table regeneration

```bash
python scripts/build_canonical_seed0.py
```

Scans every `results.json` under the committed `results_canonical/` tree, keeps seed 0
only, and writes `experiments/canonical_tables_seed0.json` with the 9-dataset x
11-baseline verdict (`_meta.n_strict_first`, `_meta.n_tied_first`, `_meta.n_beaten`)
plus every cell's raw value. This is the manuscript's headline source. No GPU, no
downloads, seconds to run. `run_scripts/reproduce_cached.sh` calls it directly.

**Legacy (superseded)**: `python scripts/canonical_paper_tables.py` regenerates the
older `experiments/canonical_tables.json` (4 datasets, 3-seed mean±std, and, unlike
the seed-0 table above, does not exclude the controller's own portfolio candidates
auth_only/dmf/kcenter from the comparison rows). Kept for audit-trail provenance only.
Do not cite its numbers in the manuscript.

## 6.5 Fidelity dual status

Two flags answer different questions and are never merged (see `docs/baseline_fidelity_ledger.md` header):

- `strict_original_protocol_reproduction = NONE`: no baseline claims the original paper's absolute numbers under the original full protocol.
- `baseline_fidelity_evidence_closure = COMPLETE_WITH_DISCLOSED_TIERS`: official implementations are used where available, otherwise the published selection rules are faithfully reimplemented (formulas, ranking direction, sampling and budget semantics checked). Every in-paper comparison shares pools, budgets, downstream models, training schedules, and seeds. The per-baseline nine-field anchor table is complete and machine-checked (`python scripts/check_fidelity_nine_field.py`).

Standing labels that never disappear: GraNd = last-layer proxy, QuaDMix = published-core transfer, DMF = published-update transfer, ImageNet-100 = reduced-scale qualitative, CCS local implementation and the official released-implementation anchor are two separate tiers.

## 7. Status tiers (exactly as used in `experiments/master_coverage.json`)

Statuses (`legend`):

- **PASS**: current artifacts verified in a named log/JSON. Every PASS carries a
  grade (below).
- **NOT_APPLICABLE** — the method cannot run in that family for an algorithmic
  input-requirement or frozen-protocol reason, stated per cell (321 cells).
- **MISSING**: cannot be verified from any artifact. **None remain** (0).
- **SUPERSEDED**: older artifact replaced. Pointer kept in the superseded registry.
- **INVALID_WITHDRAWN**: withdrawn results (16 cells): the QuaDMix *style proxy*,
  whose selection replay proved duplicate selected_ids
  (grade `PROTOCOL_INVALID_DUPLICATE_IDS`, see
  `docs/quadmix_styleproxy_invalidation.md`). `quadmix_pub` is the sole QuaDMix
  representative.

Grades on PASS cells:

- **STRICT_PASS** (110): per-seed exit evidence (python_exit=0 or validated `*_OK`
  marker with hashes) + parseable per-seed results JSON + full seed set.
- **PASS_WEAK** (26): parseable canonical JSON present but exit/marker chain
  incomplete, or a disclosed per-seed structural failure (n<3 kept, never imputed).
- **RECOVERED_RESULT_ONLY** (108): numbers live only in a re-verified log final
  table. No per-method canonical JSON.

Totals: 244 valid cells, strict completion rate 0.4508. The coverage claim is
*task-applicable computational coverage: 0 MISSING among applicable cells*, NOT a
literal Cartesian claim and NOT original-protocol reproduction.

Fidelity labels used alongside (from `canonical_tables.json` → `fidelity_labels` and
`docs/baseline_fidelity_ledger.md`):

- **TRANSFER**: a published-protocol implementation run on our testbed, not the
  original benchmark: `quadmix_pub` (published-core transfer), `dmf_pub`
  (published-update transfer), `tabpfn_*` (Tab-AICL transfer, standalone).
- **PROXY**: a mechanism-level stand-in: GraNd = last-layer gradient-norm proxy,
  DMF = dynamic-fusion proxy. Proxies are labeled in the row names.
- **NOT_CAPTURED**: evidence that was not captured at run time and is never
  re-derived post hoc (text-lane runtime fingerprints, CCS-anchor exact selected ids).

Baseline reproduction status: **reproduction_verified = NONE** across the board
(`docs/baseline_fidelity_ledger.md`, hard lock). Tier definitions T1/T2/T3 are in
`experiments/BASELINE_FIDELITY_AUDIT.md`.

## 8. Secondary / exploratory analyses

`scripts/characteristic_metrics_v4.py` and `_v5.py` carry explicit disclosures and
must be reported as such:

- v4: "DISCLOSURE: SECONDARY / EXPLORATORY, post-hoc; primary metrics remain the
  standard task metrics." (dataset-cluster bootstrap, common 7-method pool, Holm
  correction.)
- v5: "DISCLOSURE: SECONDARY / EXPLORATORY, post-hoc. Supersedes v4's dominance
  inference (whose percentile-bootstrap p-values over only 8 clusters are
  approximate; v4's 10/10 is relabeled exploratory)." (exact sign / sign-flip tests
  over 8 clusters, Holm step-down.) The per-cell FINAL numbers never come from these
  scripts.

Outputs: `experiments/characteristic_metrics_v4.json` / `_v5.json`.

## 9. Seed and pairing policy

- Seeds **0 / 1 / 2** everywhere in the quantitative tables. Report mean±std
  (sample stdev). n<3 is kept and disclosed, never imputed (e.g. TEP-calib2 GraNd
  seed2 structural failure → n=2, disclosed).
- `PAIRED_RNG=1` gives all methods within a seed the same RNG stream (paired
  batches). The pubcore batch additionally shares a per-seed `fit_seed`.
- Every method row carries a `sel_sha12` (SHA of the selected-id set) and
  `train_order_sha12` (SHA of the training order) fingerprint in `results.json`, so
  selections and training orders can be compared bit-for-bit across runs.
- Deliberate exceptions, all documented in `master_coverage.json` → `seed_flags`:
  1-seed qualitative curve families, and the legacy 2-seed text controller-pick log.
- Noise injection RNG is `np.random.default_rng(seed + 7)` per arm runner. The
  derived corruption pool is deterministic given config + seed.

## 10. Smoke test

```bash
sh scripts/repro_smoke.sh
```

Creates a venv, runs the pure-CPU sanity smoke, pytest, one tiny experiment per arm
whose prerequisites are present (raw data are not in git, so absent arms are SKIPPED,
not failed), and regenerates the canonical tables into a temp directory. Exits
nonzero on any FAIL.
