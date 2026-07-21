<!-- LANG -->
**English** | [简体中文](BENCHMARK_zh.md)

# OmniSelect-Benchmark: protocol

This document specifies the benchmark protocol implemented by this repository: what task
each arm evaluates, how the quality-fluctuating pool is constructed, the equal-budget /
candidate-reference-holdout isolation rules every method (baseline or controller) must
follow, the run artifact schema, and the seeding convention. See [README.md](README.md) for
quick start and the current results table, and
[docs/CONTRIBUTING_DATASETS.md](docs/CONTRIBUTING_DATASETS.md) for the mechanics of adding a
new dataset, modality, or baseline.

## What is being benchmarked

Given a fixed training budget, which subset of a noisy data pool should you keep? This
benchmark measures that question across four modalities (image, time series, tabular,
process-fault-diagnosis text/sensor logs) with a shared evaluation harness, so a selection
method's cross-modal profile is directly comparable: does it stay near the top everywhere,
or does it collapse on some modality because it depends on a signal that only works there.

Nine headline dataset/noise configurations ship with committed evidence
(`results_canonical/`): CIFAR-100, CIFAR-100N (real human label noise), ETTh1, ETTm1, ETTh2,
DaISy CSTR, DaISy steam generator, Tennessee Eastman (TEP21), and OpenML Electricity. CIFAR-10
(original DeepCore protocol) is carried as a separate baseline-fidelity lane, not one of the
nine.

## Quality-fluctuating pool

A pool of pure high-quality data cannot measure selection value: every method looks equally
good. Every pool here is instead built as roughly 60% high-quality plus 40% controlled,
per-record-tagged low-quality injections, drawn from at least four kinds of degradation
appropriate to the modality (e.g. label-flip / near-duplicate / cross-domain / corrupted
input for vision; corrupt / duplicate / flat-signal / shuffled for time series and process
data). The tag is never exposed to any selection method, only used afterward to report
`clean_pct`, the fraction of a method's selected subset that is actually high-quality.

## Equal-budget protocol

Every method compared for a given dataset/seed selects the **same budget** `k` from the
**same pool**, using the **same downstream model** and the **same train/eval split**, so
differences in the reported metric are attributable to *which* records were kept, not to how
many or under what training conditions. Budgets and pool/test sizes are fixed per
dataset/modality (see `scripts/run_*_experiment.py` and `docs/architecture.md`)
and never tuned per method.

## Candidate / reference / holdout isolation

Every run enforces a three-way split to prevent the selection signal from leaking into the
number that is supposed to measure it:

- **Candidate pool**: what every method selects from.
- **Reference / validation set**: drives signal computation and (for the controller) the
  adjudication decision between candidate strategies. Never touched by the final metric.
- **Held-out test set**: only ever scored once, after selection and training are complete.

A method that reads its own test-set performance to pick its selection rule, or that fits a
signal on the same records it later selects from, would violate this and is not a valid entry.

## The one hard invariant for baselines

**Every compared baseline is executed as a candidate the controller can adjudicate over, not
tabulated separately from a `results.json` it was never run through.** Concretely, baselines
are passed into the controller via `extra_strategies=[(name, fn), ...]` in the same call that
evaluates the controller itself (see `docs/CONTRIBUTING_DATASETS.md`), so the controller's
"never worse than the portfolio on validation" guarantee is measured, not assumed. A baseline
implementation whose published protocol structurally differs from this benchmark's one-shot
equal-budget setting (e.g. an iterative active-learning loop) is disclosed as such rather than
silently forced into this setting; see `docs/baseline_fidelity_ledger.md` for the per-baseline
fidelity tier (T1 official-code parity / T2 original-paper-dataset replication / T3 mechanism
test) and honest boundaries.

## Run artifact schema

Every run writes one `results.json` per `(modality, dataset, run tag, seed)`
(`outputs/<modality>/<dataset>/<tags>/seed_N/results.json`; `<tags>` includes `run_id=...`
only when `RUN_ID` is set). Schema: [docs/results_schema.json](docs/results_schema.json). Key
fields per method row: `metric` (task metric, direction is modality-specific), `n` (selected
count), `clean_pct`, `sel_sha12` / `train_order_sha12` (selection and training-order
fingerprints for exact-reproduction checks). `results_canonical/` holds the small, committed
subset of these files that every number in the paper is read from directly, no retraining
needed (`run_scripts/reproduce_cached.sh`).

## Seeds

The committed evidence in `results_canonical/` spans seeds `{0, 1, 2}` where a full 3-seed
sweep was run; single-seed evidence is seed 0 by convention. `run_scripts/run_single_arm.sh`
and `run_scripts/reproduce_full.sh` default to `SEED=0` / `SEEDS="0"` and take an explicit
`SEEDS="0 1 2"` override to reproduce the 3-seed check. New evidence or showcase additions
(e.g. `experiments/selreplay_evidence/`) show seed 0 as a representative illustration rather
than committing all three, to keep the repository lean; this does not change what was
actually measured; per-seed fingerprints for the full sweep live in `results_canonical/`.

## Dataset provenance and licensing

Every dataset's official source, pinned revision/version, SHA256, license, and "used
unmodified except for the disclosed noise injection" statement is recorded in
[docs/dataset_provenance.md](docs/dataset_provenance.md) and
[docs/ARTIFACTS_INDEX.md](docs/ARTIFACTS_INDEX.md). Datasets without a confirmed
redistribution license (CIFAR-10/100, ImageNet-100) are not committed as raw bytes; run
`python scripts/fetch_data.py` to fetch and SHA-verify them from their official sources. TEP
and DaISy ship as committed raw files under `data/` (public domain / research-reuse sources,
small enough to commit directly); their per-file SHA256 is in
`docs/provenance_evidence/`.

## Adding a baseline or a dataset

See [docs/CONTRIBUTING_DATASETS.md](docs/CONTRIBUTING_DATASETS.md) for the full template
(`run_scripts/add_dataset.sh` scaffolds a new arm). In short: write a loader with the same
controlled-noise recipe, pass every baseline (including the new one) into the controller as a
candidate, emit `results.json` in the standard layout, and record dataset provenance.

## Citation

If you use this benchmark, please cite the OmniSelect paper (citation details to be added
once the paper is public; see [README.md](README.md) for the current preprint status).
