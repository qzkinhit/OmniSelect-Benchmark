<!-- LANG -->
**English** | [简体中文](README_zh.md)

# OmniSelect

**Cross-modal robust data selection via certified policy adjudication.**

Under a fixed training budget, indiscriminately accumulating samples dilutes the useful
signal, so *which* data to keep matters. The catch OmniSelect is built around: **no single
quality signal and no fixed fusion wins across modalities**: the best signal flips with the
modality, the downstream model, and even the dataset. A method that is great on one
modality can collapse below random on another.

OmniSelect turns *which selection strategy to use* into a question each modality answers with
its own clean validation set and its own downstream model. It holds a **frozen, audited
portfolio of candidate strategies**: three complementary quality signals (authenticity,
influence, coverage), their fusions, and every compared baseline, each **executed as a
candidate** (never read from a results file). It then **adjudicates** among them. The deployed adjudicator
freezes an ordered `(reference, challenger)` pair on a construction half of the validation
set and **certifies** a switch on the other half with a metric-specific one-sided confidence
radius, so the probability of adopting a harmful challenger is at most `δ = 0.05`
(Theorem 4''). It switches only with a certificate and otherwise keeps the reference. That is
how it stays in the top tier everywhere and avoids catastrophic failure.

> The Python package keeps the historical name `mmdataselect`. **OmniSelect** is the system
> name used in the paper.

This repository doubles as **OmniSelect-Benchmark**: a shared, cross-modal harness for
evaluating data-selection methods under one equal-budget protocol, with baselines executed
as candidates (never as compare-only rows) and every number backed by a committed
`results.json`. See [BENCHMARK.md](BENCHMARK.md) for the full protocol spec (task
definitions, noise taxonomy, candidate/reference/holdout isolation, run schema, seed policy)
and [docs/CONTRIBUTING_DATASETS.md](docs/CONTRIBUTING_DATASETS.md) to add a dataset or
baseline.

---

## Quick start (CPU, no GPU, no downloads)

```bash
git clone https://github.com/qzkinhit/OmniSelect-Benchmark.git
cd OmniSelect-Benchmark
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"              # core + CPU test dependencies
pytest -q                             # CPU-only test suite
run_scripts/reproduce_cached.sh       # rebuild the paper's canonical tables from committed results
```

`reproduce_cached.sh` rebuilds the fixed-primary-run headline table
(`experiments/canonical_tables_seed0.json`) from the committed `results_canonical/`
files. Every number is read straight out of a `results.json` row, no GPU, seconds to run.
The complete 12-task artifact inventory (including text) is indexed in
[`results_canonical/README.md`](results_canonical/README.md).

## Run baselines and OmniSelect yourself

One modality × one dataset × one run, any subset of methods:

```bash
# install the training extras once (GPU recommended for vision/timeseries/tabular)
pip install -e ".[train,eval,arms]"   # arms = tabpfn/xgboost/chronos-forecasting, needed by the tabular/timeseries arms
python scripts/fetch_data.py                      # fetch + SHA-verify the datasets

# <arm> <dataset> <seed> [methods]
run_scripts/run_single_arm.sh vision      uoft-cs/cifar100    0
run_scripts/run_single_arm.sh timeseries  ETTh1       0   random,auth_only,mmds_adapt
run_scripts/run_single_arm.sh tabular     electricity 0
run_scripts/run_single_arm.sh text        five_domain 0
```

Every run writes `outputs/<arm>/<dataset>/<tags>/seed_N/results.json` with one row per
method (all baselines **and** the controller), each carrying the metric, the selected-subset
fingerprint `sel_sha12`, and the training-order fingerprint `train_order_sha12`. `<tags>`
includes a `run_id=...` segment only when the `RUN_ID` env var is set (as `reproduce_full.sh`
does internally); a plain `run_single_arm.sh` call as shown above omits it. The same
`METHODS` list runs the baselines and OmniSelect under **one equal-budget protocol**, so the
comparison is apples-to-apples by construction.

Commands for the complete 12-task benchmark inventory:

```bash
run_scripts/run_single_arm.sh vision      uoft-cs/cifar100    0   # CIFAR-100
VIS_NOISE=real run_scripts/run_single_arm.sh vision uoft-cs/cifar100 0   # CIFAR-100N (same images, real human labels)
run_scripts/run_single_arm.sh vision      uoft-cs/cifar10     0   # CIFAR-10, CLIP protocol
DATASET=imagenet100 SEED=0 python baselines/deepcore_original/run_original_protocol.py
run_scripts/run_single_arm.sh timeseries  ETTh1       0
run_scripts/run_single_arm.sh timeseries  ETTm1       0
run_scripts/run_single_arm.sh timeseries  ETTh2       0
run_scripts/run_single_arm.sh timeseries  daisy_cstr  0   # DaISy CSTR
run_scripts/run_single_arm.sh timeseries  daisy_steamgen 0   # DaISy steam generator
run_scripts/run_single_arm.sh tep         21          0   # TEP21
run_scripts/run_single_arm.sh tabular     electricity 0
run_scripts/run_single_arm.sh text        five_domain 0
```

CIFAR-10 (`run_scripts/run_single_arm.sh vision uoft-cs/cifar10 0`) is a separate
baseline-fidelity dataset (original DeepCore protocol, not one of the 9 headline rows above);
see `results_canonical/vision/cifar10_full/` and `docs/baseline_fidelity_ledger.md`.

Set `SPLIT_EXPORT_DIR=<dir>` on any of the commands above to additionally dump the pool/
validation/test index split (`pool_ids`/`val_ids`/`test_ids` + the seeding recipe) shared by
every method in that run; without it, `results.json` only carries the `sel_sha12` fingerprint.

Full reproduction (all 12 tasks; the default is the fixed primary run, and `SEEDS`
can be set explicitly for robustness runs):

```bash
run_scripts/reproduce_full.sh
SEEDS="0 1 2" run_scripts/reproduce_full.sh
```

## Add your own dataset / modality

OmniSelect is a benchmark: plugging in a new dataset is a small, contained change. See
[`docs/CONTRIBUTING_DATASETS.md`](docs/CONTRIBUTING_DATASETS.md) for the full template. In
short, you (1) write a loader + a controlled-noise recipe, (2) pass **every** baseline into the
controller as a candidate (the one hard invariant: a baseline is a candidate, never a
compare-only column), (3) emit `results.json` in the standard layout, (4) record the source +
SHA in `docs/dataset_provenance.md`. A scaffold is provided:

```bash
run_scripts/add_dataset.sh my_new_dataset          # copies an arm runner into a labeled template
```

## Repository layout

```
src/mmdataselect/      system core: quality signals, the adjudication controller, selectors
baselines/             faithful baseline implementations (method/ + run_*.py + README each)
run_scripts/           one-command entrypoints (single-arm, reproduce_cached/full, add_dataset)
scripts/               per-modality arm runners, fetch_data.py, table builders, validators
results_canonical/     committed small results behind every number in the paper
experiments/           canonical JSON ledgers + verified run logs + split-ID manifests
data/                  small raw sets in git (TEP, DaISy), large sets are pointer + SHA
docs/                  reproducibility, dataset provenance, baseline fidelity, architecture
environment/           pinned locks (CPU and CUDA 12.8 / torch 2.8.0)
tests/                 CPU-only pytest suite (system + baseline fidelity gates)
```

Start with [`data/README.md`](data/README.md) for the 12-task download matrix and
[`results_canonical/README.md`](results_canonical/README.md) for the modality/dataset/
baseline/result map. [`docs/README.md`](docs/README.md) lists the intentionally small
set of public technical documents.

## Results

OmniSelect ranks first against 11 standard baselines (random, coreset, herding,
EL2N, GraNd, CCS, Density, QuaDMix-published-transfer, DMF-published-transfer,
influence-only, fixed-weight fusion) on 9 datasets spanning image classification, long-horizon
forecasting, process fault diagnosis, and tabular classification: CIFAR-100, CIFAR-100N,
ETTh1, ETTm1, ETTh2, DaISy CSTR, DaISy steam generator, Tennessee Eastman (TEP21), and OpenML
Electricity. Strict first on 5, tied for first on 4, never beaten by any of the 11 on any of
the 9. Every method shares identical pools, budgets, downstream models, and seeds, executed
under one equal-budget protocol so every comparison is apples-to-apples by construction. Full
per-run logs are kept in `results_canonical/`. k-center shares the geometric-coreset family with
herding but participates as one of the controller's own candidates rather than an external
comparison row; likewise SemDeDup's near-duplicate rule is realized as one of the controller's
own coverage-family candidates, not a separate external row (results kept in full, only the
baseline-count bookkeeping moved); see [`docs/baseline_fidelity_ledger.md`](docs/baseline_fidelity_ledger.md).

The 9 rows above are the fixed-primary-run headline performance table. The public artifact inventory
contains 12 tasks in total: it additionally includes CIFAR-10, ImageNet-100, and the
five-domain text language-modeling lane. These extra protocols and their run coverage are
listed explicitly in [`results_canonical/README.md`](results_canonical/README.md); they are
not silently folded into the 9-row ranking claim.

The Full/NoSelect reference is available for 11 tasks. OmniSelect uses 30% of the data on
the five forecasting tasks and exceeds Full on all five; see
[`results_canonical/FULL_REFERENCE_COMPARISON.md`](results_canonical/FULL_REFERENCE_COMPARISON.md).

- **Baselines**: official implementations where available, otherwise faithful reimplementations
  of the published selection rules, fidelity-tiered per baseline in
  [`docs/baseline_fidelity_ledger.md`](docs/baseline_fidelity_ledger.md).
- **Coverage**: EL2N/GraNd/CCS are classification-error-pruning methods with no definition on
  regression (time-series MASE) targets, marked `--` there rather than run. This is a structural
  property of those methods, not a gap in OmniSelect's coverage, which spans all 6 modalities.
- **Reproducibility recipes** per table/figure: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Environment

Core import and cached-table rebuilding need only `pip install -e .`; running the
CPU test suite uses `pip install -e ".[dev]"`. The audited GPU stack is Python 3.12,
`torch 2.8.0+cu128`, driver 570-class (`environment/pip_freeze_server_vgpu.txt`,
`environment/constraints-cu128.txt`). Large data/artifacts ship as a versioned release
archive (URL pending, see [`docs/ARTIFACTS_INDEX.md`](docs/ARTIFACTS_INDEX.md)).

## License

MIT (code). Dataset licenses vary, see `docs/ARTIFACTS_INDEX.md`. Unverified-redistribution
sets are provided as download pointer + SHA only.
