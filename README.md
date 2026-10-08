# OmniSelect

[简体中文](README_zh.md)

OmniSelect selects a fixed-budget training subset for a downstream model. It scores every record
of a pool with three quality signals (authenticity, influence and coverage) and builds a portfolio
of candidate subsets from single-signal rules, fusions of the signals, cooperative candidates in
which each signal has a separate role, and published selection methods. The portfolio is frozen on
a construction split of clean validation data. A ranking split elects one candidate with the
task's own downstream learner, and a confirmation split checks the elected pair.

A cooperative candidate admits the records with the highest authenticity or learned cleanliness
score, removes outliers and near-duplicates in the coverage geometry, and spends the budget inside
the admitted set with herding or k-center. The learned cleanliness score is a classifier of clean
validation records against pool records. Every fit goes through a train-once cache keyed by the
subset, so each distinct subset is trained once, and the run record reports the cost of each
stage in full-data-training equivalents.

This repository contains the controller (`omniselect/`), the compared selection methods with
their fidelity labels (`benchmark/Methods/`), one track per data family (images, time series,
industrial process, tables and text, in `tracks/`), a per-cell driver that writes a complete run
record, and the queue templates of the paper's batches. Protocol `v2_2` is the protocol of the
paper.

## Installation

Use Python 3.10 or later.

```bash
git clone https://github.com/qzkinhit/OmniSelect-Benchmark.git
cd OmniSelect-Benchmark
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision          # CPU or Apple MPS wheels
python -m pip install -r requirements.txt        # the package with the track libraries and the test tools
python -m pytest omniselect/tests -m "not slow"
```

On a Linux machine with a CUDA 12.8 driver, install the pinned versions of the paper's GPU runs.

```bash
python -m pip install -e ".[train,arms,dev]" -c environment/constraints-cu128.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

## Five-minute smoke run

The Tennessee Eastman files are in the repository, and the ETT series are downloaded at a pinned
commit. The command below runs one small cell of the process track and one of the forecasting track
under protocol `v2_2` on a laptop CPU, writes their run records, recomputes macro-F1 from the
stored per-unit files and verifies every stored selection.

```bash
python data/fetch_data.py --only ett tep
TRACK_CELLS="process:tep21 timeseries:ETTh1" bash run.sh smoke
```

`bash run.sh smoke` without `TRACK_CELLS` adds CIFAR-100 on frozen CLIP, Electricity with
TabPFN-v2 and the text track with SmolLM2-135M. Their first run downloads CIFAR-100, CLIP
ViT-B/32, the TabPFN-v2 weights and SmolLM2-135M, and the text cell needs the text pool
(`python data/build_text_pool.py`). A cell whose model or dataset cannot be loaded is skipped with a
message. The following command runs one cell at full size.

```bash
bash run_omniselect/run_cell.sh process tep21 0 v2_2
```

## Reproducing the paper

The paper evaluates every task over multiple seeds, with the pool, the corruption draw, the validation
splits and the downstream-model initialization shared by every strategy of a run. The reported
multi-seed runs of each task are listed in `results/paper/seeds_reported.json`.
`run_omniselect/queues/` holds the corresponding commands.

| Queue | Protocol | Paper content | Record archive in `results/paper/records/` |
|---|---|---|---|
| `main_v2_3.txt` | `v2_2` | Table 3, main runs | `main_v2_3.tar.gz` |
| `ablation_v2_3.txt` | `v2` | OmniSelect-NC rows | `main_v2_3.tar.gz` |
| `main_adapt_v2_2.txt` | `v2_2`, standalone | EL2N, GraNd and CCS on forecasting and text | `main_adapt_v2_3.tar.gz` |
| `scale_up.txt` | `v2_2`, `v2`, standalone | IN-100 and Text-100k columns of Table 3 | `scale_up.tar.gz` |
| `robustness.txt` | `v2_2` | Figure 4(e)(f) | `robustness_v2_3f.tar.gz` |
| `signal_drop.txt` | `v2_2` | Signal removal in Figure 4(d) | `ablation_v2_3.tar.gz` |
| `validation_size.txt` | `v2_2` | Validation size in Figure 4(g) | `validation_size_v2_3.tar.gz` |

Every robustness condition and signal-removal comparison uses the reported runs of its task.
The 40% injection and clean-validation points reuse the main-table records. The compact
`paired_units.tar.gz` provides the 69 ranking-split cells used in Figure 3.

```bash
python data/fetch_data.py
python -m run_omniselect.run_queue run_omniselect/queues/main_v2_3.txt --gpus 0,1 --jobs-per-gpu 2 --threads 4
python -m run_omniselect.run_queue run_omniselect/queues/scale_up.txt --gpus 0,1 --jobs-per-gpu 1 --threads 8
```

IN-100 uses 256 px image arrays and 224 px crops. Text-100k uses the Adult-v2 pool with
20,000 records per domain and SmolLM2-360M. [data/README.md](data/README.md) gives the data
preparation commands. The text and native-image training runs use a GPU.

```bash
cd results/paper && bash rebuild.sh
```

`rebuild.sh` recomputes the 13-column main table, the statistics over the eleven benchmark tasks,
the two scale-up tasks, Figure 3, and the signal-removal, robustness, validation-noise and
validation-size results of Figure 4. It compares 16 outputs and the plotted fields of the combined
export with `results/paper/tables/`. [results/README.md](results/README.md) lists the records and outputs.

The torch-based k-means test in `omniselect/tests/test_torch_select.py` uses torch 2.11, the version
of the paper's GPU runs. [docs/REPRODUCING.md](docs/REPRODUCING.md) lists the protocol fields and
robustness options.

## Run record

Every cell writes one directory, and `decision.json` is written last.

```
results_and_logs/<batch>/<track>/<dataset>/<learner>/seed_<s>/
├── config.json         resolved configuration, command line, git sha, package versions, hardware
├── splits.json         pool, construction, ranking, confirmation and test ids with their sha256, corruption tags
├── signals/            one score per pool record for every signal
├── candidates/<name>/  selection.npz, per_unit/{con,rank,conf,test}.npz, scores.json
├── leaderboard.json    candidates in ranking order with their utilities on every split
├── decision.json       gate, elected candidate, audit statistics, precheck, screening trace
├── timings.json        seconds per stage, cache hits, full-data-training equivalents
├── metrics.json        the main-table rows of the track
└── log.txt
```

The per-unit files hold the predictions, targets and losses of each candidate on each split, so a
new metric is computed without training, and every selection is rebuilt and checked against its
hash.

```bash
python -m omniselect.store.recompute results_and_logs/smoke --metric balanced_accuracy --split test
python -m omniselect.store.replay results_and_logs/smoke
```

[docs/RUN_RECORD.md](docs/RUN_RECORD.md) describes every file and field.

## Adding a strategy or a data family

A strategy is a directory `benchmark/Methods/<Name>/` whose `method.py` registers a function
`fn(ctx, k) -> list[int]` with `omniselect.core.portfolio.registry.register` and whose README
states the source, the retained rule, the adaptation and the fidelity label. The function reads
the pool, the labels and the signals from the `SelectionContext` and returns k pool indices.
Import it in `benchmark/Methods/__init__.py` and add it to the membership tables in
`omniselect/core/portfolio/membership.py` to make it a portfolio member.

A data family is a subclass of `Track` in `tracks/<family>/` with a `TrackConfig` subclass for its
constants and the hooks `load` (data, corruption and splits), `signals` (the three signals and the
strategy inputs) and `learner` (an object with `fit(subset, stage)` and `score(model, split)` that
returns per-unit outputs). Register it in `TRACKS` of `tracks/common/experiment.py`.
[docs/REPRODUCING.md](docs/REPRODUCING.md) lists the optional hooks.

## Repository layout

| Directory | Content |
|---|---|
| [`omniselect/`](omniselect/README.md) | controller with configuration, signals, selection, adjudication, gates, portfolio and run-record store |
| `tracks/` | one track per data family and the per-cell driver `tracks/common/experiment.py` |
| `benchmark/` | one directory per selection method, standalone runners, dataset loaders |
| [`data/`](data/README.md) | dataset descriptions, the fetcher and the text pool builder |
| `run_omniselect/` | shell entry points, queue templates, the queue runner, CSV aggregation |
| `tools/` | gate replay, proxy ranking, run-record audit |
| `docs/` | design, code map, reproduction, run-record format, dataset provenance |
| [`results/`](results/README.md) | the paper's run records, tables and rebuild scripts |

[docs/CODE_MAP.md](docs/CODE_MAP.md) gives a reading order for the code, and
[docs/DESIGN.md](docs/DESIGN.md) explains the protocol choices.

## Citation

```bibtex
@misc{omniselect2026,
  title  = {{OmniSelect}: Task-Driven Adjudication of Data Selection Strategies for
            Modality-Specific Foundation Models},
  author = {Qian, Zekai and Ding, Xiaoou and Zhou, Muyun and Wang, Hongzhi and Wang, Chen},
  year   = {2026},
  note   = {Submitted to PVLDB}
}
```

## License

The code is released under the [MIT license](LICENSE). Datasets, model weights and the published
methods reimplemented here keep their own terms, listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
