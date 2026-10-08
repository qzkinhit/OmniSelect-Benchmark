# Code map

This page lists the files in the order a newcomer should read them. The first four steps cover
one benchmark cell from its configuration to its run record. The later steps cover the parts a
reader needs when changing a signal, a strategy, a data family or an analysis. Paths are relative
to the repository root.

## 1. Configuration

- `omniselect/config/config.py` defines `OmniSelectConfig`, one dataclass per controller concern
  (splits, cache, fidelity, gate, precheck, portfolio, influence, screening, synthesis, fixes,
  robustness, signals, coverage, cooperative, cleanliness, store), and the presets `canonical`,
  `v2`, `v2_1` and `v2_2`. Every constant of the controller is a field here with its default, and
  `preset(protocol, **overrides)` resolves one protocol with dotted-key overrides.
- `tracks/common/task.py` defines `TrackConfig` (pool, validation and test sizes, budget fraction,
  corruption fraction, grids, method rows, data root), `TaskData`, `Signals` and the `Track` base
  class with the hooks every data family implements.

## 2. The per-cell driver

- `tracks/common/experiment.py` runs one cell. It loads the task, computes the signals and the
  cooperative family, executes every portfolio strategy under the paired RNG, calls the controller
  with utility callbacks backed by the train-once cache, fits every candidate at the reported
  schedule and writes the run record. `main` is the command line (`python -m tracks.common.experiment`).

## 3. Signals (`omniselect/core/signals/`)

- `base.py` defines the `Signal` interface and `minmax`, which every channel passes through before fusion.
- `knn.py` computes kNN label agreement (authenticity on labelled features) and kNN novelty
  (redundancy) in row chunks.
- `authenticity_v2.py` combines out-of-fold label agreement, kNN agreement and a kNN inlier score
  by the minimum of their ranks, one detector per corruption mechanism.
- `authenticity.py` scores text records by completeness, non-degeneracy and structural validity.
- `influence.py` computes the probe log-likelihood of the observed label under a reference model
  fitted on the construction split, and the causal-LM loss on text.
- `redundancy.py` computes compression ratios and hashed character n-gram features on text.
- `alignment.py` computes the fourth channel of protocol `v2_1`, the cosine between each record's
  last-layer gradient and the mean construction-split gradient.
- `cleanliness.py` computes the learned cleanliness score of protocol `v2_2`, a cross-fitted
  gradient-boosted classifier of construction-split records against pool records.

## 4. Selection (`omniselect/core/selection/`)

- `budget_select.py` implements `BudgetSelector`, greedy importance plus a diversity bonus, the
  selector of every fusion cell.
- `fusion_grid.py` builds the fusion-grid cells (weighted channel sum, authenticity prefilter,
  budget selector) and parses cell names.
- `cooperative.py` builds the cooperative candidates of protocol `v2_2`, which admit records by a
  gate score, apply the outlier and duplicate vetoes and select inside with herding or k-center.
- `relaxation.py` solves the relaxed quadratic objective of InfoMax, used by the InfoMax baseline
  and the solver challengers of protocol `v2_1`.
- `similarity.py` defines the feature, gradient and concatenated similarity spaces of the
  diversity term (`coverage.space`).
- `console.py`, `budget.py`, `base.py` and `text_transfers.py` hold the signal blend of Fixed
  fusion, budget resolution, the selector interface and the token-budgeted text orders.

## 5. Adjudication (`omniselect/core/adjudication/`)

- `splits.py` divides the validation records into construction, ranking and confirmation splits,
  and forecasting time axes into contiguous zones.
- `cache.py` implements the train-once cache, one fit per distinct subset and fidelity key.
- `screening.py` runs successive halving on the construction split, at full or low fidelity.
- `synthesis.py` builds the consensus vote and the coordinate-ascent challengers.
- `precheck.py` records the headroom of full-data training and of the best reference over random selection.
- `controller.py` is Algorithm 1 (`adjudicate`). It builds and screens the candidates, freezes the
  list, ranks it on the ranking split, applies the gate and computes the confirmation audit.

## 6. Gates (`omniselect/core/gates/`)

- `margin.py` and `argmax.py` compare ranking utilities. `margin` is the deployed rule.
- `surrogate.py` maps per-unit outputs to bounded per-unit scores whose weighted mean equals the
  task metric or a bounded stand-in for it.
- `paired.py` holds a paired sample of differences with its stratified reading order and bootstrap.
- `eprocess.py`, `bootstrap.py` and `lcb.py` are the betting e-process, the paired bootstrap and
  the Hoeffding lower-confidence-bound gates.
- `audit.py` applies the statistical tests to a paired sample and computes the confirmation audit.
- `base.py` holds `GateResult` and the ordered-family runner shared by every gate.
- `paired_text_logloss.py` and `text_terminal_adapter.py` implement the gate of the canonical text protocol.

## 7. Portfolio (`omniselect/core/portfolio/`)

- `registry.py` maps strategy names to selection functions with their required inputs.
- `membership.py` lists the reference-eligible and challenger strategies of each track and
  protocol, with a reason for every exclusion.
- `roles.py` maps candidates to their roles in the run record.

## 8. Run record (`omniselect/store/`)

- `run_record.py` writes one directory per cell, every file atomically, `decision.json` last.
- `metrics.py` computes every metric from per-unit arrays.
- `recompute.py` evaluates a metric from stored per-unit files without training.
- `replay.py` rebuilds every stored selection and checks its hash.
- `index.py` appends one line per finished cell to the batch index.

## 9. Strategies (`benchmark/Methods/`)

Each directory holds one strategy family, with `method.py` (the pure selection function and its
registered adapter) and a README (source, retained rule, adaptation, fidelity label).

| Directory | Strategies |
|---|---|
| `Random/` | uniform random selection |
| `FullData/` | the whole pool (`full`) |
| `AuthenticityOnly/`, `InfluenceOnly/`, `AlignmentOnly/` | top k by one signal |
| `Coverage/`, `KCenter/`, `Herding/` | k-means medoids, greedy k-center, herding |
| `FixedFusion/` | fixed-weight fusion of the signals |
| `Cooperative/` | `coop_herding` and `clean_top` of protocol `v2_2` |
| `EL2N/`, `GraNd/`, `CCS/` | error-norm, gradient-norm and coverage-centric coreset selection |
| `Density/`, `SemDeDup/`, `D4/` | density sampling, semantic deduplication, D4 |
| `DsDm/`, `DMF/`, `QuaDMix/` | datamodel selection, multi-actor fusion, quality and mixture selection |
| `GLISTER/`, `GRADMATCH/`, `InfoMax/` | gradient-based selection and information maximization |
| `DSIR/`, `ZIP/`, `LESS/`, `IFMates/` | text selection methods |
| `TabAICL/` | TabPFN-based acquisition rules on the tabular track |
| `CleanOracle/` | diagnostic selection by the corruption tags |
| `RegMix/` | documented and not registered |

`_common.py`, `_gradients.py` and `_torch_select.py` hold shared helpers, the last-layer gradient
model of GLISTER and GRAD-MATCH, and GPU versions of herding, k-center and k-means coverage.
`benchmark/MethodsRunScript/` runs each strategy without the controller, `benchmark/Data/` holds
the dataset loaders, and `benchmark/tools/` holds the DeepCore reference implementations.

## 10. Data families (`tracks/<family>/`)

- `tracks/vision/clip_probe.py` runs CIFAR-100, CIFAR-100N and CIFAR-10 on frozen CLIP features
  with a logistic probe. `image_learners.py` adds the DINOv2 probe and ResNet-18 learners.
- `tracks/vision/native_resnet.py` trains ResNet-18 from scratch on CIFAR-10 and ImageNet-100.
- `tracks/timeseries/dlinear.py` runs ETTh1, ETTh2, ETTm1 and two DaISy streams with DLinear.
  `chronos.py` adds fine-tuned Chronos-bolt.
- `tracks/process/tep_mlp.py` runs Tennessee Eastman fault diagnosis with an MLP and alternatives.
- `tracks/tabular/tabpfn.py` runs OpenML Electricity with TabPFN-v2 in context.
- `tracks/text/smollm_finetune.py` fine-tunes SmolLM2 on the five-domain pool. `pool.py` splits
  the held-out records and `lm_eval.py` runs a downstream benchmark check.
- `tracks/common/` holds the parts shared by all families, namely corruption injection
  (`injection.py`), split helpers (`splits.py`), the learner protocol (`downstream.py`), the
  paired RNG (`pairing.py`) and the track side of the alignment channel (`alignment.py`).

## 11. Analysis tools and entry points

- `tools/gate_replay.py` replays every decision rule from stored per-unit scores.
- `tools/proxy_ranking.py` compares the candidate rankings of a cheap and an expensive learner.
- `tools/audit_records.py` checks finished run records for missing values, broken selections and outliers.
- `tools/dataset_io.py` and `tools/inject_errors.py` load one task and apply one corruption
  mechanism, for notebooks.
- `run_omniselect/generate_csv.py` aggregates run records into CSV files.
- `run_omniselect/signal_drop.py` runs the signal-removal comparisons from matching main records.
- `results/paper/rebuild.sh` rebuilds the published table and figure statistics.
- `run_omniselect/make_queues.py` and `run_omniselect/run_queue.py` write and run the queue files
  of the paper's batches. `run.sh`, `run_omniselect/run_cell.sh` and `run_omniselect/run_experiment.sh`
  are the shell entry points.
- `omniselect/api/` exposes `OmniSelect`, `AdjudicationTask` and `select_pool` for use outside the benchmark.

## 12. Tests

`omniselect/tests/` holds the CPU test suite. `test_driver_run_record.py` runs small cells end to
end and checks every file of the run record. `test_no_test_leakage_contract.py` checks that the
controller never requests test outputs before the election. `test_controller_legacy_equivalence.py`
and `test_protocol_canonical_regression.py` check that the canonical preset reproduces the earlier
controller.
