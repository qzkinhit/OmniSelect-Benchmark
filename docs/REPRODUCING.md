# Reproduction

Run every command from the repository root after activating the Python environment of the
[README](../README.md). `pyproject.toml` declares the dependencies, and
`environment/constraints-cu128.txt` pins the package versions of the GPU runs behind the paper.
Protocol `v2_2` is the protocol of the paper.

## Commands

```bash
bash run.sh setup                                                  # install, CPU tests, data checksums
bash run.sh smoke                                                  # one small cell per track under v2_2
bash run_omniselect/run_cell.sh process tep21 0 v2_2               # one cell at full size
bash run_omniselect/run_experiment.sh local "timeseries:ETTh1,process:tep21" 0,1,2 v2_2
python -m run_omniselect.run_queue run_omniselect/queues/main_v2_3.txt --gpus 0,1   # a paper batch
python -m tracks.common.experiment --help                          # every driver flag
```

`smoke` runs vision (CIFAR-100, frozen CLIP), forecasting (ETTh1, DLinear), process (TEP21, MLP),
tabular (Electricity, TabPFN-v2) and text (SmolLM2-135M) cells at small sizes, then checks that
`python -m omniselect.store.recompute --metric macro_f1 --check` reproduces the stored values from
the per-unit files and that `python -m omniselect.store.replay` verifies every selection. A cell
whose model or dataset cannot be loaded is skipped with a message (exit code 3). The first vision
run downloads CIFAR-100 and CLIP ViT-B/32 and caches the features. `PROTOCOLS="v2 v2_2" bash run.sh
smoke` runs several protocols, and `TRACK_CELLS="vision:cifar100"` restricts the tracks.

The driver takes `--track`, `--dataset`, `--seed`, `--protocol` (`canonical`, `v2`, `v2_1` or
`v2_2`), `--learner`, `--batch`, `--out`, `--methods` (main-table rows), repeated `--set key=value`
for `OmniSelectConfig` fields (for example `--set gate.kind=eprocess`), repeated `--track-set
key=value` for `TrackConfig` fields (for example `--track-set pool_n=2000`), `--smoke`,
`--standalone` (method rows only, no controller) and `--overwrite`.

| Track | Datasets | Learners (first is default) | Utility | Code |
|---|---|---|---|---|
| vision | cifar100, cifar100n, cifar10_clip | clip_vitb32 (frozen CLIP ViT-B/32, logistic probe), dinov2_vits14, resnet18_scratch | accuracy | `tracks/vision/clip_probe.py` |
| native | cifar10, imagenet100 | resnet18_scratch | accuracy | `tracks/vision/native_resnet.py` |
| timeseries | ETTh1, ETTm1, ETTh2, daisy_cstr, daisy_steamgen | dlinear, chronos_tiny, chronos_small | negative MASE | `tracks/timeseries/dlinear.py`, `chronos.py` |
| process | tep21 | mlp, rf, cnn1d, svm, knn, pca | macro-F1 | `tracks/process/tep_mlp.py` |
| tabular | electricity | tabpfn, xgboost, rf | ROC AUC | `tracks/tabular/tabpfn.py` |
| text | five_domain | smollm2_135m, smollm2_360m | negative geometric-mean perplexity | `tracks/text/smollm_finetune.py` |

Image features are encoded with the PIL image processor (`use_fast=False`), in float32, with
`torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32` set to false during
the encoding and restored afterwards (`benchmark/Data/vision.py`, `deterministic_encoding`). With
cuDNN TF32 enabled, the patch-embedding convolution of CLIP on an NVIDIA RTX A6000 moved about 4%
of the CIFAR-100 feature rows below cosine 0.9999 of the float32 features and changed selections.
Before encoding on an accelerator, `checked_device` encodes the eight images of
`benchmark/Data/fixtures/clip_vitb32_cifar100_ref8.npz` and compares them with the stored rows
(CLIP) or with the CPU encoding (other encoders). A row cosine below 0.9999 switches the encoding
to the CPU with a logged warning. `omniselect/tests/test_clip_encoding_reference.py` checks the
fixture on the CPU, and on CUDA and MPS when present.

## Protocols

`OmniSelectConfig.preset(protocol)` returns one preset, and every protocol difference is one
field, so an ablation changes one field with `--set`. [DESIGN.md](DESIGN.md) explains the choices.

| Field | canonical | v2 |
|---|---|---|
| `splits.mode`, `splits.fractions` | `two_way` (construction and ranking halves of a permutation with seed + 41) | `three_way` 0.3, 0.5, 0.2 (the ranking split keeps the canonical size) |
| `splits.time_blocks` | off (random validation and test zones) | forecasting splits are contiguous zones con, rank, conf, test with L + H gaps |
| `cache.by_subset_hash` | off (every utility call fits) | one fit per distinct subset and fidelity, grid cells with identical subsets merged |
| `fidelity.scoring_equals_reported` | off (CLIP probe 150 against 300 iterations, DLinear 40 against 60 epochs, stage seeds) | every stage uses the reported schedule and the final-fit seed |
| `gate.kind` | `margin`, 1.5% of the reference's ranking utility | `margin` on the ranking split with `k_challengers` 3 |
| `gate.audit` | off | on. Paired gain, bootstrap interval and e-value of the top 3 challengers on the confirmation split, recorded without changing the election |
| `precheck.enabled` | off | on. The headroom of full-data training and of the best reference over random selection on the ranking split is recorded (`precheck.decide` false) |
| `portfolio.membership` | `canonical` (the list of each earlier runner) | `unified` (`omniselect/core/portfolio/membership.py`) |
| `influence.reference` | clean-tagged pool records | records of the construction split |
| `fixes.fixed_fusion_gate` | `sentinel` (gated records get -1e9 before min-max normalization) | `finite_subset` (the selector runs on the passing records only) |
| `fixes.vision_flip_range` | `canonical_100` (flipped vision labels drawn from 100 classes on every dataset) | `n_classes` (drawn from the dataset's own classes) |
| `fixes.density_seed` | `zero` (Density used seed 0 on every run seed) | `run_seed` |

Protocols `v2_1` and `v2_2` start from `v2`.

| Field | v2 | v2_1 | v2_2 |
|---|---|---|---|
| `signals.alignment` | off | on (fourth channel `alignment`) | off |
| `synthesis.infomax_solver` | off | on (one challenger `<cell> solver=infomax` per screening finalist) | off |
| `cooperative.enabled` | off | off | on (cooperative challenger grid and learned cleanliness score) |
| `portfolio.membership` | `unified` | `unified_v21` (plus `alignment_only`) | `unified_v22` (plus `coop_herding` as a reference and `clean_top` as a challenger) |

Track overrides apply on top of the preset. The native and text tracks train each subset once and
run no coordinate ascent. Under `canonical` the native track elects the validation argmax
(`splits.mode=rank_only`, `gate.kind=argmax`), and the text track freezes the reference and
challenger pair on the construction split and applies the Hoeffding bound on the ranking split.
Under the `v2` family both tracks use the preset gate.

### Protocol v2_2

A cooperative candidate has three steps (`omniselect/core/selection/cooperative.py`). The gate
keeps the ceil(rho k) records with the highest gate score, ties broken by pool index (on text, the
records of the score order inside rho times each domain's token budget). Two vetoes act in the
selector's feature space (cosine distance on L2-normalized CLIP, frozen-LM and ResNet-18 features,
Euclidean distance on the L2-normalized windows of forecasting and on the standardized rows of
TEP21 and Electricity). A record is dropped when the robust z-score (d - median) / (1.4826 MAD) of
the distance to its 10th nearest pool neighbour exceeds 3, and of each group of records linked at a
distance below 0.1 times the median nearest-neighbour distance only the cleanest member stays.
Dropped records return in score order when fewer than k remain. Herding (the rule of
`benchmark/Methods/Herding`) or k-center (the rule of `benchmark/Methods/KCenter`) selects k
records inside. On text these are the token orders of `herding_text` and `coverage_text`.

The gate score is the track's authenticity signal or the learned cleanliness score
(`omniselect/core/signals/cleanliness.py`). The cleanliness score is the cross-fitted probability
of a gradient-boosted classifier that separates construction-split records (label 1) from pool
records (label 0) on per-record features computed against the pool. On classification tracks the
features are kNN label agreement, the cross-fitted probability and margin of the observed label
under a logistic regression, and the log distances to the nearest and the 10th nearest pool
record. On forecasting they are the cross-fitted ridge residual, the two distances, the window
standard deviation and the lag-1 autocorrelation. On text they are the text authenticity, the
compression ratio, the token length and the two distances. The score reads the construction
split, so the candidates that use it are challengers. `signals/cleanliness.npy` holds the score,
and `signals/timings.json` holds the feature list, the cross-validated AUC of construction-split
records against the pool and the seconds.

The grid is `coop_scores` (authenticity, cleanliness) x `coop_rhos` (1.15, 1.3, 1.5) x
`coop_selectors` (herding, kcenter) plus the ungated cells `coop_ungated` (herding, kcenter), 14
cells named `coop gate=<score> rho=<rho> within=<selector>` and `coop rho=none within=<selector>`.
Text runs 5 cells (rho 1.3 and the ungated herding cell). Cells with identical subsets are merged,
and successive halving on the construction split keeps `cooperative.screen_keep` (3) finalists,
which enter the ranked list after the fusion-grid finalists. `coop_herding` (authenticity gate,
rho 1.3, both vetoes, herding) reads the pool only and is reference-eligible. `clean_top` is the
top k by cleanliness. Under `v2_2` the native track embeds the validation records in its score runs.

```bash
python -m tracks.common.experiment --track process --dataset tep21 --smoke --protocol v2_2
```

### Protocol v2_1

The alignment channel is the cosine of each pool record's last-layer gradient g_i = phi_i e_i^T at
the warm-start parameters with the mean construction-split gradient G_V, computed as
phi_i^T G_V e_i divided by the product of the norms of phi_i, e_i and G_V, without forming the
d x m gradients (`omniselect/core/signals/alignment.py`). It equals the round-0 gain of GLISTER
divided by the two norms. The last layer is a logistic head on the CLIP features (vision), the
output layer of the MLP (128, 64) warm-started on the pool (TEP21), DLinear warm-started on the
pool (forecasting), a logistic head on TabPFN embeddings of the pool and the construction split
(Electricity), and the output layer of the last score-run ResNet-18 on its penultimate features
(native). On text the channel is the LESS score, computed once per cell and shared with the LESS
member. `signals/alignment.npy` holds the raw cosines.

With the channel every weight vector (a, f, c) of the fusion grid becomes (a, f, c, 0), and the
vectors (0, 0, 0, 1), (a/2, f/2, c/2, 1/2) for each vector and (1/4, 1/4, 1/4, 1/4) are added,
with q and lambda unchanged. A solver challenger solves s^T x - alpha x^T K x for the cell's fused
importance with the kNN kernel of the same similarity and beta = lambda
(`omniselect/core/selection/relaxation.py`). `coverage.space=gradient` replaces the feature
similarity of the diversity term by cos(phi_i, phi_j) cos(e_i, e_j) of the alignment layer (the
LESS sketches on text), and `concat` by the mean of both.

```bash
python -m tracks.common.experiment --track process --dataset tep21 --smoke --protocol v2_1
python -m tracks.common.experiment --track process --dataset tep21 --smoke --protocol v2_1 --set coverage.space=gradient
```

## Gates

`gate.kind` selects the decision rule (`omniselect/core/gates/`). `margin` and `argmax` compare the
ranking utilities of the best reference and the best challenger. `lcb`, `bootstrap` and `eprocess`
read per-unit paired differences d_i in [-1, 1] (challenger minus reference, bounded surrogate of
`core/gates/surrogate.py`) on `gate.split` for the reference and the top `gate.k_challengers`
challengers ordered on `gate.pair_split`, and elect the first challenger that passes. The deployed
gate is `margin`. The other rules are options that elect by their own verdict.

| Gate | Rule | Fields |
|---|---|---|
| `lcb` | weighted mean minus sqrt(2 log(K / delta) sum w_i^2) is positive | `delta` |
| `bootstrap` | weighted mean positive, and the share of `n_boot` bootstrap replicates with a positive mean at least 1 - (1 - `p_beat_min`) / K | `n_boot` 1000, `p_beat_min` 0.9 |
| `eprocess` | the supremum of the betting wealth reaches K / delta and the weighted mean is positive | `delta` 0.05, `eps` 0 |

The e-process bets on z_t + eps with the aGRAPA stake of the earlier values, clipped to
[0, 1 / (2 (1 - eps))]. `gate.reading_order` fixes the sequence z_t and the bootstrap replicates
(`core/gates/paired.py`).

| Surrogate | `stratified` (default) | `uniform` |
|---|---|---|
| balanced accuracy (TEP21 macro-F1) | draw a class uniformly, read the next unused unit of that class. The bootstrap resamples within classes | one shuffle of the units, plain resampling |
| AUC (Electricity) | pair the next unused positive with the next unused negative and read the kernel difference. The bootstrap resamples positives and negatives separately | one shuffle of the per-positive U-statistics |
| text | draw a domain uniformly, read the next unused record. The bootstrap resamples within domains | one shuffle, plain resampling |
| accuracy, forecasting blocks | one shuffle (blocks of `ts_block_steps`, L + H by default) | the same |

A draw from an exhausted class or domain ends the sequence. The shuffles use the run seed.
`stop_index` is the number of units read when the wealth first reached K / delta.

With `gate.audit` the controller computes, for the best ranking reference and each of the top
`k_challengers` ranking challengers on `gate.audit_split` (conf), the paired mean gain, the
bootstrap 90% interval, the win share, the e-value with K = `k_challengers`, `certified`
(e >= K / delta) and `stop_index`, and writes them to `decision.json` under `audit`. The election
does not read the audit.

```bash
python -m tracks.common.experiment --track process --dataset tep21 --smoke --set gate.kind=eprocess --set gate.split=conf
python -m tracks.common.experiment --track process --dataset tep21 --smoke --set gate.kind=bootstrap --set gate.split=conf
python -m tools.gate_replay results_and_logs/local --out results_and_logs/summary/gates   # every gate, no training
```

## Robustness options

The `robustness.*` fields and `--learner` select the robustness conditions
(`tracks/common/injection.py`). `ROBUSTNESS_SUPPORT` in `tracks/common/experiment.py` lists which
track reads which setting, and a setting that a track does not read stops the cell before it
starts, with the reason.

| Setting | Values | Tracks |
|---|---|---|
| `robustness.mechanism_set` | `canonical`, `unseen` | vision (injected datasets), timeseries, process, tabular |
| `robustness.val_noise_kind`, `robustness.val_noise_rate` | `none`, `symmetric` (rate), `natural`, `prior_shift` (rate) | symmetric and prior_shift on vision, process, tabular. natural on vision |
| `robustness.injection_ratio` | pool corruption fraction. A negative value keeps the track default 0.4 | vision, timeseries, process, tabular |
| `--learner` | see the track table above | per track |

The unseen mechanisms corrupt the records that the canonical injector picks (default_rng(seed + 7),
a round(ratio n) share of the pool), split evenly over the mechanisms of the family.

| Family | Unseen mechanisms | Tags |
|---|---|---|
| vision | Gaussian blur with standard deviation 1.5 pixels, JPEG at quality 20, both re-encoded by the track's encoders. Class-dependent flip to the next fine class of the same CIFAR-100 superclass (CIFAR-10 uses a fixed asymmetric map) | `blur`, `jpeg`, `class_flip` |
| forecasting | on the recorded segment, multiplicative drift from 1 to 1.3, additive drift from 0 to 1 window standard deviation, stuck sensor from a position in [L/2, L) | `scale_drift`, `bias_drift`, `stuck` |
| process, tabular | stuck features (a quarter of the columns set to the pool median), class-dependent flip (TEP21 c to c + 1 mod 22, Electricity positives to the negative class) | `stuck_feature`, `class_flip` |

Validation contamination acts on the validation labels before the split into construction,
ranking and confirmation, so the influence reference, the ranking and the confirmation all read
the contaminated labels. `natural` replaces the validation labels by the human labels of the same
CIFAR images (CIFAR-100N `noisy_label`, CIFAR-10N `worse_label`). `prior_shift` makes a seeded half
of the classes the minority and replaces each minority record with probability `val_noise_rate` by
a uniformly drawn majority record. `splits.json` records the kind, the rate and the number of
changed labels under `provenance.val_noise`.

A learner swap changes the downstream model only. The signals and the strategies read the same
inputs. `dinov2_vits14` is the logistic probe on DINOv2 ViT-S/14 features encoded with the same
settings as CLIP. `resnet18_scratch` on the vision track trains a CIFAR-stem ResNet-18 from one
seeded initialization on the selected images (60 epochs, SGD 0.05, momentum 0.9, cosine schedule).
`chronos_tiny` and `chronos_small` fine-tune Chronos-bolt with gradient-norm clipping 1.0 and stop
on the construction-split quantile loss. `smollm2_360m` fine-tunes SmolLM2-360M on the same token
streams.

```bash
python -m tracks.common.experiment --track vision --dataset cifar100 --smoke --set robustness.mechanism_set=unseen
python -m tracks.common.experiment --track process --dataset tep21 --smoke --set robustness.val_noise_kind=prior_shift --set robustness.val_noise_rate=0.5
python -m tracks.common.experiment --track timeseries --dataset ETTh1 --smoke --learner chronos_small
```

## Selection on a device

`--track-set selection_device=cuda` (or `mps`, or `torch_cpu`) runs herding, k-center greedy and
k-means coverage with the torch versions of `benchmark/Methods/_torch_select.py` in float32 with
TF32 disabled, in row chunks. The default `cpu` keeps the numpy and scikit-learn code. Herding and
k-center follow the numpy rules step for step (direct norms, the same start record, first index on
ties) and select the same records. Coverage runs Lloyd's k-means with the k-means++ initialization
and the tolerance rule of scikit-learn. On the frozen-feature tracks this reproduces the
scikit-learn labels, while float32 centre sums can move a medoid at a near-tie. The native track
clusters with MiniBatchKMeans on the CPU path, and the device path replaces it by full-batch Lloyd
k-means, which is a different algorithm. The native lines of the paper's queue templates use the
device path. `selection_info` in each candidate's `scores.json` records the device and the seconds.

## Further options

`--set synthesis.held_half=true` splits the construction split into two halves (stratified by
class or domain, earlier and later windows on forecasting). Coordinate ascent starts from the grid
cell with the best utility on the first half, proposes each step that raises the first half's
utility, and accepts it only if the second half's utility does not decrease. The halves are scored
from the construction fits through the cache, so no training is added.

`--set screening.kind=low_fidelity` runs successive halving of the fusion grid on a seeded
subsample (`screening.low_fidelity_fraction`, 0.4) of each cell's selection with the short schedule
of the `screen` stage, then scores the finalists at full fidelity through the fit cache.
`decision.json` then records the Kendall tau between the low-fidelity and the full-fidelity
orderings, and `timings.json` records the fits saved.

## Protocol boundaries

Strategies read the pool, its observed labels and the signals. DMF and DsDm also read the
construction utility. The controller reads the construction and ranking utilities and, for the
statistical gates, the audit and the precheck, per-unit scores of the ranking and confirmation
splits. Test outputs are requested only by the reported fits after the election is fixed
(`omniselect/tests/test_no_test_leakage_contract.py` checks the order of learner requests).
Corruption tags are read by the injector, by the canonical influence reference, by the clean
oracle (a diagnostic row) and by the purity column of the run record.

## The paper's batches

`run_omniselect/queues/` contains the commands for the reported runs of each task or condition.
`results/paper/seeds_reported.json` lists the reported seeds.

| Queue | Protocol | Role in the paper |
|---|---|---|
| `main_v2_3.txt` | `v2_2` | Main table |
| `ablation_v2_3.txt` | `v2` | OmniSelect-NC |
| `main_adapt_v2_2.txt` | `v2_2`, standalone | EL2N, GraNd and CCS on forecasting and text |
| `scale_up.txt` | `v2_2`, `v2`, standalone | IN-100 and Text-100k |
| `robustness.txt` | `v2_2` | Figure 4(e)(f), with main-table seeds |
| `signal_drop.txt` | `v2_2` | Figure 4(d) |
| `validation_size.txt` | `v2_2` | Figure 4(g) |

```bash
python -m run_omniselect.run_queue run_omniselect/queues/main_v2_3.txt --gpus 0,1 --jobs-per-gpu 2 --threads 4
python -m run_omniselect.generate_csv --batch results_and_logs/main --out results_and_logs/summary/main
```

`generate_csv` writes `cells.csv` (one row per cell), `rows.csv` (one row per cell and method) and
`summary.csv` (means by task and method). `results/paper/rebuild.sh` reconstructs the reported
statistics directly from the published archives. The two scale-up columns are reported separately
from the eleven-task aggregate statistics. [../results/README.md](../results/README.md) lists the
outputs. [../run_omniselect/queues/README.md](../run_omniselect/queues/README.md) describes the commands.

## Protocol canonical

`canonical` reproduces the earlier two-split controller of the 2026-07 runners.
`omniselect/tests/test_controller_legacy_equivalence.py` compares `adjudicate` under this preset
with a frozen copy of the earlier controller, and `omniselect/tests/test_protocol_canonical_regression.py`
pins the selections and the election of a small TEP21 cell. The membership modes
`canonical_20260716` and `canonical_ettm1_20260716` replay two earlier portfolio generations for
such comparisons. The earlier code state is commit `cd7a6c7` of this repository's history.

## Outputs

Each cell writes the run record of [RUN_RECORD.md](RUN_RECORD.md) and one line of
`results_and_logs/<batch>/index.jsonl`. Exit codes of the driver are 0 for a finished or already
complete cell, 3 for a cell skipped because a model or dataset is unavailable (no `decision.json`
is written), and 1 for an error (traceback in the log). A skipped or failed cell is not a zero
score.

## Environment

The controller, the tracks and the driver read no configuration from the environment. Every
setting is a field of `OmniSelectConfig` or `TrackConfig`, or a driver flag. `ZIP_INCREMENTAL=0`
switches `benchmark/Methods/ZIP/method.py` to the non-incremental compressor of its equivalence
test. `run_record.py` stores `CUDA_VISIBLE_DEVICES` and the thread variables in `config.json`
without reading them as settings. The shell wrappers accept `PY`, `OUT`, `BATCH`, `PROTOCOLS`,
`TRACK_CELLS` and `SEEDS` and pass them on as flags. Library variables (`HF_HOME`,
`HF_HUB_OFFLINE`, `TABPFN_MODEL_CACHE_DIR`, `SCIKIT_LEARN_DATA`, `CUDA_VISIBLE_DEVICES`) keep their
usual meaning. The data root defaults to `data/` and is changed with `--track-set data_root=<dir>`.

## Adding a task or a strategy

A data family is a `Track` subclass (`tracks/common/task.py`) with a `TrackConfig` subclass for its
constants, a `datasets` table, and the hooks `load` (data, corruption, splits), `signals`
(authenticity, influence, redundancy and the strategy inputs), `learner` (an object with
`fit(subset, stage)` and `score(model, split)`), and optionally `strategy_overrides`, `alignment`
and `cooperative_inputs`. Register the class in `TRACKS` of `tracks/common/experiment.py` and add
the track to the membership tables of `omniselect/core/portfolio/membership.py`.

A strategy is a directory under `benchmark/Methods/` whose `method.py` registers a function
`fn(ctx, k) -> list[int]` with `omniselect.core.portfolio.registry.register`, and whose README
states the source, the retained rule, the adaptation and the fidelity label. Import it in
`benchmark/Methods/__init__.py`, add it to the membership tables to make it a portfolio member,
and add a runner under `benchmark/MethodsRunScript/` to run it on its own.
