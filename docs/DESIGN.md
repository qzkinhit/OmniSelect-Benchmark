# Protocol design

This page explains the protocol choices of OmniSelect and the configuration field behind each one.
Protocol `v2_2` is the protocol of the paper. Protocol `v2` is the same protocol without cooperative
candidates, and protocol `v2_1` adds a fourth fusion signal to `v2`. Both are ablations. Protocol
`canonical` is the earlier two-split controller and is kept for regression tests. Every constant
named below is a field of `OmniSelectConfig` in `omniselect/config/config.py`, so an ablation
changes one field with `--set key=value`.

## Three validation splits

The clean validation records of a task are divided into a construction split `V_con` (30%), a
ranking split `V_rank` (50%) and a confirmation split `V_conf` (20%) (`splits.mode=three_way`,
`splits.fractions`). The three splits have separate readers.

- `V_con` is read by the strategies that need validation data, by the influence reference, by the
  learned cleanliness score, by screening and by the synthesized challengers.
- `V_rank` is read once per frozen candidate and decides the election.
- `V_conf` is read by the confirmation audit only.

A split that selected a candidate cannot then give an unbiased estimate of that candidate's gain.
The separation keeps the ranking estimate free of construction bias and the audit estimate free of
ranking bias. On forecasting tasks the splits and the test zone are contiguous time blocks
separated by gaps of L + H steps (`splits.time_blocks`), so that no window of one zone overlaps a
window of another. The text track uses fixed split sizes (`splits.text_sizes`, 400, 600 and 500
records, with a 500-record report split).

## Frozen portfolio

The candidates come from four sources.

1. Reference-eligible strategies. These read the pool, its observed labels and pool-only signals.
   The membership table of each track lists them with a reason for every exclusion
   (`omniselect/core/portfolio/membership.py`, `portfolio.membership`).
2. Fusion-grid cells. Each cell blends the min-max normalized signals with one weight vector,
   applies an authenticity quantile prefilter and selects with the budget selector. Successive
   halving on `V_con` keeps `screening.sh_keep` (4) finalists.
3. Cooperative cells (protocol `v2_2`, described below), screened on `V_con` down to
   `cooperative.screen_keep` (3) finalists.
4. Synthesized challengers built on `V_con`: a consensus vote over the top candidates and a
   coordinate ascent over the fusion weights.

Every candidate that reads `V_con` is challenger-only. The candidate list is fixed before `V_rank`
is read, and no candidate is added or changed afterwards. The ranking split therefore compares a
fixed family, which is what the election rule and its guarantees assume.

## Margin election and confirmation audit

The election compares the best reference-eligible strategy on `V_rank` with the top
`gate.k_challengers` (3) challengers on `V_rank`. A challenger is adopted when its ranking utility
exceeds the reference's by more than `gate.margin_frac` (0.015) times the absolute ranking utility
of the reference (`gate.kind=margin`). Otherwise the reference is elected.

With `gate.audit` the controller then reads `V_conf` for the reference and each of the top three
challengers. It records the paired mean gain, a bootstrap 90% interval, the share of bootstrap
replicates with a positive mean, and the e-value of a betting e-process with K = 3 and
delta = 0.05. An adoption whose e-value reaches K / delta = 60 is marked certified. The audit is
written to `decision.json` and never changes the election.

The certificate is an audit and not the election rule because of its power at the deployed split
sizes. With 160 to 250 confirmation units and paired correctness differences, the e-process
reaches its threshold only for gains near 0.1 or larger, and on forecasting tasks with 2 to 22
confirmation blocks it cannot reach the threshold at all. Electing by certificate would keep the
reference on almost every task. The statistical gates remain available as options
(`gate.kind=lcb`, `bootstrap` or `eprocess`), and `tools/gate_replay.py` replays every gate from
the stored per-unit scores without training.

The headroom precheck (`precheck.enabled`) records the gap between full-data training and random
selection on `V_rank` and does not change the election (`precheck.decide=false`). Budgeted subsets
beat full-data training on several forecasting streams, so a full-versus-random gap does not bound
the gain of every strategy.

## Train-once cache

Every fit is keyed by the hash of the sorted subset ids and the learner's fidelity key
(`cache.by_subset_hash`). A subset is trained once, and its per-unit outputs on `V_con`, `V_rank`,
`V_conf` and the test split all come from that fit. Grid cells whose subsets coincide are merged
and listed as aliases of the first cell. Scoring fits use the reported training schedule and seed
(`fidelity.scoring_equals_reported`), so the fit that decides the election is the fit that is
reported. `timings.json` records the cost of each layer in full-data-training equivalents (FDE),
the sum over fits of the subset size over the pool size times the fit's schedule over the reported
schedule.

## Cooperative candidates and the learned cleanliness score

A weighted sum of the signals lets one signal compensate for another inside a single score. The
signals fail on different records. Authenticity (label agreement with neighbours and detectors)
removes label flips and duplicates but does not see feature corruption. Coverage operators such as
k-center select corrupted rows and windows, because a corrupted record lies far from the rest of
the pool. A cooperative candidate gives each signal a separate role
(`omniselect/core/selection/cooperative.py`).

1. Admission. Keep the ceil(rho k) records with the highest gate score, ties broken by pool index.
2. Outlier veto. Drop a record when the robust z-score of the distance to its 10th nearest pool
   neighbour exceeds 3 (`cooperative.knn`, `cooperative.outlier_z`).
3. Duplicate veto. Link records closer than 0.1 times the median nearest-neighbour distance and
   keep the cleanest member of each group (`cooperative.duplicate_ratio`).
4. Refill. When fewer than k records remain, return vetoed records in score order.
5. Selection. Herding or k-center selects k records inside the admissible set.

The gate score is the authenticity signal or the learned cleanliness score
(`omniselect/core/signals/cleanliness.py`). The cleanliness score is the cross-fitted probability of
a gradient-boosted classifier that separates `V_con` records (label 1) from pool records (label 0)
on per-record features computed against the pool (neighbour label agreement, cross-fitted label
probability and margin, neighbour distances, and family-specific statistics). `V_con` records are
clean and the pool is a mixture, so the classifier score orders pool records by their similarity to
clean data. The grid has two gate scores, rho in {1.15, 1.3, 1.5} and two inner selectors, plus two
ungated cells that apply the vetoes only, 14 cells in total (5 on text). The cell with the
authenticity gate, rho = 1.3, both vetoes and herding reads the pool only and is reference-eligible
(`coop_herding`). The top k records by cleanliness form one more challenger (`clean_top`).

## Stable tie-breaks

Every selection order sorts with a stable sort on the negated score, so records with equal scores
keep ascending pool order. Top-k rules, gates and greedy selectors therefore return the same
records under every numpy build and every platform. The run record stores the hash of each sorted
selection (`sel_sha12`), and `python -m omniselect.store.replay` checks it.

## Deterministic encoding

Image features of the vision track are encoded with the PIL image processor (`use_fast=False`) in
float32, with TF32 matrix multiplication and TF32 cuDNN convolutions disabled during encoding
(`benchmark/Data/vision.py`, `deterministic_encoding`). With TF32 enabled, the patch-embedding
convolution of CLIP moves a few percent of the CIFAR-100 feature rows below cosine 0.9999 of the
float32 features and changes kNN, k-means, herding and k-center selections. Before encoding on an
accelerator, `checked_device` encodes the eight images of
`benchmark/Data/fixtures/clip_vitb32_cifar100_ref8.npz` and compares them with the stored reference
rows. A row cosine below 0.9999 switches the encoding to the CPU with a logged warning. The encoder
settings and a hash of the feature arrays are written to `config.json` of every vision cell.
