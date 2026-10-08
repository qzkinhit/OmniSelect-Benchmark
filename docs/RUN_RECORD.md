# Run record

Every cell writes one directory. A cell is one (batch, track, dataset, learner, seed, protocol). The layout is shown below.

```
results_and_logs/<batch>/<track>/<dataset>/<learner>/seed_<s>/
├── config.json
├── splits.json
├── signals/            authenticity.npy influence.npy redundancy.npy [extras] reference_sample_ids.json
├── candidates/<name>/  selection.npz  per_unit/{con,rank,conf,test}.npz  [per_unit/scoring_{con,rank}.npz]  scores.json  [model/]
├── leaderboard.json
├── decision.json       written last, marks the cell complete
├── timings.json
├── metrics.json
└── log.txt
```

The writer is `omniselect/store/run_record.py`. Every file is written through a temporary file and
`os.replace`. JSON files refuse NaN and infinity. The cell appends one line to
`results_and_logs/<batch>/index.jsonl` after `decision.json` exists. Nothing else is written
outside the cell directory. The queue and the driver skip a cell whose `decision.json` exists.

The published packages use `main` for main records and `main_nc` for NC records. These directory
aliases retain the original `config.json` batch and Git revision. Machine paths are replaced by
`<repo>` for code, `<data>` for inputs and `<host>` for the producing host. Figure 3 uses the
compact ranking arrays in `paired_units.tar.gz`, including the source array checksums.

## config.json

| Field | Content |
|---|---|
| `schema_version` | `omniselect.run-record.v1` |
| `omniselect` | the resolved `OmniSelectConfig` (protocol preset, track overrides, CLI `--set`) |
| `track` | the resolved `TrackConfig` of the track (pool, validation and test sizes, budget, learner schedule, CLI `--track-set`) |
| `cli` | the command line of the driver |
| `git` | `sha` of HEAD and `dirty` (tracked files differ from HEAD) |
| `environment` | Python, platform, hostname, versions of numpy, scipy, scikit-learn, torch, transformers, datasets, tabpfn, chronos-forecasting, xgboost, lm-eval, the accelerator (CUDA device name and `CUDA_VISIBLE_DEVICES`, or MPS), and `threads`: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` (raw strings or null), `torch_num_threads`, `cpu_count` and `load_avg_1min` at the start of the cell (records before this field lack it) |
| `start_time`, `end_time`, `elapsed_secs` | wall clock of the cell |
| `protocol`, `batch` | copies for quick filtering |
| `encoding` | vision tracks: the image encoder settings stored with the feature cache (encoder and revision, device, processor class, `use_fast`, dtype, TF32 and cuDNN flags, torch and transformers versions, the 8-image fixture check) and `features_sha256` over Xp, Xval and Xt. `settings` is null for a feature cache written before the encoder settings were recorded. The same object is in `splits.json` under `provenance.encoding` |

## splits.json

| Field | Content |
|---|---|
| `ids.pool` | pool record ids in pool order. `idx` in every `selection.npz` indexes this list |
| `ids.con`, `ids.rank`, `ids.conf` | validation ids of the construction, ranking and confirmation splits (empty lists for unused splits) |
| `ids.test` | test ids |
| `sha256` | sha256 of the JSON form of each id list |
| `counts` | length of each list |
| `pool_tags` | per pool record: `high` or the corruption tag (`flip`, `dup`, `hard`, `corrupt`, `flat`, `shuffle`. Text `high` or `low`) |
| `pool_mechanisms` | per pool record: the corruption mechanism (text: `truncation`, `template`, `crossdomain`, `lowtier`. Unseen mechanisms: `blur`, `jpeg`, `class_flip`, `scale_drift`, `bias_drift`, `stuck`, `stuck_feature`) |
| `time_blocks` | forecasting under v2: start-position zones of con, rank, conf and test, the gap (L + H), window counts per zone and the pool zone. `null` otherwise |
| `budget` | records selected (text: tokens summed over domain budgets) |
| `utility` | task utility: `accuracy`, `macro_f1`, `auc`, `neg_mase` or `neg_gmean_ppl` |
| `s0` | forecasting: mean absolute error of the last-value forecast on the pool windows (the fixed scale of the bounded surrogate) |
| `provenance` | dataset revision, cache paths, source files and checksums of the track |

Ids are namespaced where the source has several splits (`train:<i>`, `test:<i>`), window start
positions for forecasting, OpenML row numbers for Electricity, record ids for text.

## signals/

`authenticity.npy`, `influence.npy`, `redundancy.npy`: one float per pool record, the raw channel
scores before min-max normalization. Protocol v2.1 adds `alignment.npy` (raw cosine in [-1, 1]) and
`timings.json` (`seconds` of the three original channels and of the alignment channel, and
`alignment_source`). The native track adds `el2n.npy` and `grand.npy`, the text
track adds `transfer_features.npy` (frozen-LM embeddings). Protocol v2.2 adds `cleanliness.npy` (the
learned cleanliness score in [0, 1], cross-fitted over pool folds) and a `cleanliness` object in
`timings.json`: `features` (the per-record feature names), `cv_auc` (cross-validated AUC of V_con
against the pool, null with fewer V_con records than folds), `n_pool`, `n_con`, `seconds` and
`neighbour_secs` (the pool neighbour graph shared with the cooperative vetoes). `reference_sample_ids.json` holds the
ids of the influence reference sample and its `source`: `pool_clean_tag` (canonical: clean-tagged
pool records), `v_con` (v2: records of the construction split), `heldout_reference_half` (canonical
text) or `pool_uniform_sample` (native).

## candidates/NAME/

The directory name is the candidate name with unsafe characters replaced by `_`.
`leaderboard.json` maps names to directories.

`selection.npz`

| Key | Content |
|---|---|
| `idx` | int64, sorted pool indices of the selection |
| `n`, `budget` | pool size and number selected |
| `sel_sha12` | first 12 hex digits of sha256 of `str(sorted(idx))` |
| `selection_secs` | seconds the strategy took (0 for controller-synthesized candidates) |
| `role` | `reference` (reference-eligible), `challenger` (grid cell, consensus, coordinate ascent, text Fixed fusion under canonical), `screened_out` (grid cell removed by successive halving), `baseline_row` (a method row outside the portfolio, and `full`), `diagnostic` (a method row that reads the corruption tags, the clean oracle) |
| `stage` | `reference`, `finalist`, `screened_out`, `synthesized`, `solver` (v2.1 solver challenger of a finalist), `challenger_member`, `method_row`, `full` |
| `name` | candidate name |
| `weights` | optional per-record weights (no current strategy writes them) |

`per_unit/<split>.npz` holds the outputs of the fit at the reported schedule on each non-empty
split (`con`, `rank`, `conf`, `test`). The keys depend on the family as listed below.

| Family | Keys | Metrics recomputable from them |
|---|---|---|
| classification (vision, native, process, tabular) | `target`, `prediction`, `correct`, `proba` (float32, one column per class in `classes` order, when the learner has `predict_proba`), `classes` | accuracy, macro-F1, balanced accuracy, per-class recall, ROC AUC (binary or one-vs-rest), expected calibration error |
| forecasting | `prediction` and `target` (units x H), `last_value` (last input value), `start` (window start) | MASE with the last-value scale, MAE over the pool scale `s0`, per-window errors, block means for the certificate |
| text | `nll` (mean token NLL of each record), `n_tokens`, `domain`, `record_id` | per-domain perplexity, geometric-mean perplexity, token NLL sums (`nll * n_tokens`) |

When the scoring fit of a stage differs from the reported fit (canonical: CLIP probe 150 against
300 iterations, DLinear 40 against 60 epochs and stage seeds), the outputs that the controller
used are kept as `per_unit/scoring_con.npz` and `per_unit/scoring_rank.npz`.

`scores.json`: `name`, `role`, `stage`, `sel_sha12`, `n_selected`, `metrics` (per split the
standard scalar metrics of the family, computed at write time), `utility_name`, `utility` (per
split), `test_utility`, `purity` (share of `high` records in the selection), `fit_secs`,
`score_secs` (per split), `fidelity` (schedule and seed of the reported fit), `scoring` (`u_con`
and `u_rank` as the controller saw them, and `aliases` of grid cells merged into this one),
`selection_secs`, `selection_info` (strategy-specific measurements, for LESS the stage seconds and the
seconds per pool and per target record) when the strategy reports them, and `model_path` when a model
was saved.

Cooperative candidates of protocol v2.2 (`coop_herding` and the grid cells `coop gate=<score> rho=<rho>
within=<selector>`, `coop rho=none within=<selector>`) record in `selection_info`: `rule`
(`cooperative`), `gate_score` (`authenticity` or `cleanliness`, null without a gate), `rho` (null
without a gate), `selector`, `pool`, `admissible_gate` (records kept by the gate), `outlier_drops` and
`duplicate_drops` (admissible records removed by each veto), `refill` (removed records returned
because fewer than k remained, on text the records of the budget cut outside the vetoed admissible
set), `admissible_size` (records the selector saw), the veto constants and pool statistics (`metric`,
`knn`, `outlier_z`, `outlier_median`, `outlier_scale`, `pool_outliers`, `duplicate_ratio`,
`duplicate_threshold`, `pool_duplicate_groups`, `pool_duplicate_records`), `seconds` (gate, vetoes and
selector of this candidate), `neighbour_secs` (the shared neighbour graph, computed once per cell) and
`admissible_purity` (share of the admissible set tagged `high`, computed from `pool_tags` after every
selection for analysis, never read by a rule). Grid cells have role `challenger` and stage `finalist`
or `screened_out` like fusion-grid cells, and a cell whose subset equals an earlier cell's is listed
in that cell's `scoring.aliases`.

`model/`: the fitted model when `save_models` is `elected` (the elected candidate) or `all`.

Every npz file is written with `np.savez_compressed` (all commits so far). `store.per_unit_scope`
(default `all`) decides which candidates keep `per_unit/`. With `finalists` a grid cell removed by
screening (stage `screened_out`) keeps `selection.npz` and `scores.json`, whose `metrics` and
`utility` were computed from the per-unit arrays at run time, and `scores.json` records
`per_unit_written: false`. `recompute` reports such a candidate as scalar only and skips it, and
`replay` reads only `selection.npz`. Reference-eligible candidates, finalists, synthesized
challengers, method rows and `full` always keep their per-unit files, so `tools/gate_replay.py`
has every file it reads.

## leaderboard.json

`utility`. `candidate_dirs` (name to directory). `rows`, one per controller candidate sorted by
the ranking utility the controller used, each with `name`, `role`, `stage`, `sel_sha12`, `n`,
`purity`, `u_<split>` of the reported fit, `scoring_u_con`, `scoring_u_rank` and `test_metrics`.
`screened_out` lists the names removed by screening. `membership`, the portfolio table of the track with a reason for every
excluded strategy. `skipped_strategies`, strategies that could not run with the reason.

## decision.json

| Field | Content |
|---|---|
| `gate.kind` | `margin`, `argmax`, `lcb`, `bootstrap`, `eprocess`, or `precheck` when the precheck elected random |
| `gate.reference`, `gate.elected`, `gate.adopted`, `gate.k_tested` | the reference, the elected candidate, whether a challenger was adopted, and K |
| `gate.tests` | one entry per tested challenger with its statistics: margin `tau`, `difference`. Lcb `mean`, `radius`, `lcb`, `delta`, `K`, `sum_w2`, `n_units`. Bootstrap `mean`, `p_win`, `threshold`, `n_boot`, `p_beat_min`, `K`, `seed`, `n_units`, `boot_q05`. Eprocess `e_value` (capped at exp(690)), `log_e_value`, `threshold` (K / delta), `mean`, `eps`, `delta`, `K`, `seed`, `n_units`, `n_read` (length of the reading sequence), `stop_index` (units read when the wealth first reached the threshold, null if never), `final_log_wealth`, `max_stake`, `reading` (uniform, class, domain or auc_pairs). Bootstrap also records `boot_q95` and `resampling`. Margin, argmax, bootstrap and eprocess list every challenger of the family (the top `k_challengers` by the pair split), lcb stops at the first adopted one |
| `elected`, `elected_sel_sha12`, `elected_test_utility` | the election and its test utility at the reported schedule |
| `reference` | the gate's reference. `rank_best_reference` the best reference on the ranking split |
| `overall_rank_best`, `switched`, `kappa_hat`, `u_rank_elected` | ranking-split summary (kappa_hat is the best ranking utility minus random's) |
| `precheck` | `enabled`, `skipped`, `headroom`, `r1`, `threshold`, `n_eff`, `delta`, `multiplier` |
| `audit` | v2 (`gate.audit`): `split`, `reference` (best ranking reference), `K`, `delta`, `eps`, `n_boot`, `reading_order`, `seed`, and `entries`, one per top-K ranking challenger with `mean_gain`, `boot_interval90`, `p_win`, `e_value`, `log_e_value`, `threshold`, `certified` (e >= K / delta), `stop_index`, `n_units`, `n_read`, `reading`. Absent in records written before this field existed |
| `screening` | successive-halving pool sizes, evaluations, per-round construction utilities, screened-out names. Low fidelity adds `fidelity`, `subsample`, `finalists_full_u_con`, `kendall_tau_finalists`, `kendall_tau_all_cells`. Protocol v2.2 adds `cooperative`, the same fields for the cooperative grid (`cooperative.screen_keep` finalists) |
| `construction_errors` | optional construction stages that failed, with the error |
| `protocol`, `utility` | copies |

A standalone method run (`--standalone`) writes `{"controller": "off", "elected": null,
"method_rows": [...]}` plus the protocol fields.

## timings.json

`stages`: seconds of `load`, `signals`, `member_selection` (all strategies),
`controller_grid`, `controller_screening`, `controller_construction` (construction utilities and
synthesis), `controller_ranking`, `controller_confirmation`, `final_fit` (reported fits), `total`.
Protocol v2.2 adds `cooperative_signals` (neighbour graph and cleanliness score), `cooperative_grid`
(selections of the cooperative grid) and `controller_cooperative_screening`.
`cache`: per fit stage (`con`, `member`, `rank`, `conf`, `report`) the calls, fits, cache hits,
fit and score seconds and records trained. `fde`: full-data-training equivalents, the sum over
fits of records over pool size times the stage schedule over the reported schedule, by stage and by
layer (construction, ranking, confirmation, report). TabPFN fits count 0. Screen-stage fits of
low-fidelity screening count in the construction layer, and `screening` (low fidelity only) lists
the low-fidelity fits and FDE against the avoided full-fidelity fits of screened-out cells. `candidates`: selection,
fit and score seconds per candidate.

## metrics.json

`rows`: one entry per method row of the track's `methods` (the main-table rows), plus `full` and
`mmds_adapt` (the elected candidate, with `picked`). Each entry has the fields of a leaderboard row.

## Tools

```bash
python -m omniselect.store.replay <dir>                    # verify every selection.npz under <dir>
python -m omniselect.store.replay <cell> --ids random      # pool ids of one candidate's subset
python -m omniselect.store.recompute <dir> --metric macro_f1 --split test --check
python -m tools.gate_replay <batch> --out <dir>             # every gate from stored per-unit files, one CSV per batch
python -m run_omniselect.generate_csv --batch <dir> --out <dir>/summary
```

`gate_replay` writes `gates_<batch>.csv` (cell, gate, reference, elected, adopted, certified,
units read, K, unit split, elected test utility, random's test utility, gain over random,
per-challenger statistics as JSON) and `gates_summary_<batch>.{csv,md}` (per gate, including the margin rule replayed at margin_frac
0, 0.005, 0.015, 0.03 and 0.05 as gates `margin@<value>`, the adoptions,
certified adoptions, adoption rate, mean gain over random and mean gain relative to
|u_test(random)|, and per task and gate the mean elected test utility, adoptions, certified
adoptions and the median units read at the threshold). An adoption is certified when the
e-value of its pair with K = `gate.k_challengers` reaches K / delta on the confirmation split. It
reads records of every schema version so far (gate constants missing from an older `config.json`
take the defaults of `GateConfig`, the reading order defaults to stratified).

`recompute --check` compares each recomputed value with the value stored in `scores.json` at write
time and exits 1 on a difference above `--tol` (default 1e-9). `replay` exits 1 when an index file
is unsorted, out of range, or its sha differs from the stored one or from `decision.json`.
