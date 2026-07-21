# Adding a dataset or modality to OmniSelect

OmniSelect is a cross-modal selection benchmark. Plugging in a new dataset is a small,
contained change that reuses the shared signal library, the controller, and the equal-budget
evaluation harness. This doc is the full template behind `run_scripts/add_dataset.sh`.

## The one hard invariant

**Every baseline is passed into the controller as an executed candidate, never as a
compare-only column.** The controller's "never worse than the portfolio on validation"
guarantee (Proposition 2) holds *because* each baseline is a member of the frozen portfolio
that the controller adjudicates over. If you evaluate a baseline separately and only tabulate
its number next to the controller, the guarantee is broken and the comparison is no longer
apples-to-apples. Concretely, in the runner you call:

```python
controller.select(records, channel_scores, k, features=feats,
                   held_out_gain=held_out_gain,
                   extra_strategies=[(name, fn) for name, fn in baseline_candidates])
```

where each `fn(k) -> selected indices`. Copy the `mmds_adapt` block from any existing
`scripts/run_<arm>_experiment.py` verbatim and only swap the dataset/model specifics.

## Two ways to extend

### A. Add a first-class modality arm (own dataset, own downstream model)

Template: any `scripts/run_<arm>_experiment.py`. `run_scripts/add_dataset.sh <name> <arm>`
copies one for you.

1. **Loader + controlled noise.** Load your data into the pool, then inject a controlled
   fraction of tagged low-quality records so selection has something to do:
   `NOISE_FRAC = 0.40`, four tagged kinds appropriate to the modality
   (e.g. corrupt / duplicate / flat / shuffle), RNG `np.random.default_rng(seed + 7)`, one
   provenance tag per record. A pure-clean pool cannot measure selection value.
2. **Signals.** Reuse `src/mmdataselect/signals/` (authenticity, influence, redundancy). Only
   the influence signal is model-aware; authenticity and coverage are model-free and cross-modal.
3. **Downstream model + metric.** Set the downstream model (e.g. a linear probe, DLinear, an
   MLP, TabPFN) and the primary metric with its direction (higher- or lower-is-better).
4. **Controller wiring.** Pass every baseline as a candidate (the invariant above). To use the
   certified adjudication mode, gate it behind `METHOD_V3=1`; the empirical mode is the default.
5. **Emit results.** Write `outputs/<arm>/<dataset>/run_id=...-<sorted tags>/seed_<N>/results.json`
   with one row per method carrying the metric, `sel_sha12`, `train_order_sha12`, and dump the
   split manifest to `SPLIT_EXPORT_DIR`.
6. **Provenance.** Record source URL + pinned revision + SHA256 + the noise recipe in
   `docs/dataset_provenance.md` and `docs/ARTIFACTS_INDEX.md`.
7. **Tests.** Add a smoke test to `tests/`.

### B. Add a domain to the text quality-variance pool

Template: `scripts/add_timeseries_modality.py`; contract: `tools/standardize/_common.py`.

1. Write a standardizer that streams an HF dataset and maps each example to a
   `UnifiedRecord(id, modality, domain, text, meta={quality, noise})` via
   `tools.standardize._common.standardize_stream` (lazy `datasets` import; exit code 3 = skip
   offline).
2. Emit high-quality rows plus the four tagged noise kinds at `LOW_FRAC ≈ 0.4`, appending to
   `data/processed/qpool_{train,heldout}.jsonl`.
3. Delete the stale influence cache `data/processed/qpool_influence.npz` so it recomputes.
4. Run `scripts/run_experiment.py` (or `scripts/run_permodality_sweep.sh` with
   `ONLY_DOMAIN=<new>`).

## Add an external baseline

Follow `baselines/README.md` §5: create `baselines/<name>/{method/ (pure, IO-free selection),
run_<name>.py (symmetric runner → manifest), README.md}`, pass it into the controller as an
`extra_strategies` candidate, add a smoke test plus a T3 mechanism test to
`tests/test_baseline_fidelity.py` (the mandatory anti-regression gate).

## Checklist before you open a PR

- [ ] Every baseline runs as a controller candidate (not a compare-only column).
- [ ] `results.json` carries `sel_sha12` + `train_order_sha12`; split manifest dumped.
- [ ] Source + pinned revision + SHA256 + noise recipe recorded in provenance.
- [ ] A smoke test passes on CPU.
- [ ] `pytest -q` is green.
