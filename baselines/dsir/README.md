# DSIR baseline

**Data Selection with Importance Resampling** — select raw examples whose feature
distribution matches a *target* distribution, by resampling proportional to an
estimated importance weight `w_i = p(x_i) / q(x_i)`.

This is a CPU-light, dependency-free re-implementation (pure `numpy` + stdlib; no
torch/transformers/datasets) that operates on the standardized `UnifiedRecord.text`
field and plugs into the shared manifest contract.

## How it works

1. **Features** — hashed word n-gram (1..`ngram`) bag-of-features via a stable
   `crc32` hash (`method/dsir_select.py: hashed_ngram_counts`), so selection is
   reproducible across runs/processes.
2. **Distributions** — fit a Laplace-smoothed multinomial `q` over the whole raw
   pool (proposal) and `p` over the **target** rows. The target is the `math` +
   `code` domain records; if a pool has none, it falls back to the full pool.
3. **Importance weight** — each example's log-weight is the per-feature log-ratio
   `log(p_f) - log(q_f)` accumulated over its own n-gram counts:
   `log w_i = sum_f c_{i,f} (log p_f - log q_f)`.
4. **Resampling** — Gumbel top-k on the log-weights = exact sampling-without-
   replacement proportional to `w_i` (DSIR importance resampling). Set
   `dsir.noise: 0` for deterministic importance top-k.

## Run

```bash
python baselines/dsir/run_dsir.py --config configs/experiments/<exp>.yaml
# -> outputs/<exp_id>_dsir/manifests/{manifest.json,selected.jsonl}
```

The budget (`k`) is resolved with `mmdataselect.budget.Budget` from the same
`select.budget` config block the main system uses, so DSIR and `mmdataselect`
select the same number of records and can be trained/evaluated identically.

### Optional config block

```yaml
dsir:
  dim: 4096        # hashed feature dimension
  ngram: 2         # use 1..ngram word grams
  smoothing: 1.0   # Laplace smoothing on the multinomials
  noise: 1.0       # Gumbel scale; 0 = deterministic importance top-k
```

(`seed` is read from `select.seed`, shared with the main system.)

## Reference

Xie, Santurkar, Ma, Liang. *Data Selection for Language Models via Importance
Resampling (DSIR)*. NeurIPS 2023. https://arxiv.org/abs/2302.03169
