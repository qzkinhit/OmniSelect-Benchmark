# ZIP / Entropy-Law baseline

**ZIP** selects a low-redundancy data subset by *minimizing the compression ratio*
of the chosen set. The Entropy Law observes that a set which compresses well is
repetitive (low information), so a set that compresses *poorly* carries more
distinct information. With a lossless compressor `C` (here `zlib`) and byte size
`Bits(.)`, the set compression ratio is

```
g(D) = Bits(D) / Bits(C(D))
```

A **lower** `g` means the set is harder to compress, i.e. more informative / less
redundant. ZIP greedily builds the kept set to keep `g` as low as possible.

This is a model-free, CPU-only, dependency-free re-implementation (pure `zlib` +
stdlib; no torch/transformers/datasets) that operates on the standardized
`UnifiedRecord.text` field and plugs into the shared manifest contract.

## How it works (faithful to Algorithm 1)

The outer loop repeats three nested stages until `k` samples are selected:

1. **Stage 1 — global.** Score every remaining sample by its *own* sample-level
   ratio `g({d})` and take the **Bottom-K1** (lowest individual ratio = highest
   information density). This initializes the information-redundancy state and
   forms a coarse candidate pool. `K1 = max(1, round(k1_ratio * n_remaining))`.
2. **Stage 2 — local, coarse.** For each Stage-1 candidate compute the *merged*
   ratio `g(selected ∪ {d})` and keep the **Bottom-K2** with the smallest merged
   ratio (those that least inflate the kept set's compressibility).
   `K2 = max(1, round(k2_ratio * K1))`.
3. **Stage 3 — local, fine.** Greedily add Stage-2 candidates one at a time, each
   time recomputing the marginal merged ratio against the growing selected set and
   always taking the current argmin, until the K2 block is consumed or the budget
   `k` is met.

Ties on the ratio are broken deterministically by original index, so selection is
fully reproducible across runs/processes.

When `k == len(pool)` the full greedy *ordering* of the pool is returned, so the
caller can truncate it under a token budget.

## Run

```bash
python baselines/zip/run_zip.py --config configs/experiments/<exp>.yaml
# -> outputs/<exp_id>_zip/manifests/{manifest.json,selected.jsonl}
```

The budget (`k`) is resolved with `mmdataselect.budget.Budget` from the same
`select.budget` config block the main system uses, so ZIP and `mmdataselect`
select the same number of records and can be trained/evaluated identically.

### Optional config block

```yaml
zip:
  k1_ratio: 0.1   # Stage-1 pool size as a fraction of remaining candidates
  k2_ratio: 0.5   # Stage-2 block size as a fraction of the Stage-1 pool
  level: 6        # zlib compression level (deterministic)
```

(`seed` is read from `select.seed`, shared with the main system.)

## Reference

Yin, Wu, Wang, et al. *Entropy Law: The Story Behind Data Compression and LLM
Performance* (ZIP). https://github.com/USTC-StarTeam/ZIP
