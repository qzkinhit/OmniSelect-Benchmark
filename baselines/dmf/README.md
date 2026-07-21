# DMF baseline

> This `dmf` (proxy/base variant) is one of the controller's own portfolio
> candidates and is excluded from the paper's "vs 11 baseline" comparison table
> by definition, not by outcome (see `experiments/canonical_tables_seed0.json` ->
> `_meta.internal_only_excluded`). The published-update-transfer variant
> `dmf_pub` IS one of the 11 external comparison baselines and remains in every
> headline table.

**Dynamic Multi-Signal Fusion (base variant)** — fuse several heterogeneous
utility signals into one per-record score with *adaptive, feedback-driven*
weights, instead of a fixed hand-set blend. This is the representative
multi-signal dynamic-fusion comparison our system aims to surpass.

This baseline reuses the project's signals and dynamic-fusion controller but keeps
only the **basic dynamic weighting** active: a single learnable weight vector over
the signals, softmax-blended, optionally nudged by one feedback update. Every
advanced fusion mechanism is deliberately **off** — no sample-level conflict
gating, no curriculum prior, no group-wise (domain/modality) weights, and no
authenticity/truthfulness front-gate — so DMF isolates plain dynamic multi-signal
fusion from those additions.

It is CPU-runnable end to end: `torch`/`transformers` are reached only lazily inside
the influence signal, which degrades to a deterministic CPU proxy when they are
missing, so importing this baseline never requires torch.

## How it works

1. **Signals** — two modality-agnostic per-record utilities over
   `UnifiedRecord.text`: a `redundancy` (information-density) signal and an
   `influence` (downstream learnability) signal. Each is min-max normalized to
   `[0,1]` so the two are comparable.
2. **Dynamic fusion** — blend the signals with the dynamic-fusion controller using
   default (closed) upgrade parameters, i.e. a softmax-weighted dynamic blend
   `importance_i = sum_s w_s * score_{s,i}`.
3. **One feedback update (optional)** — when a small held-out probe is configured
   (`dmf.holdout_frac > 0`), apply a single dynamic-weight update from each
   signal's proxy gain on that probe (black-box, no per-sample gradients), so the
   weights adapt to which signal actually helps before final scoring. With no
   hold-out the update is skipped (pure base fusion).
4. **Top-K** — keep the highest-importance records. With `dmf.diversity: true`
   (default) the ranking is augmented by a lightweight greedy coverage term
   (`importance + lam * marginal_coverage`) over hashed char n-gram features; set
   `dmf.diversity: false` for plain importance Top-K. When `k == len(pool)` the
   method returns a full ranking, so a token-budget caller can truncate it.

## Run

```bash
python baselines/dmf/run_dmf.py --config configs/experiments/<exp>.yaml
# -> outputs/<exp_id>_dmf/manifests/{manifest.json,selected.jsonl}
```

The budget (`k`) is resolved with `mmdataselect.budget.Budget` from the same
`select.budget` config block the main system uses, so DMF and `mmdataselect` select
the same number of records and can be trained/evaluated identically.

### Optional config block

```yaml
dmf:
  lr: 0.5            # learning rate for the single dynamic-weight update
  diversity: true    # greedy coverage augmentation; false = plain importance top-k
  lam: 0.5           # diversity weight (used when diversity=true)
  holdout_frac: 0.0  # >0 carves a probe for one feedback update; 0 = pure base fusion
  influence_model:   # optional downstream model id; empty -> CPU proxy (no torch)
```

(`seed` and `influence_model` also default to the shared `select.*` block.)

## Reference

Representative line of work on **dynamic / adaptive multi-signal data-selection
fusion** — combining heterogeneous data-quality signals (e.g. information density /
redundancy and downstream influence / learnability) with feedback-adapted weights
rather than a fixed blend. Implemented here as a base variant for comparison.
