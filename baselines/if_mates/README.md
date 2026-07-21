# IF / MATES baseline

**Influence-driven data selection** — keep the examples whose presence most helps
the downstream model, scored by a per-sample *influence* signal and selected by
Top-K.

This follows two faithful threads:

- **Influence functions** (Koh & Liang, ICML 2017) estimate the effect of a single
  training point on the model via a first-order (gradient / Hessian-vector) proxy;
  examples with the largest positive influence on the target objective are kept.
- **MATES** (Yu et al., NeurIPS 2024; [`cxcscmu/MATES`](https://github.com/cxcscmu/MATES))
  trains a small *data influence model* to predict each example's per-sample
  influence on the reference loss, then selects the Top-K scoring examples —
  model-aware data selection.

Both reduce to the same recipe: compute a per-sample influence score, then take the
Top-K. This baseline keeps that logic pure and modality-agnostic, operating on the
standardized `UnifiedRecord.text` field and plugging into the shared manifest
contract.

## How it works

1. **Influence score** — each record gets a per-sample influence value from
   `mmdataselect.signals.InfluenceSignal`, which uses the downstream base model's
   own per-sample loss / learnability as the influence proxy.
   - Torch / transformers are **lazy-imported** inside the signal. When the
     `influence_model` is unset or the torch path is unavailable, the signal
     transparently falls back to a deterministic CPU proxy — so importing or running
     this baseline never requires torch.
2. **Top-K** — `method/if_select.py: select` ranks records by descending influence
   and keeps the budget-resolved `k`. Ties keep their original pool order (stable
   sort), so selection is deterministic for a fixed input.

### Reusing precomputed influence

`select(records, k, *, influence=...)` accepts a precomputed per-sample influence
array (e.g. gradient-aligned scores or MATES data-influence-model outputs produced
by an experiment). When supplied, selection is a pure Top-K over those values and no
model is loaded. When `k == len(records)`, `select` returns the **full descending
ranking**, so a token-budget caller can truncate it afterwards.

## Run

```bash
python baselines/if_mates/run_if_mates.py --config configs/experiments/<exp>.yaml
# -> outputs/<exp_id>_if_mates/manifests/{manifest.json,selected.jsonl}
```

The budget (`k`) is resolved with `mmdataselect.budget.Budget` from the same
`select.budget` config block the main system uses, so IF/MATES and `mmdataselect`
select the same number of records and can be trained/evaluated identically.

### Config

The runner reads the downstream model id from `select.influence_model` and passes it
straight to `InfluenceSignal`:

```yaml
select:
  budget:
    kind: fraction   # fraction | records | tokens
    value: 0.5
  seed: 0
  influence_model: null   # e.g. "gpt2"; null/unavailable -> deterministic CPU proxy
```

(`seed` is read from `select.seed`, shared with the main system. Influence Top-K is
deterministic; `seed` is kept only for a uniform selector signature.)

## References

- Koh, Liang. *Understanding Black-box Predictions via Influence Functions*. ICML
  2017. https://arxiv.org/abs/1703.04730
- Yu, Das, Xiong, et al. *MATES: Model-Aware Data Selection for Pretraining with Data
  Influence Models*. NeurIPS 2024. `cxcscmu/MATES`,
  https://github.com/cxcscmu/MATES
