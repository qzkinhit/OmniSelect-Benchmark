# DeepCore coreset baselines

The standard **image data-selection / coreset** baselines collected by the DeepCore
benchmark, kept self-contained here as the recognized external comparison and folded
into our adaptive controller's portfolio (so the controller is `>=` each of them by
construction, not only `>=` our own symmetric arms).

> `kcenter` specifically is one of the controller's own portfolio candidates and is
> excluded from the paper's "vs 11 baseline" comparison table by definition (see
> `experiments/canonical_tables_seed0.json` -> `_meta.internal_only_excluded`). The
> other DeepCore baselines here (`herding`, `el2n`, `grand`, `ccs`) ARE part of the
> 11 external comparison baselines.

Unlike the other baselines in this folder (text, jsonl `->` manifest), DeepCore is
**image-native**, so its runner reproduces the original image-coreset domain directly
rather than the text-manifest contract.

## Methods (`method/coreset_select.py`, pure `numpy`)

| name | family | rule | reference |
|---|---|---|---|
| `herding` | geometric | greedily pull the selected-set mean toward the full-set mean | Welling, ICML 2009 |
| `kcenter` | geometric | farthest-point / greedy k-center coverage | Sener & Savarese, ICLR 2018 |
| `el2n` | score-based | keep top-k highest `‖softmax(logits) − onehot(y)‖₂` (hardest) | Paul et al., NeurIPS 2021 |
| `grand` | score-based | last-layer gradient-norm proxy `EL2N × ‖φ‖`, keep top-k | Paul et al., NeurIPS 2021 |

These same functions are wired into the four cross-modal runners
(`scripts/run_{vision,tabular,timeseries,tep}_experiment.py`) as comparison columns and
portfolio members. `tests/test_baseline_deepcore_consistency.py` asserts the standalone
copy here and the integrated `src/.../external_baselines.py` return identical selections.

## Run (faithfulness reproduction on a small CIFAR subset)

```bash
python baselines/deepcore/run_deepcore.py            # CIFAR-10, ~2.5k images, cached after the first run
DEEPCORE_DATASET=uoft-cs/cifar100 python baselines/deepcore/run_deepcore.py
```

It encodes a small CIFAR subset once with a frozen CLIP encoder, then for both a CLEAN
pool and a 40%-label-noise pool runs each method at a 30% budget, fits a linear probe on
the selection, and reports top-1 accuracy.

## Effect vs the original papers

- **Clean data.** The methods cluster around a strong random baseline and none consistently
  beats it. This matches DeepCore's own headline finding (Guo et al. 2022): random is a hard
  baseline and no coreset method dominates across settings, especially with already-strong
  features. k-center can dip below random because farthest-point picks outliers, a sensitivity
  the paper also notes.
- **Label noise.** EL2N and GraNd select the highest-error samples, which under injected label
  noise are exactly the mislabelled ones, so they crater far below random. This is the
  documented EL2N noise-sensitivity (Paul et al. 2021 and follow-ups) and is precisely the
  failure our adaptive controller avoids by letting each modality's own held-out validation
  vote, rather than committing to any one recognized signal.

## References

- Guo, Zhao, Bai. *DeepCore: A Comprehensive Library for Coreset Selection in Deep Learning.* DEXA 2022. https://github.com/PatrickZH/DeepCore
- Welling. *Herding Dynamical Weights to Learn.* ICML 2009.
- Sener, Savarese. *Active Learning for Convolutional Neural Networks: A Core-Set Approach.* ICLR 2018.
- Paul, Ganguli, Dziugaite. *Deep Learning on a Data Diet: Finding Important Examples Early in Training.* NeurIPS 2021.
