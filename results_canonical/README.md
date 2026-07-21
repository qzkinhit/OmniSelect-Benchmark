# Canonical result map

This directory contains the small JSON artifacts needed to inspect every reported task.
All `results.json` files are machine-readable; paths preserve dataset, protocol, and seed.
The benchmark has **6 modality/task families, 12 datasets/tasks, 11 external baseline
rows, plus OmniSelect**. Not every baseline is mathematically applicable to every task;
unsupported cells are marked `N/A` in the coverage ledger rather than silently replaced.
The current release contains **42 parseable `results.json` files**, including six added
text full-portfolio/control artifacts.

The complete fixed-primary-run comparison with the Full/NoSelect reference is in
[`FULL_REFERENCE_COMPARISON.md`](FULL_REFERENCE_COMPARISON.md), with its machine-readable
source in [`experiments/full_reference_primary_run.json`](../experiments/full_reference_primary_run.json).

## Datasets and available seeds

| Modality / task family | Dataset / task | Result path | Public run coverage | Scope |
|---|---|---|---|---|
| image | CIFAR-100 | `vision/uoft-cs_cifar100/run_id=pubcore-*/` | three frozen runs | unified CLIP protocol |
| image | CIFAR-100N | `vision/uoft-cs_cifar100/run_id=backfill-cifar100n-*/` | one frozen run | real human-noise labels |
| image | CIFAR-10 | `vision/uoft-cs_cifar10/` and `vision/cifar10_full/` | one CLIP run; three original-protocol runs | two explicitly separated protocols |
| image | ImageNet-100 | `vision/imagenet100/` | three frozen runs | original ResNet-18 protocol |
| time series | ETTh1 | `timeseries/ETTh1/` | three frozen runs | replacement controller artifact retained beside its superseded source run |
| time series | ETTm1 | `timeseries/ETTm1/` | three frozen runs | DLinear |
| time series | ETTh2 | `timeseries/ETTh2/` | three frozen runs | DLinear |
| process diagnosis | TEP21 | `tep/tep21/` | three frozen runs | MLP, macro-F1 |
| tabular | Electricity | `tabular/electricity/` | three frozen runs | TabPFN-v2, ROC AUC |
| process time series | DaISy CSTR | `timeseries/daisy_cstr/` | three frozen runs | DLinear, MASE |
| process time series | DaISy steamgen | `timeseries/daisy_steamgen/` | three frozen runs | DLinear, MASE |
| text | five-domain pool | `experiment/run_id=text-qz4-*/` and `text/` | three frozen runs | SmolLM2-135M, five-domain gmean PPL + lm-eval |

The single-run CIFAR-100N and CIFAR-10 CLIP backfills are intentionally labeled as such;
their presence must not be mistaken for three-seed evidence.

## Method rows

The paper-facing rows are Random, Coverage/Coreset, Herding, EL2N, GraNd, CCS,
Density, QuaDMix-pub, DMF-pub, Influence-only, Fixed-fusion, and OmniSelect.

- EL2N/GraNd/CCS require supervised classification error or gradient scores and are not
  defined for regression/time-series targets.
- The frozen text protocol directly contains Random, QuaDMix-pub, Fixed-fusion and
  OmniSelect; `if_mates` is the text-native Influence-only implementation and `dmf` is
  explicitly a local dynamic-fusion proxy, not a DMF-pub original-protocol reproduction.
- Coverage/Coreset, Herding, EL2N, GraNd, CCS and Density have no frozen text adapter in
  this result release. They are `N/A` for that column, with reasons recorded in the paper
  and coverage ledger.

## Text artifacts

Three complementary folders are public:

1. `experiment/run_id=text-qz4-*/`: three QZ4 runs for `quadmix_pub`, `zip`, and the
   reduced-portfolio controller row. Only the first two are canonical text baseline rows.
2. `text/five_domain_full_portfolio/`: three frozen runs for NoSelect, Random, DSIR,
   IF/MATES, quality-PPL, local DMF proxy, Balance, Fixed-fusion (`mmdataselect`),
   PerpCorr, and OmniSelect (`mmds_adapt`).
3. `text/frozen_controls/`: three frozen runs for Base, Random and NoSelect under the frozen
   control lane.

The full-portfolio lane and QZ4 use the same 25,000-record pool, SmolLM2-135M, fixed
token budget, per-domain held-out PPL and lm-eval tasks, but they are not one single code
state. Do not combine them into a paired significance test. Historical text split/noise/
initialization fingerprints were not captured at runtime; the text evidence remains
`PASS_WEAK`, and this repository does not fabricate missing sidecars.

## Read a file

```python
import json
from pathlib import Path

p = next(Path("results_canonical").glob("**/results.json"))
d = json.loads(p.read_text())
print(d.get("dataset"), d.get("seed"), [r["method"] for r in d["results"]])
```

Use `run_scripts/reproduce_cached.sh` to rebuild the paper-facing fixed-primary-run table.
The larger run logs and model/score caches are intentionally excluded from Git; see
[`docs/ARTIFACTS_INDEX.md`](../docs/ARTIFACTS_INDEX.md).
