# OmniSelect versus the full-data reference

`Full`/`NoSelect` uses all available training examples or tokens and is therefore a
reference, **not an equal-budget data-selection baseline**. OmniSelect uses 30% of the
pool for forecasting/process tasks and 50% for image, tabular, and text tasks.

The table below is read from the frozen primary-run artifacts. “Relative change” is
direction-normalized: positive always favors OmniSelect. Accuracy, macro-F1, and AUC use
`(OmniSelect-Full)/abs(Full)`; MASE and PPL use `(Full-OmniSelect)/abs(Full)`.

| Task | Primary metric | Full | OmniSelect | Relative change | OmniSelect / Full budget |
|---|---:|---:|---:|---:|---:|
| CIFAR-100 | top-1 accuracy ↑ | 0.536 | 0.432 | -19.4% | 50% |
| CIFAR-100N | top-1 accuracy ↑ | 0.447 | 0.388 | -13.2% | 50% |
| CIFAR-10 (frozen-CLIP probe) | top-1 accuracy ↑ | 0.914 | 0.912 | -0.2% | 50% |
| ETTh1 | MASE ↓ | 1.003 | 0.968 | +3.6% | 30% |
| ETTm1 | MASE ↓ | 1.101 | 1.037 | +5.8% | 30% |
| ETTh2 | MASE ↓ | 0.652 | 0.619 | +5.0% | 30% |
| TEP21 | macro-F1 ↑ | 0.414 | 0.407 | -1.7% | 30% |
| Electricity | ROC AUC ↑ | 0.882 | 0.876 | -0.7% | 50% |
| DaISy-CSTR | MASE ↓ | 1.106 | 1.023 | +7.5% | 30% |
| DaISy-steamgen | MASE ↓ | 0.982 | 0.823 | +16.2% | 30% |
| Five-domain text | geometric-mean PPL ↓ | 10.976 | 11.058 | -0.7% | 50% token budget |

The full-data reference covers 10 of the 12 benchmark tasks; ImageNet-100 and the native
ResNet-18 CIFAR-10 instance have no comparable Full row. The CIFAR-10 row above is the
separate frozen-CLIP probe instance used in the paper's working-boundary analysis; it is
registered here as an artifact but is not one of the 12 benchmark tasks. OmniSelect exceeds
Full on all five forecasting tasks while using 30% of the training windows. Across the 10
covered benchmark tasks, the unweighted direction-normalized change is +0.2%, and the
average retained data/token budget is 38.0% (62.0% reduction); including the frozen-CLIP
row, the 11-row averages are +0.2% and 39.1% (60.9% reduction). These are descriptive
summaries; the task-standard raw metrics remain primary.

The machine-readable source, including artifact sources and the runner-recorded timing
scope, is [`experiments/full_reference_primary_run.json`](../experiments/full_reference_primary_run.json).
Wall-clock logs are hardware- and runner-specific and are not presented as a speedup claim.
