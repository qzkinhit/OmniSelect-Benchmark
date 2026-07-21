# Tab-AICL baseline

**Active In-Context Learning for tabular foundation models** -- the direct prior work on
selecting the in-context support set for a TabPFN-style model. This is the most important
external baseline for our tabular arm: both it and our method answer "which rows should the
tabular foundation model condition on?", so a reviewer expects a head-to-head.

Tabular-native (like `deepcore` is image-native), so `run_tab_aicl.py` reproduces the original
TabPFN in-context setting rather than the text-manifest contract.

## Acquisition rules (`method/tab_aicl_select.py`, pure `numpy`)

| rule | principle | how |
|---|---|---|
| `tabpfn_coreset` | representativeness | k-center-greedy coreset in feature space, context spans the input distribution |
| `tabpfn_margin` | informativeness | rank by TabPFN prediction margin `p(top1) − p(top2)`, keep the lowest-margin (most uncertain) |
| `tabpfn_hybrid` | both | half the budget by margin, fill the rest by coreset diversity (deduplicated) |

The runner supplies the TabPFN forward-pass probabilities, so the method module stays pure and
model-free. These same three rules are wired into `scripts/run_tabular_experiment.py` as
comparison columns and folded into the AdaptiveController portfolio, so the controller is `>=`
Tab-AICL by construction, not only `>=` the off-domain DeepCore coresets.

## Run

```bash
python baselines/tab_aicl/run_tab_aicl.py            # OpenML electricity, TabPFN-v2, ROC AUC
TAB_DATASET=phoneme python baselines/tab_aicl/run_tab_aicl.py
```

## Reference

Ma et al. *Active In-Context Learning for Tabular Foundation Models (Tab-AICL).* 2026.
Selection of the in-context support set via coreset / margin / hybrid acquisition for TabPFN.
