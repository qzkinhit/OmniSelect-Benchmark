# Baseline fidelity

Each method lives in `benchmark/Methods/<Name>/` with a README that states its source, what the
implementation transfers, and its fidelity tier. Portfolio membership per track, with the reason
for every exclusion, is in `omniselect/core/portfolio/membership.py`.

This page is the public comparison contract. Every reported row uses the same task-specific
candidate pool, split, budget, downstream learner and metric. A method is called a reproduction of
an original protocol only when that full protocol is executed. Otherwise the transfer is named.

## Fidelity tiers of the reported methods

| Method | Public source | Implementation | Published rule retained | Benchmark adaptation | Fidelity tier |
|---|---|---|---|---|---|
| Random | Standard control | `benchmark/Methods/Random/` | Uniform sampling without replacement | None | negative control |
| Influence-only | Operator-family control | `benchmark/Methods/InfluenceOnly/` (text: `IFMates/`) | Rank by task-model loss contribution | Perplexity on text | operator baseline |
| Coverage | Operator-family control | `benchmark/Methods/Coverage/` | Representation-space coverage | Task encoder supplies the representation | operator baseline |
| Fixed fusion | Controller ablation | `benchmark/Methods/FixedFusion/` | Fixed combination of the quality channels | Same channels as OmniSelect | controller ablation |
| Herding (ICML 2009) | Welling, herding dynamical weights | `benchmark/Methods/Herding/` | Greedy mean matching | Deep representation replaces the kernel feature map | local implementation |
| EL2N (NeurIPS 2021) | Data Diet | `benchmark/Methods/EL2N/` | Early-training error score | Only where supervised class probabilities exist | protocol transfer |
| GraNd (NeurIPS 2021) | Data Diet | `benchmark/Methods/GraNd/` | Expected gradient contribution ranking | Last-layer gradient-norm proxy | disclosed proxy |
| CCS (ICLR 2023) | Coverage-centric coreset selection | `benchmark/Methods/CCS/` | Budget allocation across score strata | Task classifier supplies the score | local implementation |
| Density (ICLR 2026) | Ask-LLM density sampler | `benchmark/Methods/Density/` | Inverse-density sampling | k-nearest-neighbor estimator replaces the LSH-KDE estimator | local implementation |
| QuaDMix (arXiv 2025) | Quality and domain-mixture sampler | `benchmark/Methods/QuaDMix/` | Domain-conditioned sigmoid quality sampler | Fixed budget, no proxy-model parameter search | published-core transfer |
| DMF (ACL 2025) | Multi-actor weighting | `benchmark/Methods/DMF/` | Centered additive actor-weight update | Actors are the three quality signals | published-core transfer |
| OmniSelect | This repository | `omniselect/` | Validation-guided adjudication over a frozen portfolio | Task adapters expose comparable validation evidence | proposed method |

The repository also implements GLISTER, GRAD-MATCH, InfoMax, LESS and a clean-tag oracle. They are
exercised by the tests but are not part of the tables the paper reports.

## Invariants

- No row claims a strict reproduction of an original protocol.
- EL2N, GraNd and CCS stay unavailable where their classification signals are undefined, and they
  are not replaced by unrelated task-specific proxies.
- Candidate operators used only inside the controller are not counted as baselines.
- `quadmix_pub` is the only QuaDMix representative. The retired style proxy never enters a
  reported cell.

Paper-facing names, applicable cells and result paths are described in
[RUN_RECORD.md](RUN_RECORD.md). Implementation notes and citations are in
`benchmark/Methods/<Name>/README.md`. The fidelity assertions are exercised by
`omniselect/tests/test_baseline_fidelity.py` and
`omniselect/tests/test_published_core_fidelity.py`.

## Departures from the published formulations

Each external baseline is implemented from its own selection rule under the shared protocol. Its
inputs differ by method and by data family, and each entry below names them. The registry ids
`quadmix_pub` and `dmf_pub` name those two implementations in the code. The paper states the
contract, and the full record is here. The appendix released with the paper, `docs/OmniSelect_20261027_appendix.pdf`, carries the
same list.

### QuaDMix (`benchmark/Methods/QuaDMix/method.py`, registry id `quadmix_pub`)

Retained. Normalized quality criteria are merged inside a domain, the merged score is turned into a
token-weighted within-domain percentile rank, and the rank passes through a sigmoid in the rank
threshold `omega` with slope `lambda`, so records above the threshold keep only the floor `epsilon`.

Departures.

- The implemented weight inside the threshold is `eta / (1 + exp(-lambda * (omega - r))) + epsilon`,
  while Eq. 3 of the paper reads `[2 / (1 + exp(-lambda * (omega - r)))]^eta + epsilon`. At the
  default `eta = 1` the two differ by the factor 2 on the sigmoid part, and the floor `epsilon` is
  not scaled with it, so the induced sampling distributions are close but not identical.
- The percentile rank is accumulated over records sorted by the merged score, so tied records
  receive distinct ranks, while Eq. 2 assigns the whole tie group the rank of its last element.
- Token weights default to one per record outside the text track, which uses document token counts.
- Domains are native domain labels where a track provides them and k-means clusters of the
  representation otherwise.
- Expected repeat counts are converted into a fixed-size subset by Gumbel top-k sampling without
  replacement, because the benchmark fixes the budget of every strategy, as a record count outside
  the text track and as a per-domain token count inside it.
- The quality input is the authenticity signal outside the text track and the reference-model loss
  inside it. The published proxy-model training and the search over sampling parameters are not
  run, and fixed default parameters are used, which is why the mechanism table of the paper marks
  the method as reading no target-task validation feedback.

### DMF (`benchmark/Methods/DMF/method.py`, registry id `dmf_pub`)

Retained. Actor scores are combined into one score by the published fusion form, and the weights
are moved by the centered additive rule of the published update equation.

Departures.

- The actors are authenticity, influence and redundancy, in place of the published quality, domain
  and topic actors.
- The reward of an actor is the construction-split utility of the subset that this actor alone
  selects, in place of the published aggregation of category weights and category rewards. The
  subset that the current fused weights select is scored as well, and that score only decides which
  iteration is reported.
- The actor-memory update is not run, and the actor set is fixed over the iterations.
- The updated weights are clipped at zero and renormalized, a step the published update equation
  does not contain.
- The reported subset is the best of the iterations under the construction-split score, rather than
  the subset of the last iteration.
- On the text track the reward is the negative mean per-domain loss of a reference set after a
  short fine-tuning run.

### Other baselines

The two sections above cover the methods that fuse several signals. For every other baseline the fidelity
tier and the adaptation are in the table of this page, and the registered implementation is the
single source of truth. Density replaces the published locality-sensitive density estimator by a
k-nearest-neighbor estimator. GraNd uses a last-layer gradient-norm proxy. Their published scores need supervised class probabilities, so EL2N, GraNd
and CCS report separately evaluated adaptations on forecasting and text, which are not members of
the adjudicated portfolio of those runs.
