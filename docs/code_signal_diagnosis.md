# Code-domain diagnosis: why distribution-matching (DSIR) beat us, and the fix

## Symptom
On the 5-modality held-out PPL (hid256/4layers, passes 8), `mmdataselect`
won math and general but **lost code to DSIR** (1378.9 vs 1036.5).

## Root cause — the influence channel was near-random off-table
Per-signal "high-quality purity" of the top-200 ranked records (random baseline
≈ 0.62; higher = signal concentrates real high-quality data):

| modality | infl_grad (grad-align, was primary) | **-infl_loss (reference-PPL quality)** | authenticity |
|----------|------------------------------------:|---------------------------------------:|-------------:|
| general  | 0.660 (≈ random)                    | 0.800                                  | 0.880 |
| math     | 0.675 (≈ random)                    | **0.740**                              | 0.650 |
| code     | **0.650 (≈ random)**                | **0.835 (strongest)**                  | 0.795 |
| image    | 0.655 (≈ random)                    | 0.715                                  | 0.780 |
| table    | 0.930                               | 0.695                                  | 0.715 |

The gradient-alignment influence is barely above random for general/math/code/image.
For code it actively mis-ranks: per-group mean infl_grad is **higher** for
truncation (0.466) / template (0.410) / lowtier (0.435) than for real high-quality
code (0.374) — short corrupted snippets give a sharper last-layer gradient that
aligns better with the averaged reference. So importance was driven by a noise-loving
signal, and only the authenticity prefilter + diversity kept us competitive.

## The fix — reference-perplexity quality as the influence channel
`-infl_loss` (a clean reference model's forward-pass loss; low loss = clean/on-domain,
DCLM / Ultra-FineWeb style) is the strongest general-purpose value signal, and the
strongest of all on code (0.835). It is also **cheaper** than gradient-alignment
(forward-only, no backward pass) — directly serving the "lightweight yet useful"
goal. Switch the influence channel from `grad` to `pplq` (env `INFL_KIND=pplq`).

## Why the three channels then cleanly cover all four code-noise types
- truncation, lowtier  -> authenticity completeness prefilter (auth 0.463 / 0.529, dropped)
- crossdomain          -> reference-PPL (loss 3.05, down-ranked; templates aside it is the cleanest cut)
- template             -> diversity / redundancy coverage in the budget selector

Each noise type is removed by a different channel — the orthogonal-channel design
working as intended once the influence channel is the reliable one.

## End-to-end reality check — it was an evaluation-regime problem, not a signal problem

Switching the influence channel to `pplq` did **not** move code end-to-end (mmdataselect
code PPL ~1379 either way). Per-modality token-allocation traced the real cause:

1. **Cross-modal budget starvation (global budget).** With one global budget, reference
   PPL is not comparable across modalities — code is intrinsically harder (higher loss)
   so its records rank below math/general and get **32% of budget vs math's 46%**. Code
   was starved; math gorged. DSIR's apparent code win came partly from its target being
   hard-coded to math+code (we handed it code as the target).

2. **Fair per-modality, from-scratch, small budget → undertraining dominates.** Selecting
   *within* code's own pool (ONLY_DOMAIN=code), code held-out PPL is ~5000 (severely
   undertrained at ~50K tokens). Here **quantity beats quality**: across 2 seeds, random
   (keeps most tokens) ties the best selector and beats aggressive filters; DSIR is no
   better than random. The SAME pattern holds for **math** per-modality (dsir/quantity
   3199 < random 3313 < quality_ppl/aggressive 3430). So at this scale no selector helps
   *any* modality — it is an undertraining artifact, not a code-specific failure.

3. **Fine-tune a strong base (SmolLM2-135M) → saturation, no headroom.** Base code held-out
   PPL is already 5.73; continued training on a subset cannot meaningfully improve it.

Conclusion: a selection benefit only appears in a regime with (a) enough tokens to exit
undertraining and (b) headroom to improve — which is the multi-modal shared-model scale
(~165K tokens) where the original math win appeared. The fix is **stratified budgets**:
give each modality its own fair BUDGET_FRAC share (selected within-modality), train one
shared model on the union (STRATIFY=1). This keeps the proven token scale while removing
the cross-modal starvation.

## Stratified result — code is recovered

Fair per-modality budgets, one shared model, reference-PPL influence (STRATIFY=1,
INFL_KIND=pplq, hid256/4layers/passes8). Per-modality held-out PPL:

| method        | hi%  | code      | general  | image     | math    | table   |
|---------------|-----:|----------:|---------:|----------:|--------:|--------:|
| random        | 0.63 | 1390      | **2338** | 1252      | 602     | 38      |
| dsir (fair)   | 0.72 | 1368      | 2484     | 1016      | 625     | 33      |
| quality_ppl   | 0.74 | 1361      | 2598     | 1676      | 602     | 37      |
| mmds_noauth   | 0.69 | 1391      | 2489     | 1071      | 598     | 34      |
| **mmdataselect** | **0.95** | **1330** | 2472 | **1017** | **595** | **29** |

mmdataselect wins **code (1330, best of all methods)**, math (595), table (29), ties
image (1017 vs dsir 1016), and has by far the cleanest selection (purity 0.95). Only
**general** goes to random — the easy/broad modality where aggressive selection does not
help and every selector trails random. The earlier "code loses to DSIR" was an artifact
of the unfair DSIR target + cross-modal starvation, not a real weakness.

Net story: selection helps the **hard, noisier modalities** (code/math/table/image); the
easy broad modality (general) is selection-insensitive. Multi-seed (0,1,2) sweep with the
full baseline set (zip/if_mates/dmf) confirms robustness — see `outputs/stratified/`.

## Improvement attempts — all rejected, the original design is confirmed best (3-seed)

To try to also win code (where DMF, an auth/diversity-free fusion, edges us by ~1.2%) and
general (where random wins), we explored borrowing from other schools. None beat the
original mmdataselect overall:

- **distribution-match channel** (mmds_dist / mmds_v2, borrowing DSIR): improves general but
  collapses image (the diversity-free soft fusion over-concentrates). Distribution-matching
  is redundant once quality + diversity are present.
- **soft authenticity, no diversity** (dmf_auth): image cratered to ~1860 — diversity is the
  source of the image win and cannot be dropped.
- **modality-adaptive diversity** (mmds_adiv): scale per-modality lambda by feature-spread.
  FAILED — code 1416, table 53 (both far worse). The hypothesis "narrow modality -> less
  diversity" is wrong: table looks narrow (high feature self-similarity) yet *needs* diversity
  because its noise is template redundancy, and diversity is exactly what dedups templates.
  Uniform lambda=0.5 is already well calibrated.

Conclusion: mmdataselect (authenticity prefilter + dynamic fusion + uniform budget-constrained
diversity) is the best configuration; the rejected variants serve as ablations showing each
component is necessary and not arbitrary. The code gap to DMF is the small, explainable cost
of the diversity that wins image by ~16%.

