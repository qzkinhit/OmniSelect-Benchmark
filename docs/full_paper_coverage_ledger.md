# Full-paper coverage ledger (2026-07-16, server-week close-out)

Scope: current-formal-implementation (controller re-validation v5 adopted as canonical).
Legend: DONE = 3-seed current-code artifacts present; PARTIAL = <3 seeds; PROXY = auxiliary
not in main table; EXCLUDED = out of this week's scope (compute extension, unchanged claim).

## Main cross-modal table (paper Table: four primary modalities)

| Modality | Downstream model | Seeds | Status | Source |
|---|---|---|---|---|
| Vision CIFAR-100 | frozen CLIP + linear head | 0,1,2 | DONE | vision_full / newbaselines / controller-reval |
| Time ETTh1 | DLinear from scratch | 0,1,2 | DONE | split_protocol / controller-reval |
| Process TEP | MLP fault classifier | 0,1,2 | DONE | tep_full / tep_fdr_full / controller-reval |
| Tabular electricity | TabPFN-v2 | 0,1,2 | DONE | tabular_full / controller-reval |

## External baselines table

herding, k-center, EL2N, GraNd, CCS, SemDeDup, Density, QuaDMix, DMF, D4, DsDm - all
3-seed in external_baselines / newbaselines logs and folded into controller portfolio. DONE.

## Supplementary datasets

| Dataset | Seeds | Status |
|---|---|---|
| ETTh2 (LSF) | 0,1,2 | DONE |
| ETTm1 (LSF) | 0,1,2 | DONE (chronos + DLinear) |
| DaISy CSTR | 0,1,2 | DONE |
| DaISy steamgen | 0,1,2 | DONE |
| CIFAR-100N real noise | 0,1,2 | DONE |

## Time-series foundation model (Chronos) arm

ETTh1/ETTh2/daisy_cstr/daisy_steamgen/ETTm1 x 3 seeds x 16 methods. DONE (chronos_fm +
chronos_ettm1 logs; 12+3 recovered JSONs).

## Text arm

| Item | Seeds | Status |
|---|---|---|
| Main text lane (STRATIFY=1, pplq, SmolLM2 finetune, lm-eval ARC-e/c+HellaSwag+OBQA) | 0,1,2 | DONE (D_VALIDATED_OK) |
| global-mix RegMix proxy (STRATIFY=0) | 0 done; 1,2 RUNNING (P0-cond) | PARTIAL->DONE | labeled PROXY always, never in STRATIFY main table |

## Original-protocol reproduction (Data-Diet / CCS)

| Setting | Seeds | Status |
|---|---|---|
| CIFAR-10 full (45k, ResNet-18 from scratch, 160ep, SCORE_RUNS=3) | 0 done; 1,2 RUNNING (P0) | PARTIAL->DONE |
| ImageNet-100 @112px/40ep (reduced-scale qualitative) | 0,1 done; 2 RUNNING (P1) | PARTIAL->reduced-scale 3-seed |

## Ablations

margin (bitwise), VAL_N sweep (Thm4), V1/V2 split, self-improving candidates, noise-ratio
20/60% (F), channel-drop reduced-portfolio (G: drop infl/red/auth), validation contamination,
SH prescreen, budget-response. All DONE.

## Theory checks

Thm3 identification, Thm4 finite-sample, Thm5 kappa-scaling (ImageNet high-kappa: EL2N/GraNd
collapse, CCS>random confirmed), Thm6 flip-magnitude scatter (fig present; refresh with
current-code deltas at backfill), Prop5 gamma_hat=8.9%, Prop6 contamination. DONE / backfill-refresh.

## EXCLUDED this week (compute extension, paper wording already says so)

DCLM / DataComp filtering track / GIFT-Eval full leaderboards; full ImageNet-1k; full LSF
sweep; GSM8K / MATH / HumanEval / MBPP (need code-pool retraining and larger batches - cost
and protocol gap must be reported before any expansion). Unchanged paper claim.

## Backfill obligation

All numbers refreshed from CURRENT-code canonical artifacts (controller_current_canonical_v5.json
+ isolated results.json). The 19 historical-vs-current controller deltas documented in
controller_reval_v5_changed19_classification.md; ETTh1 stays second-place temporal-drift.
No test-set dominance claims (validation-selection + generalization-gap + multi-seed only).
