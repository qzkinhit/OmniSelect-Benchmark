# Frozen body-comparison baseline list + fidelity gates (2026-07-16)

Release rule: freeze the baselines that receive a NUMERICAL comparison in the paper body,
then fidelity-gate ONLY those on their original-paper data + official protocol. Do not
optimize baselines, do not change datasets. Everything else is cited-only / compute-extension.

## Tier 1 - body numerical comparison, original-protocol fidelity RUNNABLE locally

Clean CIFAR-10, ResNet-18 from scratch (the DeepCore / Data-Diet / CCS protocol; data present).
Fidelity gate = reproduce the published pruning-curve SHAPE within tolerance.

| Baseline | Original paper | Fidelity gate | Status |
|---|---|---|---|
| Random | - | lower bound | trivially present |
| EL2N | Paul et al. 2021 (Data-Diet) | keep-fraction sweep: near-random at low pruning, collapse at high pruning | curve RUNNING |
| GraNd (proxy) | Paul et al. 2021 | same curve; labeled last-layer gradient-norm PROXY, not full-network GraNd | curve RUNNING |
| CCS | Zheng et al. 2023 | robust at high pruning where EL2N collapses; local equal-width+dynamic-realloc impl (toy-tested) | curve RUNNING; official-impl unavailable (no GitHub on server) - stated |

## Tier 2 - body numerical comparison, official-package fidelity verified

| Baseline | Original | Fidelity | Status |
|---|---|---|---|
| DSIR | Xie et al. 2023 | official `data-selection` pkg: spearman 0.855, top-50% overlap 85% on shared pool | DONE (pkg consistency; original-task effect = cited scale-up) |

## Tier 3 - body comparison but LOCAL REIMPLEMENTATION (must be labeled, not "faithful")

herding, k-center, SemDeDup, Density, QuaDMix, DMF, D4, DsDm. These are unified-testbed
reimplementations from the original scoring rules. They are NOT original-protocol fidelity
reproductions. Paper wording: "reproduced from the scoring rules under the unified record
format", never "faithfully reproduces the original result". herding/k-center are DeepCore
geometric coresets and CAN be added to the CIFAR-10 curve later if a numerical comparison is
kept; otherwise they stay unified-testbed only.

## Tier 4 - cited only / compute-extension (NOT body numerical comparison)

DoReMi, QuRating, DsDm-full-datamodels, DCLM/DataComp filtering track, Chronos-curation,
T-MARS, MetaCLIP, RegMix (global-mix PROXY, 3-seed done but never in the STRATIFY main
table). Paper already lists these as compute-extension; they get no body number.

## Tab-AICL - retracted from numerical comparison

Reversed ordering under our setting -> not same-protocol comparable. Both drafts now cite it
as related work only (AAAI already; ICLR fixed this round). No body number.

## Immediate paper corrections done this round

- AAAI: GraNd re-labeled as last-layer gradient-norm PROXY (was "faithfully implemented").
- ICLR: Tab-AICL positive numerical comparison removed, replaced with related-work-only note.

## Fidelity gate now running

CIFAR-10 clean pruning-curve for {random, el2n, grand, ccs} across keep in {0.1,0.2,0.3,0.5,0.8},
ResNet-18 from scratch, to reproduce the Data-Diet/CCS published curve shape within tolerance.
Isolated under run_id=fidelity-curve. After it passes, proceed to method-v2.

## Tier-1 fidelity gate RESULT (PASSED, FIDELITY_CURVE_OK)

CIFAR-10 clean, ResNet-18 from scratch, SCORE_RUNS=3, test acc by keep fraction:

| keep | random | EL2N | GraNd | CCS |
|---|---|---|---|---|
| 0.1 | 0.638 | 0.183 | 0.232 | 0.462 |
| 0.2 | 0.726 | 0.359 | 0.372 | 0.586 |
| 0.3 | 0.817 | 0.486 | 0.606 | 0.779 |
| 0.5 | 0.879 | 0.900 | 0.894 | 0.875 |
| 0.8 | 0.917 | 0.928 | 0.929 | 0.896 |

Reproduces the published shape: (1) high pruning (keep<=0.3) EL2N/GraNd collapse far below
random; (2) CCS robust there (>> EL2N/GraNd); (3) low pruning (keep>=0.5) EL2N/GraNd recover
to slightly above random (keeping hard examples helps at moderate pruning - the Data-Diet
point), so CCS is legitimately NOT dominant at low pruning. This is the original-protocol
fidelity evidence for EL2N / GraNd(proxy) / CCS. Note: CCS is our equal-width+dynamic-realloc
local implementation (toy-tested); no official CCS repo (GitHub blocked on server).
