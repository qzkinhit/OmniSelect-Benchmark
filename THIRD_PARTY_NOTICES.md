# Third-party notices

The root [MIT license](LICENSE) covers the project's own code. Datasets, model weights and the
methods reimplemented under `benchmark/Methods/` keep their original terms. The evidence below was
collected on 2026-07-16. A citation
requirement alone does not establish redistribution terms for a particular copy.

## Datasets

| Local material or source | License evidence | Status |
| --- | --- | --- |
| `data/tep/` (Tennessee Eastman, Braatz files) | The Braatz distribution (github.com/camaramm/tennessee-eastman-profBraatz) carries a license with source and binary notice and non-endorsement conditions. | The lineage of this copy and the applicable notice remain unverified. Checksums in `data/tep/SHA256SUMS.txt`. |
| `data/daisy/` (DaISy 98-002, 98-003) | KU Leuven ESAT-SISTA DaISy page asks users to cite the database. | No explicit redistribution license found. |
| `data/cifar_n/` (CIFAR-10N, CIFAR-100N labels) | UCSC-REAL/cifar-10-100n release (Wei et al., ICLR 2022). | Redistribution terms of this copy remain unverified. |
| CIFAR-10, CIFAR-100 (Hugging Face `uoft-cs/cifar10`, `uoft-cs/cifar100`) | The CIFAR page states a citation requirement and no license. The dataset cards say "unknown". | Downloaded at pinned revisions, not redistributed here. |
| ImageNet-100 (Hugging Face `clane9/imagenet-100`) | Subset of ImageNet, subject to the ImageNet terms of access. | Downloaded at a pinned revision, not redistributed here. |
| ETTh1, ETTh2, ETTm1 (zhouhaoyi/ETDataset) | CC BY-ND 4.0. | Downloaded at a pinned commit and re-saved by pandas. Not redistributed here. |
| OpenML Electricity (id 151) | OpenML API licence field "Public". | Fetched through scikit-learn, not redistributed here. |
| Text pool sources | fineweb-edu ODC-BY 1.0. Finemath, codeparrot-clean-valid, coco-karpathy and mstz/adult per their dataset cards. COCO captions under the COCO terms of use. | The pool is built locally by `data/build_text_pool.py` and not redistributed here. Text-100k uses `data/build_text_pool_adult_v2.py` and additionally downloads the original UCI Adult training file; those source files are not redistributed here. |

## Model weights

CLIP ViT-B/32 (`openai/clip-vit-base-patch32`), SmolLM2-135M and 360M (`HuggingFaceTB`),
chronos-bolt tiny and small (`amazon`) and TabPFN-v2 (Prior Labs) are downloaded at run time and
remain under their model-card licenses. TabPFN-v2 weights carry the Prior Labs license and its
attribution requirement.

## Method code

No third-party source code is vendored. The methods under `benchmark/Methods/` are
reimplementations of published selection rules. Each README names the source paper, what is
transferred and the fidelity label. `benchmark/tools/deepcore_reference.py` reimplements the DeepCore
herding, k-center, EL2N and GraNd rules for fidelity checks. The DSIR ordering was aligned with
the official `data-selection` package, which is not included. The text Entropy-Law (ZIP) and
Tab-AICL implementations are written from the papers and the authors' public descriptions.

GLISTER and GRAD-MATCH (`benchmark/Methods/GLISTER`, `benchmark/Methods/GRADMATCH`) are written from
the papers. Their greedy update of the validation gradient and the regularized non-negative OMP
follow the structure of CORDS (github.com/decile-team/cords, MIT License, Copyright (c) 2021
decile-team), files `cords/selectionstrategies/SL/glisterstrategy.py`, `gradmatchstrategy.py` and
`cords/selectionstrategies/helpers/omp_solvers.py`, which were consulted. No CORDS source is copied.
LESS (`benchmark/Methods/LESS`) is written from the paper. The official repository
(princeton-nlp/LESS) is MIT licensed and was not copied. InfoMax (`benchmark/Methods/InfoMax`) is
written from the objective of the paper only, because its repository carries no license.
