# Dataset provenance

All source datasets are kept separate from generated candidate pools. Controlled
corruption is deterministic and does not modify upstream files.

| Task | Source | Registered version or check |
|---|---|---|
| CIFAR-100 | Hugging Face `uoft-cs/cifar100` | revision `aadb3af77e9048adbea6b47c21a81e47dd092ae5` |
| CIFAR-10 | Hugging Face `uoft-cs/cifar10` | revision `0b2714987fa478483af9968de7c934580d0bb9a2` |
| CIFAR-100N | UCSC-REAL `cifar-10-100n` | `CIFAR-100_human.pt`, SHA-256 `bd2d80409754d420292d622e15e25248ba21e37d27429efac49c8da723f44394` |
| ImageNet-100 | Hugging Face `clane9/imagenet-100` | revision `0519dc2f402a3a18c6e57f7913db059215eee25b` |
| ETTh1, ETTh2, ETTm1 | ETT repository | file checks performed by `data/fetch_data.py` |
| TEP21 | Tennessee Eastman process files | 44-file SHA list in `data/tep/SHA256SUMS.txt` |
| DaISy CSTR | KU Leuven DaISy 98-002 | tracked research copy, SHA checked by `data/fetch_data.py` |
| DaISy steam generator | KU Leuven DaISy 98-003 | tracked research copy, SHA checked by `data/fetch_data.py` |
| Electricity | OpenML data ID 151, version 1 | checksum of the numeric arrays checked by the tabular track |
| Text, general | `HuggingFaceFW/fineweb-edu` | revision recorded in the generated pool manifest |
| Text, mathematics | `HuggingFaceTB/finemath`, `finemath-4plus` | revision recorded in the generated pool manifest |
| Text, code | `codeparrot/codeparrot-clean-valid` | revision recorded in the generated pool manifest |
| Text, captions | `yerevann/coco-karpathy` | revision recorded in the generated pool manifest |
| Text, table records | `mstz/adult`; Text-100k additionally uses UCI `adult.data` | pinned revisions and source SHA-256 values in `benchmark/Data/text_manifest_adult_v2.py` and the generated pool manifest |

## Candidate-pool construction

The vision, forecasting, process and tabular tracks corrupt 40% of the pool by default
(`TrackConfig.noise_frac`, mechanisms in `tracks/common/injection.py`). They select the affected
records with `np.random.default_rng(seed + 7)` and attach one tag to every record, written to
`splits.json` of each run record.
Corruption types are appropriate to each modality, including label flips, feature or
sensor corruption, temporal degeneration, and near duplication.

The text builder (`data/build_text_pool.py`) creates a five-domain pool and records source
revisions, shard names, sizes, hashes and construction parameters in
`data/processed/pool_manifest.json`. The Text-100k builder (`data/build_text_pool_adult_v2.py`)
retains the registered held-out file and records the UCI Adult mapping and selected source rows.

## Split isolation

Pool, validation and test ids are disjoint. Every run record stores them with their sha256 in
`splits.json`. The test split is not read to construct candidates or to elect the final strategy.

## License scope

The MIT license covers code only. Dataset terms remain those of each upstream source, summarized in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
