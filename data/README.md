# Data setup for the 12-task benchmark

The repository does **not** silently substitute missing data. Large or license-sensitive
datasets must be downloaded from their official source; every runner fails closed when a
required artifact is missing. Exact source/version/SHA details are recorded in
[`docs/dataset_provenance.md`](../docs/dataset_provenance.md) and
[`docs/ARTIFACTS_INDEX.md`](../docs/ARTIFACTS_INDEX.md).

## One-command setup

```bash
pip install -e ".[train,eval,arms]"
python scripts/fetch_data.py
```

This downloads/pre-warms the public ETT, CIFAR, OpenML and text sources and verifies the
small local sources. Use `--verify-only` to hash an existing checkout without downloading.

## Dataset matrix

| # | Modality / task | Dataset | How to obtain |
|---:|---|---|---|
| 1 | image classification | CIFAR-100 | `python scripts/fetch_data.py --only hf` (HF `uoft-cs/cifar100`, pinned revision in provenance) |
| 2 | real-noise image classification | CIFAR-100N | Download `CIFAR-100_human.pt` from `UCSC-REAL/cifar-10-100n`, place at `data/cifar_n/`, verify SHA from provenance |
| 3 | image classification | CIFAR-10 | `python scripts/fetch_data.py --only hf` (HF `uoft-cs/cifar10`) |
| 4 | image classification | ImageNet-100 | HF `clane9/imagenet-100`; the original-protocol runner downloads/caches it on first use |
| 5 | time-series forecasting | ETTh1 | `python scripts/fetch_data.py --only ett` |
| 6 | time-series forecasting | ETTm1 | same |
| 7 | time-series forecasting | ETTh2 | same |
| 8 | process fault diagnosis | Tennessee Eastman (TEP21) | verify the 44 `data/tep/*.dat` files against `docs/provenance_evidence/tep_files_sha256.txt`; if absent, obtain from the source listed in provenance |
| 9 | tabular classification | OpenML Electricity (data_id=151) | `python scripts/fetch_data.py --only hf` pre-warms the OpenML cache |
| 10 | industrial process forecasting | DaISy CSTR 98-002 | verify `data/daisy/cstr.dat`; source and SHA in provenance |
| 11 | industrial process forecasting | DaISy steam generator 98-003 | verify `data/daisy/steamgen.dat`; source and SHA in provenance |
| 12 | text language modeling | five-domain text pool | `python scripts/fetch_data.py --only text`; see below |

## Five-domain text pool

`scripts/build_pool_failclosed.py` constructs the text pool from five official sources:

- general: `HuggingFaceFW/fineweb-edu`;
- math: `HuggingFaceTB/finemath` (`finemath-4plus`);
- code: `codeparrot/codeparrot-clean-valid`;
- image-caption proxy: `yerevann/coco-karpathy` (caption text only);
- table-text proxy: `mstz/adult`.

The frozen builder takes 5,000 candidate records per domain (25,000 total) and 400
held-out records per domain (2,000 total). It writes:

```text
data/processed/qpool_train.jsonl
data/processed/qpool_heldout.jsonl
data/processed/pool_manifest.json
```

The manifest records the resolved upstream revisions, shard filenames, byte sizes, and
SHA256 values. The pool files are intentionally not stored in Git; rebuild them locally
and verify their manifest before running the text arm.

## Important scope notes

- CIFAR-100N, TEP and DaISy licensing/redistribution status differs by source. Follow the
  source pointers and checksum the exact bytes; do not replace them with a different copy.
- Text split IDs were not exported by the historical run. The public result is therefore
  labeled `PASS_WEAK/NOT_CAPTURED`, and no post-hoc split manifest is fabricated.
- Git stores code, small results, manifests and checksums—not model weights, HF caches, or
  multi-gigabyte score caches.
