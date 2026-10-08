# Datasets

`data/fetch_data.py` downloads or verifies every dataset the tracks read. Loaders live in
`benchmark/Data/`. Each track reads its files through them and stops when a file is missing or
its sha256 differs from the recorded value. Sources, revisions and checksums are listed in
[dataset provenance](../docs/dataset_provenance.md).

```bash
python data/fetch_data.py --verify-only --only tep daisy cifar_n   # committed files
python data/fetch_data.py --only ett hf text                       # downloads and the text pool
python data/fetch_data.py --only imagenet                          # ImageNet-100 (about 8 GB), native track only
```

| Track | Task | Files | In Git |
|---|---|---|---|
| vision | CIFAR-100, CIFAR-10 (frozen CLIP) | Hugging Face `uoft-cs/cifar100`, `uoft-cs/cifar10` at pinned revisions. CLIP features cached in `processed/vision_*.npz` | no |
| vision | CIFAR-100N | [`cifar_n/`](cifar_n/README.md) human labels | yes |
| native | CIFAR-10 (ResNet-18) | [`cifar10_np/cifar10.npz`](cifar10_np/README.md), written by `fetch_data.py --only hf` | no |
| native | ImageNet-100 | Hugging Face `clane9/imagenet-100`, or [`imagenet100/`](imagenet100/README.md) after `prepare_imagenet100.py` | no |
| timeseries | ETTh1, ETTh2, ETTm1 | [`processed/`](processed/README.md) `etth1.csv`, `etth2.csv`, `ettm1.csv` | no |
| timeseries | DaISy CSTR, steam generator | [`daisy/`](daisy/README.md) `.dat.gz` (the `.dat` copies are decompressed on first use) | `.gz` only |
| process | Tennessee Eastman (TEP21) | [`tep/`](tep/README.md) `d00` to `d21`, training and test files | yes |
| tabular | OpenML Electricity (id 151) | scikit-learn OpenML cache. See [`electricity/`](electricity/README.md) | no |
| text | five-domain pool | [`processed/`](processed/README.md) `qpool_train.jsonl`, `qpool_heldout.jsonl`, `pool_manifest.json`, built by `build_text_pool.py` | no |

The text pool has 25,000 training records (5,000 per domain, 3,000 clean and 2,000 corrupted by
truncation, template, cross-domain or low-tier text) and 2,000 held-out records (400 per domain).
The registered files have sha256 `84e174dbb097288c6b4473af2af8d6cb46a0b00a000b6534545725e53f9939c5`
(`qpool_train.jsonl`) and `1e4a45c9c959995a3c10c840dc2a8b84ce33bd82ba17682c8f50c3b4b3a1e785`
(`qpool_heldout.jsonl`), and the default `benchmark/Data/text.py` loader checks these hashes.
`build_text_pool.py` fetches pinned shards of five public sources and fails when a source falls
short. `validate_text_pool.py` checks the result independently.

The Text-100k scale-up pool uses `build_text_pool_adult_v2.py`. Each domain contains 12,000
clean and 8,000 corrupted records; the registered 2,000 held-out records are unchanged.
The table domain supplements the pinned `mstz/adult` training rows with UCI `adult.data`,
using the same typed fields and excluding held-out rows and duplicate clean records.
The explicit manifest records the sources, transformations and selected rows. Set
`OMNISELECT_TEXT_MANIFEST` only when running this 100k pool.

```bash
python data/build_text_pool_adult_v2.py --data-root data/text100k \
  --registered-held data/processed/qpool_heldout.jsonl \
  --hub-root "${HF_HOME:-$HOME/.cache/huggingface}/hub" --external-root data/uci_adult
python data/validate_text_pool.py --data-root data/text100k \
  --manifest data/text100k/processed/pool_manifest.json \
  --external-root data/uci_adult --marker data/text100k/VALIDATED_OK
export OMNISELECT_TEXT_MANIFEST=data/text100k/processed/pool_manifest.json
```

The IN-100 scale-up cells read `imagenet100/imagenet100_256.npz`. `imagenet100_npz.py` builds
it from the 18 parquet shards of Hugging Face `clane9/imagenet-100` at revision
`0519dc2f402a3a18c6e57f7913db059215eee25b`. It applies the per-image transform of
`prepare_imagenet100.py` (RGB, shorter side to 256 px with PIL bilinear, center crop 256 by 256,
uint8) in shard and row order. The file holds 126,689 training and 5,000 validation images of
100 classes (25,892,165,530 bytes, sha256
`9291888c7e7f3a1b687d5950f0d57f94e2753545e08715a7c190b6270ec8de1f`). The adjacent manifest lists
the source, shard and array hashes. The native track reads it with
`--track-set imagenet_source=npz --track-set imagenet_npz_size=256 --track-set in_res=224` and takes 224 px crops.
The main-table 112 px cells use the corresponding 128 px NPZ.

```bash
python data/imagenet100_npz.py --snapshot <hf snapshot dir of clane9/imagenet-100> --out data/imagenet100 --size 256
```

Dataset files do not inherit the code's MIT license. [Third-party notices](../THIRD_PARTY_NOTICES.md)
records the terms and the open provenance questions of each copy.
