# ImageNet-100

The main-table native cells use `imagenet100_128.npz` with 112 px crops. The IN-100 scale-up
cells use `imagenet100_256.npz` with 224 px crops. Both use Hugging Face `clane9/imagenet-100`
at revision `0519dc2f402a3a18c6e57f7913db059215eee25b` and the same RGB, shorter-side resize
and center-crop preparation. Files here are ignored by Git.

`python data/prepare_imagenet100.py --size 256` builds the scale-up NPZ through `datasets`.
`python data/imagenet100_npz.py --snapshot <snapshot dir> --out data/imagenet100 --size 256`
builds the same arrays from downloaded parquet shards and writes a manifest containing the
shard, array and file hashes. Use `--track-set imagenet_source=npz --track-set imagenet_npz_size=256 --track-set in_res=224`
for the scale-up run. See [../README.md](../README.md) for the recorded fingerprint.
