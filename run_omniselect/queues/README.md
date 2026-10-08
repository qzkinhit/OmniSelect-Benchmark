# Queues

Each line is one command followed by `# cell=<run record directory> name=<job>`.
Paths are relative to the repository root. The queue runner skips cells that already contain
`decision.json`, so an interrupted queue can resume.

| File | Protocol | Cells | Content |
|---|---|---:|---|
| `main_v2_3.txt` | `v2_2` | 36 | Main runs, including auxiliary C-10-CLIP |
| `ablation_v2_3.txt` | `v2` | 36 | OmniSelect-NC |
| `main_adapt_v2_2.txt` | `v2_2`, standalone | 18 | EL2N, GraNd and CCS on forecasting and text |
| `scale_up.txt` | `v2_2`, `v2`, standalone | 15 | IN-100 and Text-100k, including NC and text adaptations |
| `robustness.txt` | `v2_2` | 153 | Injection ratios, validation-label conditions, unseen mechanisms and learner swaps |
| `signal_drop.txt` | `v2_2` | 45 | Remove each signal on the five recorded tasks |
| `validation_size.txt` | `v2_2` | 9 | CIFAR-100 with 200, 400 and 1600 validation records |

The run seeds are listed in `results/paper/seeds_reported.json`.
The robustness 40% injection and clean-validation controls, and the validation-size 800 point,
reuse the matching main runs. The signal-removal queue reads its controls from
`results/paper/work/`, which `bash results/paper/rebuild.sh` prepares. It reuses a cached
candidate when the complete prediction files are present and fits missing candidates otherwise.

## Running a queue

Prepare the inputs using [../../data/README.md](../../data/README.md), then run from the repository root.

```bash
python -m run_omniselect.run_queue run_omniselect/queues/main_v2_3.txt --dry-run
python -m run_omniselect.run_queue run_omniselect/queues/main_v2_3.txt --gpus 0,1 --jobs-per-gpu 2 --threads 4
python -m run_omniselect.run_queue run_omniselect/queues/scale_up.txt --gpus 0,1 --jobs-per-gpu 1 --threads 8
```

With `--gpus`, each job sees one device through `CUDA_VISIBLE_DEVICES`, and at most
`--jobs-per-gpu` jobs share a device. `--threads` sets the BLAS and OpenMP threads of each job.
Logs are written to `results_and_logs/logs/<job>.log`. Exit code 3 indicates a missing model or
dataset and is recorded as skipped. Other nonzero exit codes are recorded as failed.

## Custom runs

The generator copies the reported templates by default. Explicit seed or protocol overrides
generate custom commands from the corresponding experiment preset. Generated queues are written
to `results_and_logs/generated_queues/`. `--output-dir` (also `--out`) selects another directory.

```bash
python -m run_omniselect.make_queues --batches main_v2_3 robustness
python -m run_omniselect.make_queues --batches main_v2_3 --seeds 3,4,5 --heavy-seeds 3,4,5
python -m run_omniselect.make_queues --results-root /data/omniselect_runs
```

Set `HF_HOME`, `TABPFN_MODEL_CACHE_DIR` and `SCIKIT_LEARN_DATA` to choose cache locations,
and `HF_HUB_OFFLINE=1` to use already cached model files.
