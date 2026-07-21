# Reproduce

Two reproduction tracks, sharing one codebase:

- **Pure-CPU track** — no downloads, no torch, runs in seconds. Verifies the full
  system path (signals -> Multi-Actor fusion -> budget selection -> manifest).
- **Real track** — SmolLM2-135M on a merged FineWeb-Edu / FineMath / Python-Edu text
  pool, with a real downstream train + lm-eval evaluation (needs a GPU or Apple MPS
  for a tolerable runtime).

All paths in configs are repo-relative; the runners resolve them against the repo
root, so commands below are run from the repo root.

## 0. Install

Core only is enough to `import mmdataselect` and run the pure-CPU smoke (no torch):

```bash
pip install -e .
```

Full local stack (downstream training + benchmark evaluation):

```bash
pip install -e ".[train,eval]"        # torch, transformers, datasets, accelerate, peft, lm-eval
# or, including dev (pytest):
pip install -e ".[train,eval,dev]"
```

Or use the helper, which creates `.venv` and installs `[train,eval,dev]`:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

Extras and why (`pyproject.toml`):

| extra | brings | needed for |
|---|---|---|
| (core) | numpy, pyyaml, zstandard, rich, tqdm | import + pure-CPU select + sanity smoke |
| `train` | torch, transformers, datasets, accelerate, peft | `tools/standardize/*` (datasets), influence model path, `run_train` |
| `eval` | lm-eval | `run_eval` (lm-evaluation-harness) |
| `dev` | pytest | `pytest` |

If an optional extra is missing, the corresponding stage degrades gracefully:
standardizers and `run_train` / `run_eval` print a hint and **exit code 3 = skipped**
(the model-free influence proxy keeps selection runnable), so a partial install still
demonstrates the closed loop.

## 1. Pure-CPU track (seconds, no downloads, no torch)

### 1a. One-shot smoke

```bash
python scripts/sanity_smoke.py
```

This builds a tiny standardized pool (general/math/code with planted
near-duplicates), runs the full system path, contrasts
`All vs Random vs Influence-only(Top-K) vs MMDataSelect`, writes
`outputs/sanity_smoke/manifests/`, and asserts basic invariants (selected count ==
budget; the diversity term does not increase set-redundancy vs Influence-only).

### 1b. Make a demo pool, then select

```bash
# stand-in for the real standardizers: writes data/processed/demo.jsonl
python tools/standardize/make_demo.py            # default --n 60

# select half the pool -> outputs/demo_select/manifests/{manifest.json,selected.jsonl}
python run_mmdataselect/run_select.py --config configs/experiments/demo_select.yaml
```

`demo_select.yaml` sets `influence_model: null`, so selection uses the model-free CPU
proxy — no torch required. Running `run_pipeline.py` on this config will then *skip*
train/eval (exit 3) but still produce the manifest, exercising the application layer
end to end.

## 2. Real track (SmolLM2-135M, language + math + code)

### 2a. Standardize the three sources, then merge

Each standardizer streams a Hugging Face dataset and writes a `UnifiedRecord` jsonl;
concatenate them into one processed pool (the config's `processed_path`):

```bash
python tools/standardize/fineweb.py  --n 1000 --out data/processed/fineweb.jsonl   # general
python tools/standardize/finemath.py --n 1000 --out data/processed/finemath.jsonl  # math
python tools/standardize/pyedu.py    --n 1000 --out data/processed/pyedu.jsonl     # code

cat data/processed/fineweb.jsonl data/processed/finemath.jsonl \
    data/processed/pyedu.jsonl > data/processed/text_pool.jsonl
```

These need `datasets` (the `train` extra) and network/cache access; offline or
gated-repo failures print a hint and exit 3 rather than crashing.

### 2b. Run the full pipeline (select -> train -> eval)

```bash
python run_mmdataselect/run_pipeline.py \
    --config configs/experiments/text_smollm135m_demo.yaml
# or the wrapper:
bash run_mmdataselect/run.sh --config configs/experiments/text_smollm135m_demo.yaml
```

This config uses `influence_model: HuggingFaceTB/SmolLM2-135M` (downstream per-sample
loss as the influence signal), `budget: { kind: fraction, value: 0.5 }`, trains
`SmolLM2-135M` on the selected subset, and evaluates `[arc_easy, gsm8k]` via lm-eval.
All artifacts land under `outputs/text_smollm135m_demo/`:
`manifests/`, `models/`, `eval/results.json`.

Run individual stages (each shares the same `--output-dir`):

```bash
python run_mmdataselect/run_select.py --config configs/experiments/text_smollm135m_demo.yaml
python run_mmdataselect/run_train.py  --config configs/experiments/text_smollm135m_demo.yaml
python run_mmdataselect/run_eval.py   --config configs/experiments/text_smollm135m_demo.yaml
```

### 2c. Compare against baselines

Run a baseline on the *same* config, then reuse train/eval by pointing
`--output-dir` at the baseline's output:

```bash
python baselines/random/run_random.py --config configs/experiments/text_smollm135m_demo.yaml
python run_mmdataselect/run_train.py \
    --config configs/experiments/text_smollm135m_demo.yaml \
    --output-dir outputs/text_smollm135m_demo_random
```

## 3. Hardware notes (Apple M2 / RTX 4070)

The real track is sized for a single 8-16 GB device; SmolLM2-135M is small enough to
train and to score influence on either.

- **Device selection is automatic.** `InfluenceSignal` and (via transformers)
  `run_train` pick `cuda` -> `mps` -> `cpu` in that order. No flag needed.
- **Apple M2 (MPS).**
  - Use a recent `torch` (`>=2.1`) for working MPS; install the arm64 wheel.
  - If you hit unimplemented MPS ops, fall back to CPU for that run:
    `PYTORCH_ENABLE_MPS_FALLBACK=1 python run_mmdataselect/run_pipeline.py ...`.
  - MPS has no `bf16`; keep the default fp32 for stability on 135M.
  - Keep `train.batch_size` small (4) and `max_length` at 512 to stay within unified
    memory; lower `eval.limit` if evaluation is slow.
- **RTX 4070 (12 GB CUDA).**
  - SmolLM2-135M trains comfortably in fp32 at `batch_size: 4`; you can raise it.
  - For lm-eval, `batch_size: auto` lets the harness pick a safe size; set an explicit
    integer if you see OOM during evaluation.
  - First run downloads the model + datasets to the HF cache; set `HF_HOME` to a disk
    with space and pre-warm with the standardize step.
- **Smoke first, scale later.** Start with small `--n` (e.g. 200-1000 per source) and
  `eval.limit: 200`. Once the loop is green, scale `--n` up. The pure-CPU track (Section 1)
  needs none of this and is the fastest way to confirm the install is sane.
