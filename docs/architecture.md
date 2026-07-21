# Architecture

OmniSelect is a **unified, modality-agnostic data selection system** for
downstream foundation models. This document describes the system's structure, its
data flow, and the single contract (`manifest`) that couples a selection method to
the train/eval stack. It is written to mirror the layout already in `README.md`.

## 1. System / application separation (the core discipline)

The most important invariant in this repo is a hard wall between **methods** and
**plumbing**:

- **System** (`src/mmdataselect/`) — pure methods. It knows nothing about CLI
  arguments, dataset paths, YAML configs, or evaluation harnesses. It operates only
  on `UnifiedRecord` objects and numpy arrays. Crucially, `import mmdataselect` must
  succeed **without torch/transformers/datasets/lm-eval installed**: heavy
  dependencies are imported lazily inside the one signal that needs them
  (`signals/influence.py`) and degrade gracefully to a model-free CPU proxy.
- **Application** (`run_mmdataselect/run_*.py`) — the runners. This is where
  `argparse`, YAML loading, path resolution, and file I/O live. A runner loads a
  processed jsonl pool, calls a system method (`select_pool`), and writes a manifest.
- **Baselines** (`baselines/<name>/`) — each is *someone else's system* kept
  self-contained (`method/`) plus a **symmetric runner** (`run_<name>.py`) that emits
  the *same* manifest format. The manifest is the only coupling; the methods stay
  independent.

The payoff: the same selection operator is reused across vision-language / math /
code, and every method (ours and each baseline) plugs into the shared
`run_train` / `run_eval` / `tools/eval` without modification.

```
src/mmdataselect/   core system (pure method; installable package, torch-free import)
run_mmdataselect/   application layer — run_select / run_train / run_eval / run_pipeline
baselines/<name>/   each baseline = its own method/ + a symmetric run_<name>.py
tools/              standardize (-> UnifiedRecord) · eval (lm-eval wrapper) · metrics
configs/            YAML for datasets / models / signals / experiments
data/               raw (gitignored) · processed (UnifiedRecord jsonl) · manifests
outputs/{exp_id}/   single output root: manifests · models · eval · logs
papers/mmdataselect/ problem statement · draft · references
docs/               architecture (this file) · reproduce
scripts/            sanity_smoke.py (pure-CPU end-to-end) · setup_env.sh
tests/              pytest
```

## 2. Data flow

The pipeline is a straight line; every arrow is a concrete artifact on disk or a
typed in-memory object, so any stage can be inspected or swapped.

```
  raw source (HF dataset / images / code)
        |  tools/standardize/<name>.py        (modality -> textualized content)
        v
  data/processed/<name>.jsonl                 (list[UnifiedRecord] as jsonl)
        |  run_mmdataselect/run_select.py  ->  mmdataselect.api.select_pool
        v
  +-----------------------------------------------------------------------+
  |  S I G N A L S   (modality-agnostic, per-record value in [0,1])        |
  |    redundancy : RedundancySignal  (Entropy-Law byte compression ratio) |
  |    influence  : InfluenceSignal   (downstream per-sample loss / proxy) |
  +-----------------------------------------------------------------------+
        |  MultiActorConsole  (fusion: theta <- theta + lr*(R_actor - mean_R))
        v
  importance = sum_a softmax(theta)_a * score_a            (fused per-record value)
        |  Budget.resolve(n, token_counts)  ->  k           (token-budget constraint)
        v
  BudgetSelector.select  (greedy facility-location: importance + lam*(1 - max_sim))
        |                                                    (importance x diversity)
        v
  SelectionResult { selected_idx, selected_ids, importance, weights, diagnostics }
        |  utils/manifest.write_manifest
        v
  outputs/<exp_id>/manifests/{manifest.json, selected.jsonl}
        |  run_train (HF Trainer SFT) -> outputs/<exp_id>/models/
        v  run_eval  (tools/eval -> lm-eval) -> outputs/<exp_id>/eval/results.json
  downstream metrics {task: score}
```

Stage by stage:

1. **Standardize** — `tools/standardize/<name>.py` maps any source (FineWeb-Edu,
   FineMath, Python-Edu; later image-text / MathVista) onto `UnifiedRecord`
   (`id / modality / domain / text / meta`) and writes `data/processed/<name>.jsonl`.
   This is the *single coupling between modalities*: everything downstream operates on
   the `.text` field and never branches on the original modality.
2. **Signals** — each `Signal.score(records)` returns a per-record value in `[0,1]`
   (higher = more valuable to keep), so signals are directly comparable and fusible.
   `RedundancySignal` is model-free CPU (byte compression ratio); `InfluenceSignal`
   uses the downstream model's per-sample loss, falling back to a deterministic
   lexical-richness proxy when no model / torch is available.
3. **Multi-Actor fusion** — `MultiActorConsole` treats each signal as an *actor* and
   fuses their `[0,1]` scores with softmax-normalized weights `theta`. Weights can be
   nudged toward actors whose measured gain on a small holdout exceeds the mean
   (`theta <- theta + lr*(R_actor - mean_R)`), so the mix adapts across stages/modalities.
4. **Budget selection** — `Budget.resolve(n, token_counts)` turns the config budget
   (`fraction` / `records` / `tokens`) into a concrete keep-count `k`; `BudgetSelector`
   then runs a greedy facility-location that trades importance against diversity
   (`gain_i = importance_i + lam * (1 - max_sim(i, selected))`), penalizing redundancy
   against the already-selected set. A `gumbel` variant is provided for ablations.
5. **Train** — `run_train.py` fine-tunes the downstream base model (default
   `SmolLM2-135M`) on `selected.jsonl` via the HuggingFace `Trainer`.
6. **Eval** — `run_eval.py` delegates to `tools/eval.evaluate_model`, a thin wrapper
   over lm-evaluation-harness, returning a flat `{task: metric}`.

## 3. The manifest contract

`mmdataselect.utils.manifest.write_manifest` is the **only** coupling between a
selection method and the train/eval stack. Every method — our `select_pool` and
every baseline — emits the same two files under `outputs/<exp_id>/manifests/`:

- `manifest.json`
  ```json
  {
    "experiment_id": "demo_select",
    "method": "mmdataselect",
    "n_total": 60,
    "n_selected": 30,
    "selected_ids": ["m0001", "c0002", "..."],
    "diagnostics": { "keep_ratio": 0.5, "set_redundancy_selected": 0.0, "...": "..." },
    "actor_weights": { "redundancy": 0.5, "influence": 0.5 }
  }
  ```
- `selected.jsonl` — the selected `UnifiedRecord` dicts (`run_train` reads `.text`).

The public signature is fixed:

```python
write_manifest(
    out_dir, *, experiment_id, method, n_total,
    selected_ids, selected_rows=None, extra=None,
) -> dict
```

`extra` is merged into the manifest top-level (we pass `diagnostics`,
`actor_weights`, and the `select_config` snapshot). Because the contract is fixed,
`run_train` / `run_eval` / `tools/eval` consume any method uniformly, and a baseline
can reuse our train/eval by pointing `--output-dir` at its own output directory.

## 4. Directory reference

| Path | Role |
|---|---|
| `src/mmdataselect/datatypes.py` | `UnifiedRecord` + `Modality` + `DOMAIN_*` — the modality-agnostic record |
| `src/mmdataselect/signals/` | `RedundancySignal`, `InfluenceSignal`, `hashed_features`, `set_redundancy` |
| `src/mmdataselect/fusion/console.py` | `MultiActorConsole` — dynamic signal fusion |
| `src/mmdataselect/selectors/budget_select.py` | `BudgetSelector` — importance x diversity greedy |
| `src/mmdataselect/budget.py` | `Budget.from_cfg` / `.resolve` — token-budget -> keep-count |
| `src/mmdataselect/api/select.py` | `select_pool` — the one-call system entry |
| `src/mmdataselect/utils/manifest.py` | `write_manifest` — the shared output contract |
| `run_mmdataselect/run_*.py` | application layer (argparse + YAML + I/O) |
| `tools/standardize/` | source -> `UnifiedRecord` jsonl (FineWeb / FineMath / Python-Edu / demo) |
| `tools/eval/harness.py` | lm-evaluation-harness wrapper (`evaluate_model`) |
| `baselines/{random,full_data,dsir}/` | reference methods, same manifest format |
| `configs/experiments/` | per-experiment YAML (`demo_select`, `text_smollm135m_demo`) |
| `outputs/<exp_id>/` | single output root: `manifests/`, `models/`, `eval/`, logs |

## 5. Cross-modal selection contract

OmniSelect measures data value relative to the downstream model and task rather
than assuming a universal, model-free quality score. A common budget and manifest
contract lets heterogeneous strategies select records from the same frozen pool,
after which the same training and evaluation stack measures downstream utility.
This separation keeps the selection logic modality-agnostic while allowing each
runner to preserve its task-specific model, metric, split, and preprocessing.
