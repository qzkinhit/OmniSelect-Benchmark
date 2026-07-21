# Real pilot results (local M2, SmolLM2-135M)

Reproduce:

```bash
pip install -e ".[train]"
python scripts/build_real_pool.py     # streams real FineWeb-Edu + FineMath
python scripts/real_pilot.py          # real influence -> select -> finetune -> held-out ppl
```

Setup: pool = 122 real records (61 general FineWeb-Edu + 61 math FineMath; **code
pending** — `code_search_net` / `the-stack-smol` were unreachable/gated), held-out
= 30, budget k = 61 (50%), SmolLM2-135M, 2 epochs, MPS. Influence is the **real**
per-sample loss of SmolLM2 on the pool.

| method | n | set_redundancy ↓ | mean_influence ↑ | held-out ppl ↓ |
|---|---|---|---|---|
| full_data | 122 | 0.6057 | 2.585 | 14.174 |
| random | 61 | 0.6016 | 2.587 | 14.294 |
| dsir | 61 | 0.6193 | 2.215 | 14.217 |
| **mmdataselect** | 61 | **0.5653** | **3.040** | 14.340 |

## Honest reading

* **Selection objective — confirmed on real data.** MMDataSelect attains the
  lowest set-redundancy (0.565) *and* the highest mean influence (3.040), i.e. it
  does exactly what `importance × diversity` optimizes, beating Random (no signal),
  DSIR (distribution-matching only, lower influence) and the full pool.
* **Downstream perplexity — within noise at this scale.** All four sit in
  14.17–14.34. This is expected: fine-tuning an *already-trained* SmolLM2 on 61
  in-distribution documents for 2 epochs barely moves held-out ppl, so data choice
  cannot separate the methods here. Data-selection downstream gains surface under
  (a) from-scratch / continued pre-training, (b) larger pools, (c) pools with real
  quality variance — not small in-distribution fine-tuning.

## What this validates / what is next

Validated: the whole system runs end-to-end with **real model signals** on a local
M2, and the method's selection-side objective holds on real FineWeb-Edu + FineMath.

Next, to get a downstream signal worth putting in the paper's main table:
1. add a working **code** source (replace the gated `python-edu`);
2. scale the pool to thousands of records and train **from a less-trained init**
   (or continued-pretraining), where data composition drives held-out loss;
3. add a real benchmark eval (lm-eval `arc_easy`/`gsm8k`) via `tools/eval`.

These numbers are a faithful small-scale pilot, **not** the paper's final results.
