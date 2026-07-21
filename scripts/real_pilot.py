"""Real downstream pilot on a local M2 (SmolLM2-135M).

Uses *real* model signals end-to-end:
  1. load the cached real pool + held-out set (scripts/build_real_pool.py);
  2. compute real per-sample influence = SmolLM2-135M per-sample loss on the pool;
  3. select a budget-constrained subset with each method
     (full / random / dsir / mmdataselect);
  4. fine-tune a fresh SmolLM2-135M on each subset (manual loop, MPS);
  5. evaluate held-out perplexity.

Prints a comparison table and writes outputs/real_pilot/results.json. All numbers
are real (small-scale pilot); sizes are env-configurable for scaling up.
"""
from __future__ import annotations

import json
import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from mmdataselect.datatypes import UnifiedRecord  # noqa: E402
from mmdataselect.selectors.budget_select import BudgetSelector  # noqa: E402
from mmdataselect.signals import RedundancySignal, hashed_features, minmax, set_redundancy  # noqa: E402
from mmdataselect.utils.io import read_jsonl  # noqa: E402
from mmdataselect.utils.manifest import write_manifest  # noqa: E402

sys.path.insert(0, os.path.join(_REPO, "baselines/dsir"))
from method.dsir_select import dsir_select  # noqa: E402

MODEL = os.environ.get("PILOT_MODEL", "HuggingFaceTB/SmolLM2-135M")
MAXLEN = int(os.environ.get("PILOT_MAXLEN", "192"))
EPOCHS = int(os.environ.get("PILOT_EPOCHS", "2"))
BS = int(os.environ.get("PILOT_BS", "4"))
LR = float(os.environ.get("PILOT_LR", "5e-5"))
SEED = 0


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def per_sample_loss(model, tok, records, dev):
    model.eval()
    out = np.zeros(len(records))
    with torch.no_grad():
        for i, r in enumerate(records):
            enc = tok(r.text, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
            ids = enc["input_ids"]
            if ids.size(1) < 2:
                continue
            out[i] = float(model(input_ids=ids, labels=ids).loss.item())
    return out


def finetune(texts, dev):
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(dev).train()
    tok = TOK
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    enc = tok(texts, truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    n = ids.size(0)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, BS):
            b = perm[s : s + BS]
            x = ids[b].to(dev)
            m = mask[b].to(dev)
            labels = x.masked_fill(m == 0, -100)
            loss = model(input_ids=x, attention_mask=m, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
    return model


def heldout_ppl(model, tok, records, dev):
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    with torch.no_grad():
        for r in records:
            enc = tok(r.text, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
            ids = enc["input_ids"]
            if ids.size(1) < 2:
                continue
            ntok = ids.size(1) - 1
            tot_loss += float(model(input_ids=ids, labels=ids).loss.item()) * ntok
            tot_tok += ntok
    return math.exp(tot_loss / max(1, tot_tok))


def main() -> int:
    global TOK
    dev = device()
    pool = [UnifiedRecord.from_dict(d) for d in read_jsonl(os.path.join(_REPO, "data/processed/real_pool.jsonl"))]
    held = [UnifiedRecord.from_dict(d) for d in read_jsonl(os.path.join(_REPO, "data/processed/real_heldout.jsonl"))]
    n, k = len(pool), len(pool) // 2
    print(f"device={dev} | pool={n} | heldout={len(held)} | budget k={k} | model={MODEL} epochs={EPOCHS}")

    TOK = AutoTokenizer.from_pretrained(MODEL)
    if TOK.pad_token is None:
        TOK.pad_token = TOK.eos_token

    print("[1/3] real influence = SmolLM2 per-sample loss on the pool ...")
    base = AutoModelForCausalLM.from_pretrained(MODEL).to(dev)
    influence = per_sample_loss(base, TOK, pool, dev)
    del base
    red_score = RedundancySignal().score(pool)
    feats = hashed_features(pool)
    importance = 0.5 * minmax(influence) + 0.5 * minmax(red_score)

    ids = [r.id for r in pool]
    texts_pool = [r.text for r in pool]
    domains = [r.domain for r in pool]
    rng = np.random.default_rng(SEED)
    methods = {
        "full_data": list(range(n)),
        "random": list(rng.choice(n, size=k, replace=False)),
        "dsir": list(dsir_select(texts_pool, ids, k, domains=domains, seed=SEED)[0]),
        "mmdataselect": BudgetSelector(lam=0.5).select(pool, importance, k, features=feats),
    }

    results = []
    for name, idx in methods.items():
        idx = list(idx)
        sel = [pool[i] for i in idx]
        print(f"\n[2/3] train on '{name}' subset (n={len(idx)}) ...")
        model = finetune([r.text for r in sel], dev)
        print(f"[3/3] eval held-out perplexity for '{name}' ...")
        ppl = heldout_ppl(model, TOK, held, dev)
        row = {
            "method": name,
            "n_selected": len(idx),
            "set_redundancy": round(set_redundancy(sel), 4),
            "mean_influence": round(float(np.mean(influence[idx])), 4),
            "heldout_ppl": round(ppl, 3),
        }
        results.append(row)
        print(f"   -> {row}")
        del model

    print("\n================ REAL PILOT RESULTS ================")
    print(f"{'method':<14}{'n':>5}{'set_redundancy↓':>18}{'mean_influence↑':>18}{'heldout_ppl↓':>16}")
    for r in results:
        print(f"{r['method']:<14}{r['n_selected']:>5}{r['set_redundancy']:>18}{r['mean_influence']:>18}{r['heldout_ppl']:>16}")

    out_dir = os.path.join(_REPO, "outputs", "real_pilot")
    write_manifest(
        out_dir,
        experiment_id="real_pilot",
        method="mmdataselect",
        n_total=n,
        selected_ids=[pool[i].id for i in methods["mmdataselect"]],
        selected_rows=[pool[i].to_dict() for i in methods["mmdataselect"]],
        extra={"results": results, "config": {"model": MODEL, "epochs": EPOCHS, "k": k, "maxlen": MAXLEN, "lr": LR}},
    )
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"results": results, "config": {"model": MODEL, "epochs": EPOCHS, "k": k, "n": n}}, f, indent=2)
    print(f"\nsaved -> {os.path.join(out_dir, 'results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
