"""Scaled experiment: does selection move downstream loss on a quality-variance pool?

Pipeline (all methods share one influence pass + one token budget):
  1. influence = pretrained SmolLM2-135M per-sample loss on the pool (cached);
  2. each method ranks the pool; we cut the ranking at a fixed TOKEN budget
     (equal tokens across methods -> "fixed token budget", per the paper);
  3. train a fresh from-scratch mini LM (Llama-family, ~tens of M params) on the
     selected text for a fixed number of token-passes;
  4. report per-modality held-out perplexity (math / code / general).

Methods: noselect (upper ref, all tokens) | random | influence_only | dedup_only |
dsir | balance (static 0.5/0.5 fusion) | mmdataselect (dynamic conflict-aware console).

Sizes are env-configurable (small defaults for a fast verify; scale up for the
overnight run). Real numbers only; nothing is hard-coded.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM  # noqa: E402

from mmdataselect.selectors.external_baselines import (perpcorr_select, quadmix,
                                                       quadmix_published_core, regmix_mixture)
from mmdataselect.datatypes import UnifiedRecord  # noqa: E402
from mmdataselect.fusion.console import MultiActorConsole  # noqa: E402
from mmdataselect.selectors.budget_select import BudgetSelector  # noqa: E402
from mmdataselect.signals import AuthenticitySignal, InfluenceSignal, RedundancySignal, hashed_features, minmax  # noqa: E402
from mmdataselect.signals.redundancy import set_redundancy  # noqa: E402
from mmdataselect.utils.io import read_jsonl  # noqa: E402
from baselines.dsir.method.dsir_select import dsir_select  # noqa: E402
from baselines.zip.method import select as zip_select  # noqa: E402
from baselines.if_mates.method import select as if_select  # noqa: E402

REF_MODEL = os.environ.get("REF_MODEL", "HuggingFaceTB/SmolLM2-135M")
# Method-v3 binds model and tokenizer to one immutable upstream snapshot.  The
# legacy path deliberately keeps its historical ``revision=main`` behaviour.
FROZEN_TEXT_MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
REF_MODEL_REVISION = os.environ.get(
    "REF_MODEL_REVISION", FROZEN_TEXT_MODEL_REVISION
)
BUDGET_FRAC = float(os.environ.get("BUDGET_FRAC", "0.5"))
PASSES = float(os.environ.get("PASSES", "3"))
CTX = int(os.environ.get("CTX", "512"))
HID = int(os.environ.get("MINI_HID", "320"))
LAYERS = int(os.environ.get("MINI_LAYERS", "6"))
HEADS = int(os.environ.get("MINI_HEADS", "5"))
LR = float(os.environ.get("LR", "3e-4"))
BS = int(os.environ.get("BS", "16"))
SEED = int(os.environ.get("SEED", "0"))
LAM = float(os.environ.get("LAM", "0.5"))
W_INFL = float(os.environ.get("W_INFL", "0.5"))  # importance = W_INFL*influence + (1-W_INFL)*redundancy
AUTH_Q = float(os.environ.get("AUTH_Q", "0.25"))  # authenticity prerequisite filter: drop bottom-q (garbage)
METHODS = os.environ.get("METHODS", "noselect,random,influence_only,dedup_only,dsir,balance,mmdataselect").split(",")
# Training paradigm: "scratch" = from-scratch mini-LM (clean but token-hungry / under-
# trained at small budgets); "finetune" = continue-train a pretrained base (REF_MODEL),
# the standard quality-sensitive selection eval used by DSIR/MATES/DoReMi.
TRAIN_MODE = os.environ.get("TRAIN_MODE", "scratch")
FT_LR = float(os.environ.get("FT_LR", "2e-5"))         # fine-tune learning rate
FT_STEPS_CAP = int(os.environ.get("FT_STEPS_CAP", "0"))  # cap fine-tune steps (0 = token-budget driven)
FT_FREEZE = int(os.environ.get("FT_FREEZE", "0"))      # freeze embed + lowest N layers for speed
# STRATIFIED budget: give EACH modality its own BUDGET_FRAC share of tokens (selected
# within-modality by the method's own ranking), then train one shared model on the
# union. Keeps the proven multi-modal token scale while removing the cross-modal budget
# starvation that a single global ranking causes (hard modalities like code lose out).
STRATIFY = os.environ.get("STRATIFY", "0") == "1"
METHOD_V3 = os.environ.get("METHOD_V3", "0") == "1"
METHOD_V3_DELTA = float(os.environ.get("METHOD_V3_DELTA", "0.05"))
METHOD_V3_TEXT_CLIP = float(os.environ.get("METHOD_V3_TEXT_CLIP", "1.0"))


def _methodv3_pretrained_kwargs():
    return {"revision": REF_MODEL_REVISION} if METHOD_V3 else {}


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def per_sample_loss(model, tok, records, dev, maxlen=CTX):
    model.eval()
    out = np.zeros(len(records))
    with torch.no_grad():
        for i, r in enumerate(records):
            enc = tok(r.text, truncation=True, max_length=maxlen, return_tensors="pt").to(dev)
            ids = enc["input_ids"]
            if ids.size(1) >= 2:
                out[i] = float(model(input_ids=ids, labels=ids).loss.item())
    return out


def extract_text_transfer_features(model, tok, records, dev, maxlen=CTX):
    """Mean-pooled frozen-LM representations for conservative text transfers."""
    model.eval()
    hidden_size = int(model.config.hidden_size)
    features = np.zeros((len(records), hidden_size), dtype=np.float32)
    batch_size = int(os.environ.get("TEXT_EMBED_BS", "32"))
    if batch_size <= 0:
        raise ValueError("TEXT_EMBED_BS must be positive")
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            encoded = tok(
                [record.text for record in batch],
                padding=True,
                truncation=True,
                max_length=maxlen,
                return_tensors="pt",
            ).to(dev)
            output = model.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                use_cache=False,
            )
            hidden = output.last_hidden_state.float()
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp_min(1e-12)
            features[start : start + len(batch)] = (
                pooled.cpu().numpy().astype(np.float32, copy=False)
            )
            completed = start + len(batch)
            if completed == len(records) or completed % (batch_size * 10) == 0:
                elapsed = max(time.time() - started, 1e-9)
                eta = elapsed / completed * (len(records) - completed)
                print(
                    f"[text-transfer] embedded {completed}/{len(records)} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
    return features


def grad_align_influence(model, tok, records, refs_by_dom, dev, maxlen=CTX):
    """LESS/MATES-style influence with a *per-modality* clean reference.

    score(x) = cos(grad L(x), grad L(ref_domain(x))); higher = training on x reduces
    the loss on x's own modality's clean reference more = more valuable & on-target.
    Gradients restricted to the last decoder layer to stay cheap on M2.
    """
    model.train()
    params = list(model.model.layers[-1].parameters())

    def sample_grad(text):
        enc = tok(text, truncation=True, max_length=maxlen, return_tensors="pt").to(dev)
        ids = enc["input_ids"]
        if ids.size(1) < 2:
            return None
        model.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids).loss
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        return torch.cat([g.detach().flatten() for g in grads if g is not None])

    g_ref = {}
    for dom, refs in refs_by_dom.items():
        acc = None
        for r in refs:
            g = sample_grad(r.text)
            if g is None:
                continue
            acc = g.clone() if acc is None else acc + g
        if acc is not None:
            g_ref[dom] = acc / (acc.norm() + 1e-8)

    out = np.zeros(len(records))
    for i, r in enumerate(records):
        gr = g_ref.get(r.domain)
        if gr is None:
            continue
        g = sample_grad(r.text)
        if g is None:
            continue
        out[i] = float((g @ gr).item() / (g.norm().item() + 1e-8))
    model.zero_grad(set_to_none=True)
    return out


def make_blocks(texts, tok):
    ids = []
    for t in texts:
        ids.extend(tok(t).input_ids + [tok.eos_token_id])
    nb = len(ids) // CTX
    if nb == 0:
        return torch.empty(0, CTX, dtype=torch.long)
    arr = np.array(ids[: nb * CTX], dtype=np.int64).reshape(nb, CTX)
    return torch.from_numpy(arr)


def _new_model(dev):
    """Fresh model per TRAIN_MODE: from-scratch mini-LM, or a pretrained base to
    continue-train/fine-tune (the standard, quality-sensitive selection eval)."""
    if TRAIN_MODE == "finetune":
        m = AutoModelForCausalLM.from_pretrained(
            REF_MODEL, **_methodv3_pretrained_kwargs()
        ).to(dev).train()
        if FT_FREEZE > 0:  # optionally freeze the lowest layers for speed; top layers carry adaptation
            for p in m.model.embed_tokens.parameters():
                p.requires_grad_(False)
            for layer in m.model.layers[:FT_FREEZE]:
                for p in layer.parameters():
                    p.requires_grad_(False)
        return m
    cfg = LlamaConfig(
        vocab_size=TOK.vocab_size, hidden_size=HID, intermediate_size=HID * 4,
        num_hidden_layers=LAYERS, num_attention_heads=HEADS, num_key_value_heads=HEADS,
        max_position_embeddings=CTX, tie_word_embeddings=True, bos_token_id=TOK.bos_token_id,
        eos_token_id=TOK.eos_token_id,
    )
    return LlamaForCausalLM(cfg).to(dev).train()


def train_mini(blocks, dev, total_tokens):
    torch.manual_seed(SEED)
    model = _new_model(dev)
    lr = FT_LR if TRAIN_MODE == "finetune" else LR
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    steps_target = max(1, int(total_tokens / (CTX * BS)))
    if TRAIN_MODE == "finetune" and FT_STEPS_CAP > 0:
        steps_target = min(steps_target, FT_STEPS_CAP)  # few steps suffice from a pretrained base
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps_target, pct_start=0.1)
    g = torch.Generator().manual_seed(SEED)
    n = blocks.size(0)
    step = 0
    while step < steps_target:
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, BS):
            if step >= steps_target:
                break
            b = blocks[perm[s : s + BS]].to(dev)
            loss = model(input_ids=b, labels=b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
    return model, steps_target


def heldout_ppl(model, tok, records, dev):
    model.eval()
    tl, tt = 0.0, 0
    with torch.no_grad():
        for r in records:
            enc = tok(r.text, truncation=True, max_length=CTX, return_tensors="pt").to(dev)
            ids = enc["input_ids"]
            if ids.size(1) < 2:
                continue
            nt = ids.size(1) - 1
            tl += float(model(input_ids=ids, labels=ids).loss.item()) * nt
            tt += nt
    return math.exp(tl / max(1, tt)) if tt else float("nan")


def per_record_mean_nll(model, tok, records, dev):
    """Ordered record-level evidence; tokens are not treated as iid units."""
    model.eval()
    values, counts = [], []
    with torch.no_grad():
        for r in records:
            enc = tok(r.text, truncation=True, max_length=CTX, return_tensors="pt").to(dev)
            ids = enc["input_ids"]
            if ids.size(1) < 2:
                raise ValueError(f"record {r.id!r} has no prediction token")
            values.append(float(model(input_ids=ids, labels=ids).loss.item()))
            counts.append(int(ids.size(1) - 1))
    return values, counts


def _canonical_sha(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def split_methodv3_text_controller_records(records, seed):
    """ID-only deterministic V1/V2 split of the historical controller slice."""
    ordered = sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"methodv3-text-split-v1|{int(seed)}|{record.id}".encode("utf-8")
        ).digest(),
    )
    cut = len(ordered) // 2
    if cut == 0 or cut == len(ordered):
        raise ValueError("controller records cannot form non-empty V1 and V2")
    return ordered[:cut], ordered[cut:]


def _state_sha256(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        if tensor.dtype == torch.bfloat16:
            # NumPy has no native bfloat16 dtype.  Hash the exact underlying
            # two-byte representation while preserving the dtype label; this
            # changes no weights and keeps float32/float16 hashes unchanged.
            dtype_name = "bfloat16"
            shape = tuple(tensor.shape)
            raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        else:
            arr = tensor.numpy()
            dtype_name = str(arr.dtype)
            shape = tuple(arr.shape)
            raw = arr.tobytes(order="C")
        h.update(name.encode("utf-8") + b"\0")
        h.update(dtype_name.encode("ascii") + b"\0")
        h.update(str(shape).encode("ascii") + b"\0")
        h.update(raw)
    return h.hexdigest()


def make_exact_blocks(selected_indices, records, tok, effective_token_cap):
    """Build the exact same number of training tokens for every v3 candidate."""
    ids = []
    for index in selected_indices:
        ids.extend(tok(records[index].text).input_ids + [tok.eos_token_id])
        if len(ids) >= effective_token_cap:
            break
    if len(ids) < effective_token_cap:
        raise ValueError("selected token stream is shorter than the frozen effective-token cap")
    ids = ids[:effective_token_cap]
    if len(ids) % CTX:
        raise ValueError("effective-token cap must be divisible by CTX")
    array = np.asarray(ids, dtype=np.int64)
    return torch.from_numpy(array.reshape(-1, CTX)), hashlib.sha256(array.astype("<i8").tobytes()).hexdigest()


def make_exact_blocks_by_domain(selected_indices, records, tok, domain_caps):
    chunks = []
    for domain in sorted(domain_caps):
        indices = [index for index in selected_indices if records[index].domain == domain]
        ids = []
        for index in indices:
            ids.extend(tok(records[index].text).input_ids + [tok.eos_token_id])
            if len(ids) >= domain_caps[domain]:
                break
        if len(ids) < domain_caps[domain]:
            raise ValueError(f"selected stream for domain {domain!r} is below its frozen cap")
        chunks.extend(ids[: domain_caps[domain]])
    array = np.asarray(chunks, dtype=np.int64)
    if len(array) == 0 or len(array) % CTX:
        raise ValueError("stratified effective token stream must be non-empty CTX blocks")
    return torch.from_numpy(array.reshape(-1, CTX)), hashlib.sha256(array.astype("<i8").tobytes()).hexdigest()


def train_mini_v3(blocks, dev, total_tokens):
    """Train once while recording initial state and the complete block permutation."""
    torch.manual_seed(SEED)
    model = _new_model(dev)
    initial_state_sha256 = _state_sha256(model)
    lr = FT_LR if TRAIN_MODE == "finetune" else LR
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    steps_target = max(1, int(total_tokens / (CTX * BS)))
    if TRAIN_MODE == "finetune" and FT_STEPS_CAP > 0:
        steps_target = min(steps_target, FT_STEPS_CAP)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps_target, pct_start=0.1)
    generator = torch.Generator().manual_seed(SEED)
    n = blocks.size(0)
    step, complete_order = 0, []
    while step < steps_target:
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, BS):
            if step >= steps_target:
                break
            batch_ids = perm[start : start + BS]
            complete_order.extend(int(v) for v in batch_ids)
            batch = blocks[batch_ids].to(dev)
            loss = model(input_ids=batch, labels=batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
    return model, {
        "steps": steps_target,
        "initial_state_sha256": initial_state_sha256,
        "training_block_order_sha256": _canonical_sha(complete_order),
        "training_block_order": complete_order,
        "optimizer_config_sha256": _canonical_sha(
            {
                "optimizer": "AdamW",
                "lr": lr,
                "betas": [0.9, 0.95],
                "weight_decay": 0.1,
                "scheduler": "OneCycleLR",
                "pct_start": 0.1,
                "steps_target": steps_target,
                "batch_size": BS,
                "ctx": CTX,
            }
        ),
    }


def rank_to_budget(order, tok_counts, budget):
    sel, used = [], 0
    for i in order:
        sel.append(i)
        used += tok_counts[i]
        if used >= budget:
            break
    return sel


def main() -> int:
    global TOK
    dev = device()
    pool = [UnifiedRecord.from_dict(d) for d in read_jsonl(os.path.join(_REPO, "data/processed/qpool_train.jsonl"))]
    held = [UnifiedRecord.from_dict(d) for d in read_jsonl(os.path.join(_REPO, "data/processed/qpool_heldout.jsonl"))]
    # PER-MODALITY mode: restrict the whole experiment to one modality so selection
    # and evaluation happen within that modality's own pool/budget/held-out — i.e.
    # "data selection for that modality's own base model" (the paper's actual setting).
    ONLY = os.environ.get("ONLY_DOMAIN", "")
    dom_mask = None
    if ONLY:
        dom_mask = np.array([r.domain == ONLY for r in pool])
        pool = [r for r in pool if r.domain == ONLY]
        held = [r for r in held if r.domain == ONLY]
    n = len(pool)
    TOK = AutoTokenizer.from_pretrained(
        REF_MODEL, **_methodv3_pretrained_kwargs()
    )
    if TOK.pad_token is None:
        TOK.pad_token = TOK.eos_token

    tok_counts = [len(TOK(r.text, truncation=True, max_length=CTX).input_ids) for r in pool]
    total_tok = sum(tok_counts)
    budget = int(BUDGET_FRAC * total_tok)
    pool_doms = sorted({r.domain for r in pool})
    dom_budget = {d: int(BUDGET_FRAC * sum(tok_counts[i] for i in range(n) if pool[i].domain == d)) for d in pool_doms}
    by_dom_all = {}
    for r in held:
        by_dom_all.setdefault(r.domain, []).append(r)
    held_by_dom, refs_by_dom, ctrl_by_dom = {}, {}, {}
    v1_by_dom, v2_by_dom = {}, {}
    for d, rs in by_dom_all.items():
        half = max(1, len(rs) // 2)
        refs_by_dom[d] = rs[:half]      # clean per-modality reference for grad-alignment
        rest = rs[half:]
        q = max(1, len(rest) // 2)
        ctrl_by_dom[d] = rest[:q]       # controller ADJUDICATION split (disjoint from report)
        held_by_dom[d] = rest[q:]       # disjoint eval set for held-out ppl (REPORT split)
        if METHOD_V3:
            # The historical controller split is deterministically divided without
            # using text or outcomes.  REPORT remains byte-for-byte the old suffix.
            v1_by_dom[d], v2_by_dom[d] = split_methodv3_text_controller_records(
                ctrl_by_dom[d], SEED
            )
    doms = sorted(held_by_dom)
    print(f"device={dev} | pool={n} ({total_tok} tok) | budget={budget} tok | heldout={ {d: len(v) for d, v in held_by_dom.items()} }")
    print(f"mini-LM: hid={HID} layers={LAYERS} ctx={CTX} | passes={PASSES} | train_mode={TRAIN_MODE} | methods={METHODS}")
    if METHOD_V3:
        if TRAIN_MODE != "finetune":
            raise ValueError("METHOD_V3 text evidence requires TRAIN_MODE=finetune")
        if os.environ.get("SELECT_ONLY", "0") == "1" or os.environ.get(
            "FROZEN_SELECTION_REPLAY_JSON", ""
        ):
            raise ValueError("text v3 select-only/replay is not implemented; refusing ambiguous evidence")
        if not math.isfinite(METHOD_V3_DELTA) or not 0.0 < METHOD_V3_DELTA < 1.0:
            raise ValueError("METHOD_V3_DELTA must lie in (0, 1)")
        if not math.isfinite(METHOD_V3_TEXT_CLIP) or METHOD_V3_TEXT_CLIP <= 0.0:
            raise ValueError("METHOD_V3_TEXT_CLIP must be positive")
        probe_references = tuple(
            value.strip()
            for value in os.environ.get(
                "METHOD_V3_TEXT_REFERENCE_METHODS", ""
            ).split(",")
            if value.strip()
        )
        probe_challengers = tuple(
            value.strip()
            for value in os.environ.get(
                "METHOD_V3_TEXT_CHALLENGER_METHODS", ""
            ).split(",")
            if value.strip()
        )
        probe_table_baselines = tuple(
            value.strip()
            for value in os.environ.get(
                "METHOD_V3_TEXT_TABLE_BASELINES", ""
            ).split(",")
            if value.strip()
        )
        probe_skipped = tuple(
            value.strip()
            for value in os.environ.get(
                "METHOD_V3_TEXT_SKIPPED_METHODS", ""
            ).split(",")
            if value.strip()
        )
        if not probe_references or not probe_challengers:
            raise ValueError("method-v3 reference/challenger registries must be non-empty")
        if set(probe_references) & set(probe_challengers):
            raise ValueError("reference and challenger portfolios must be disjoint")
        if set(METHODS) != set(probe_references) | set(probe_challengers):
            raise ValueError("METHODS must exactly equal the two frozen v3 portfolios")
        if not probe_table_baselines or not probe_skipped:
            raise ValueError("table-baseline and skipped-method registries must be non-empty")
        if len(set(probe_table_baselines)) != len(probe_table_baselines):
            raise ValueError("table-baseline registry contains duplicates")
        if set(METHODS) & set(probe_skipped):
            raise ValueError("a skipped text baseline cannot also be scheduled")
        if set(probe_table_baselines) != set(METHODS) | set(probe_skipped):
            raise ValueError(
                "table-baseline registry must equal scheduled plus skipped methods"
            )
        probe_max_portfolio = int(
            os.environ.get("METHOD_V3_TEXT_MAX_PORTFOLIO", "8")
        )
        if len(METHODS) > probe_max_portfolio:
            raise ValueError("frozen text portfolio exceeds its configured maximum")
    if os.environ.get("CONFIG_PROBE", "0") == "1":
        # Validate the complete formal contract, then stop before cache loading,
        # scoring, model construction, or training. This is safe without a GPU.
        print("[config-probe] contract valid; exiting before scoring/training")
        return 0

    # ---- influence variants (cached): grad-align (primary) + raw-loss + ppl-quality ----
    # cache key = pool content hash + reference model + CTX + INFL_KIND (audit: a new
    # pool must never silently reuse the pilot pool's influence cache)
    import hashlib as _h
    _pool_sha = _h.sha256(open(os.path.join(_REPO, "data/processed/qpool_train.jsonl"), "rb").read()).hexdigest()[:12]
    _revision_cache_suffix = (
        f"_rev{REF_MODEL_REVISION[:12]}"
        if METHOD_V3 and REF_MODEL_REVISION != FROZEN_TEXT_MODEL_REVISION
        else ""
    )
    _ck = (
        f"{_pool_sha}_{REF_MODEL.split('/')[-1]}{_revision_cache_suffix}_"
        f"c{CTX}_{os.environ.get('INFL_KIND', 'grad')}"
    )
    full_cache = os.path.join(_REPO, f"data/processed/qpool_influence_{_ck}.npz")
    icache = os.path.join(_REPO, f"data/processed/qpool_influence_{_ck}_{ONLY}.npz") if ONLY else full_cache
    # Reuse the full-pool cache by subsetting to this modality when available.
    if ONLY and not os.path.exists(icache) and os.path.exists(full_cache) and dom_mask is not None:
        zf = np.load(full_cache)
        np.savez(icache, grad=zf["grad"][dom_mask], loss=zf["loss"][dom_mask])
    if os.path.exists(icache):
        z = np.load(icache)
        infl_grad, infl_loss = z["grad"], z["loss"]
        print(f"[influence] loaded cache ({len(infl_grad)})")
    else:
        ref = AutoModelForCausalLM.from_pretrained(
            REF_MODEL, **_methodv3_pretrained_kwargs()
        ).to(dev)
        infl_loss = per_sample_loss(ref, TOK, pool, dev)
        # grad-alignment is only needed for INFL_KIND in {grad,blend}; skip it for the
        # default pplq path to keep influence forward-only (much faster on large pools).
        if os.environ.get("INFL_KIND", "grad") in ("grad", "blend"):
            print("[influence] SmolLM2-135M: grad-alignment + per-sample loss ...")
            infl_grad = grad_align_influence(ref, TOK, pool, refs_by_dom, dev)
        else:
            print("[influence] SmolLM2-135M: per-sample loss (pplq, forward-only) ...")
            infl_grad = np.zeros_like(infl_loss)
        del ref
        np.savez(icache, grad=infl_grad, loss=infl_loss)
    infl_pplq = -infl_loss          # reference-ppl quality: low reference loss = clean/high-quality
    # INFLUENCE CHANNEL: which reference-model value signal drives importance.
    #   grad  = gradient-alignment (LESS/MATES style; needs backward pass)
    #   pplq  = reference-perplexity quality (DCLM/Ultra-FineWeb style; forward-only, cheaper)
    #   blend = min-max sum of both
    # Diagnostics show pplq is the more reliable per-modality quality signal (esp. code).
    INFL_KIND = os.environ.get("INFL_KIND", "grad")
    influence = {"grad": infl_grad, "pplq": infl_pplq,
                 "blend": minmax(infl_grad) + minmax(infl_pplq)}[INFL_KIND]
    text_transfer_methods = {
        "coverage_text",
        "herding_text",
        "density_text",
    }
    text_transfer_features = None
    text_transfer_cache = None
    if set(METHODS) & text_transfer_methods:
        transfer_key = _h.sha256(
            "|".join(
                [
                    _pool_sha,
                    REF_MODEL,
                    REF_MODEL_REVISION if METHOD_V3 else "main",
                    f"ctx={CTX}",
                    f"batch={int(os.environ.get('TEXT_EMBED_BS', '32'))}",
                    "schema=lm-embedding-transfer-v1",
                ]
            ).encode("utf-8")
        ).hexdigest()[:16]
        text_transfer_cache = os.path.join(
            _REPO,
            "data/processed",
            f"text_transfer_cache_{transfer_key}.npz",
        )
        if os.path.exists(text_transfer_cache):
            transfer_payload = np.load(text_transfer_cache)
            text_transfer_features = transfer_payload["features"]
            if len(text_transfer_features) != n:
                raise ValueError("text-transfer cache length mismatch")
            print(
                f"[text-transfer] loaded {os.path.basename(text_transfer_cache)}"
            )
        else:
            print("[text-transfer] frozen-LM mean-pooled embeddings")
            transfer_model = AutoModelForCausalLM.from_pretrained(
                REF_MODEL, **_methodv3_pretrained_kwargs()
            ).to(dev)
            text_transfer_features = extract_text_transfer_features(
                transfer_model, TOK, pool, dev
            )
            del transfer_model
            np.savez_compressed(
                text_transfer_cache,
                features=text_transfer_features,
            )
            if dev == "cuda":
                torch.cuda.empty_cache()
    red = RedundancySignal().score(pool)
    feats = hashed_features(pool)
    imp_static = W_INFL * minmax(influence) + (1 - W_INFL) * minmax(red)

    console = MultiActorConsole(
        [("redundancy", RedundancySignal()), ("influence", InfluenceSignal())],
        weights=np.log(np.array([1 - W_INFL, W_INFL]) + 1e-9),
        conflict_gate=True, anneal=0.6, group_key="domain", trust_region=0.5, ema_beta=0.3, min_weight=0.02,
    )
    S = np.stack([minmax(red), minmax(influence)], axis=0)
    imp_dynamic = console.importance(pool, scores=S, progress=0.5)

    # --- 3 orthogonal channels: authenticity (clean/on-domain) x influence x coverage ---
    auth = AuthenticitySignal().score(pool)
    console3 = MultiActorConsole(
        [("authenticity", AuthenticitySignal()), ("influence", InfluenceSignal()), ("redundancy", RedundancySignal())],
        weights=np.log(np.array([0.45, 0.45, 0.10]) + 1e-9),
        conflict_gate=True, anneal=0.6, group_key="domain", trust_region=0.5, ema_beta=0.3, min_weight=0.02,
    )
    S3 = np.stack([minmax(auth), minmax(influence), minmax(red)], axis=0)
    imp_auth = console3.importance(pool, scores=S3, progress=0.5)

    rng = np.random.default_rng(SEED)
    ids = [r.id for r in pool]
    texts = [r.text for r in pool]
    domains = [r.domain for r in pool]
    ref_texts = [r.text for d in refs_by_dom for r in refs_by_dom[d]]  # clean multi-modal DSIR target
    selection_metadata = {}
    dmf_probe_cache = {}

    # DISTRIBUTION-MATCH channel (borrow DSIR): per-record log p(target)/q(pool) on hashed
    # n-grams, higher = more like the clean held-out distribution. dmf (quality+redundancy)
    # and DSIR (distribution) each lack what the other has; fusing both is the improvement.
    from baselines.dsir.method.dsir_select import hashed_ngram_counts, importance_log_weights  # noqa: E402
    _Xd = hashed_ngram_counts(texts)
    _tgt = hashed_ngram_counts(ref_texts).sum(axis=0) if ref_texts else None
    distmatch = importance_log_weights(_Xd, np.zeros(n, dtype=bool), target_counts=_tgt)

    def _fuse(score_list, weights):
        """Conflict-aware dynamic fusion of pre-computed channel scores (argsort-ready)."""
        sigs = [(f"c{i}", RedundancySignal()) for i in range(len(score_list))]
        con = MultiActorConsole(sigs, weights=np.log(np.asarray(weights) + 1e-9),
                                conflict_gate=True, anneal=0.6, group_key="domain",
                                trust_region=0.5, ema_beta=0.3, min_weight=0.02)
        return con.importance(pool, scores=np.stack([minmax(s) for s in score_list], axis=0), progress=0.5)

    def _formal_budget_selection(order):
        if STRATIFY:
            selected = []
            for domain in pool_doms:
                domain_order = [
                    int(index)
                    for index in order
                    if pool[int(index)].domain == domain
                ]
                selected.extend(
                    rank_to_budget(
                        domain_order,
                        tok_counts,
                        dom_budget[domain],
                    )
                )
            return selected
        return rank_to_budget([int(index) for index in order], tok_counts, budget)

    def _dmf_construction_reward(order):
        """Downstream probe on construction-only records; V1/V2/REPORT stay sealed."""
        selected = _formal_budget_selection(order)
        selected_sha256 = _canonical_sha([str(pool[index].id) for index in selected])
        cached = dmf_probe_cache.get(selected_sha256)
        if cached is not None:
            return cached
        blocks = make_blocks([pool[index].text for index in selected], TOK)
        if blocks.numel() == 0:
            raise ValueError("DMF-pub construction probe received an empty token stream")
        probe_tokens = int(os.environ.get("DMF_PROBE_TOKENS", "150000"))
        model, steps = train_mini(blocks, dev, total_tokens=probe_tokens)
        log_ppl = [
            math.log(max(heldout_ppl(model, TOK, refs_by_dom[domain], dev), 1e-12))
            for domain in doms
        ]
        reward = -float(np.mean(log_ppl))
        dmf_probe_cache[selected_sha256] = reward
        del model, blocks
        if dev == "cuda":
            torch.cuda.empty_cache()
        print(
            f"    [dmf-pub/probe] reward={reward:.6f} "
            f"steps={steps} selection={selected_sha256[:12]}"
        )
        return reward

    def _dmf_published_text_order():
        """Eqs. 6--8 transfer with token-budgeted LM construction rewards."""
        scores = np.stack(
            [minmax(auth), minmax(influence), minmax(red)],
            axis=0,
        )
        rounds = int(os.environ.get("DMF_ROUNDS", "6"))
        eta = float(os.environ.get("DMF_ETA", "0.5"))
        theta = np.ones(scores.shape[0], dtype=np.float64) / scores.shape[0]
        actor_orders = [
            [int(index) for index in np.argsort(-scores[actor], kind="mergesort")]
            for actor in range(scores.shape[0])
        ]
        actor_rewards = np.asarray(
            [_dmf_construction_reward(order) for order in actor_orders],
            dtype=np.float64,
        )
        mean_actor_reward = float(actor_rewards.mean())
        best_order = None
        best_reward = -np.inf
        trace = []
        for step in range(rounds):
            fused = theta @ scores
            order = [
                int(index)
                for index in np.argsort(-fused, kind="mergesort")
            ]
            fused_reward = _dmf_construction_reward(order)
            if fused_reward > best_reward:
                best_reward = fused_reward
                best_order = order
            raw_next = theta + eta * (actor_rewards - mean_actor_reward)
            projected = np.maximum(raw_next, 0.0)
            if projected.sum() <= 1e-12:
                projected = np.ones_like(projected)
            projected /= projected.sum()
            trace.append(
                {
                    "round": step,
                    "theta": theta.tolist(),
                    "actor_rewards": actor_rewards.tolist(),
                    "fused_reward": fused_reward,
                    "projected_next": projected.tolist(),
                }
            )
            theta = projected
        if best_order is None:
            raise RuntimeError("DMF-pub failed to construct an ordering")
        selection_metadata["dmf_pub"] = {
            "fidelity": "published-update unified-token-budget transfer",
            "construction_split_only": True,
            "probe_tokens": int(os.environ.get("DMF_PROBE_TOKENS", "150000")),
            "rounds": rounds,
            "eta": eta,
            "best_reward": best_reward,
            "trace": trace,
        }
        return best_order

    def ranking(method):
        if method == "noselect":
            return list(range(n))
        if method == "random":
            return list(rng.permutation(n))
        if method in ("coverage_text", "herding_text"):
            from mmdataselect.selectors.text_transfers import (
                coverage_token_order,
                herding_token_order,
            )

            transfer_domains = domains if STRATIFY else ["all"] * n
            transfer_budgets = dom_budget if STRATIFY else {"all": budget}
            builder = (
                coverage_token_order
                if method == "coverage_text"
                else herding_token_order
            )
            return builder(
                text_transfer_features,
                transfer_domains,
                tok_counts,
                transfer_budgets,
            )
        if method == "density_text":
            from mmdataselect.selectors.external_baselines import density_select

            return list(
                density_select(
                    text_transfer_features,
                    n,
                    seed=SEED,
                )
            )
        if method == "dmf_pub":
            return _dmf_published_text_order()
        if method == "influence_only":
            return list(np.argsort(-influence))
        if method in ("regmix", "perpcorr"):
            # Construction-side probe: train a small mini-LM on a candidate subset and
            # score per-domain PPL on the SIGNAL/reference split only (refs_by_dom).
            # The adjudication split (ctrl) and the report split are never touched, so
            # the controller protocol (Prop 2 / Thm 4) is preserved.
            import math as _math
            dom_arr = np.array([r.domain for r in pool])
            probe_tok = int(os.environ.get("PROBE_TOKENS", "150000"))

            def _probe(sel_idx):
                sub = [pool[i] for i in sel_idx]
                blocks = make_blocks([r.text for r in sub], TOK)
                mdl, _ = train_mini(blocks, dev, total_tokens=probe_tok)
                logs = []
                for d in doms:
                    p = heldout_ppl(mdl, TOK, refs_by_dom[d], dev)
                    logs.append(_math.log(max(p, 1e-9)))
                del mdl
                return -float(np.mean(logs))          # higher = better

            if method == "perpcorr":
                # one random-subset probe -> per-domain gains drive the PPL-weighted allocation
                ridx = list(np.random.default_rng(SEED + 21).permutation(n)[: max(50, n // 5)])
                sub = [pool[i] for i in ridx]
                blocks = make_blocks([r.text for r in sub], TOK)
                mdl, _ = train_mini(blocks, dev, total_tokens=probe_tok)
                dgain = {}
                for d in doms:
                    p = heldout_ppl(mdl, TOK, refs_by_dom[d], dev)
                    dgain[d] = -_math.log(max(p, 1e-9))
                del mdl
                k_cnt = max(1, int(0.5 * n))
                sel = perpcorr_select(-infl_pplq, dom_arr, dgain, k_cnt)
                rest = [i for i in np.argsort(-infl_pplq) if i not in set(sel)]
                return [int(i) for i in sel] + [int(i) for i in rest]

            def _mix_eval(wts):
                rng2 = np.random.default_rng(SEED + 31)
                take = []
                k_cnt = max(20, n // 4)
                for d, w in wts.items():
                    idx = np.where(dom_arr == d)[0]
                    m2 = min(len(idx), max(1, int(round(w * k_cnt))))
                    take.extend(rng2.permutation(idx)[:m2].tolist())
                return _probe(take)

            mix = regmix_mixture([r.domain for r in pool], _mix_eval,
                                 n_probe=int(os.environ.get("REGMIX_PROBES", "8")), seed=SEED)
            order = []
            for d in sorted(mix, key=lambda d: -mix[d]):
                idx = np.where(dom_arr == d)[0]
                order.extend(idx[np.argsort(-infl_pplq[idx])].tolist())
            # interleave by mixture weight: emit domain blocks proportionally
            alloc = np.array([mix[d] for d in doms]); alloc = alloc / (alloc.sum() + 1e-9)
            per_dom = {d: list(np.where(dom_arr == d)[0][np.argsort(-infl_pplq[np.where(dom_arr == d)[0]])]) for d in doms}
            out, t = [], 0
            while any(per_dom[d] for d in doms):
                for j, d in enumerate(doms):
                    m2 = max(1, int(round(alloc[j] * 10)))
                    for _ in range(m2):
                        if per_dom[d]:
                            out.append(int(per_dom[d].pop(0)))
            return out
        if method == "influence_loss":   # ablation: raw per-sample loss (favors noise)
            return list(np.argsort(-infl_loss))
        if method == "quality_ppl":      # ablation: ppl-quality filter (Ultra-FineWeb style)
            return list(np.argsort(-infl_pplq))
        if method == "dedup_only":
            return BudgetSelector(lam=1.0).select(pool, np.zeros(n), n, features=feats)
        if method == "dsir":           # fair DSIR: target = clean multi-modal held-out reference
            return list(dsir_select(texts, ids, n, target_texts=ref_texts, seed=SEED)[0])
        if method == "dsir_mc":        # legacy/diagnostic: target = math+code pool rows (modality-biased)
            return list(dsir_select(texts, ids, n, domains=domains, seed=SEED)[0])
        if method == "zip":            # Entropy-Law / ZIP compression-greedy (model-free)
            # ZIP_CACHE=1 (ZIP_REUSE audit): zip_select is deterministic given the pool
            # (index tie-break, fixed zlib level, seed interface-only) and the pool file
            # is seed-independent, so the full greedy ordering is IDENTICAL across seeds.
            # Cache the exact ordering once; later seeds reuse it bit-for-bit. Selection
            # semantics untouched - cold path calls the ORIGINAL implementation.
            if os.environ.get("ZIP_CACHE", "0") == "1":
                import hashlib as _h
                _pool_file = os.path.join(_REPO, "data/processed/qpool_train.jsonl")
                _zip_impl = os.path.join(_REPO, "baselines/zip/method/zip_select.py")
                _fsha = lambda p: _h.sha256(open(p, "rb").read()).hexdigest()
                key_src = "|".join([
                    _fsha(_pool_file), _fsha(_zip_impl),
                    "k1=0.1", "k2=0.5", "level=6",              # zip_select frozen defaults
                    f"n={n}", f"stratify={os.environ.get('STRATIFY','0')}",
                    f"budget_frac={BUDGET_FRAC}", f"ctx={CTX}", f"ref={REF_MODEL}",
                    f"only={ONLY}"] + (
                        [f"revision={REF_MODEL_REVISION}"]
                        if METHOD_V3 and REF_MODEL_REVISION != FROZEN_TEXT_MODEL_REVISION
                        else []
                    ))
                key = _h.sha256(key_src.encode()).hexdigest()[:16]
                cpath = os.path.join(_REPO, "data/processed", f"zip_order_cache_{key}.json")
                if os.path.exists(cpath):
                    cached = json.load(open(cpath))
                    assert cached["n"] == n and len(cached["order"]) == n, "zip cache shape mismatch"
                    print(f"    [zip-cache] HIT {os.path.basename(cpath)} (exact ordering reuse)")
                    return [int(i) for i in cached["order"]]
                order = list(zip_select(pool, n))
                json.dump({"n": n, "key_src": key_src, "order": [int(i) for i in order]},
                          open(cpath, "w"))
                print(f"    [zip-cache] MISS -> computed once, saved {os.path.basename(cpath)}")
                return order
            return list(zip_select(pool, n))
        if method == "quadmix_pub":    # QuaDMix Eqs. 1-3 published-core (TEXT_LANE_HARD_STOP item 3)
            # REAL domain labels (the pool's five textualized domains, no k-means
            # substitute) + REAL token weights (tokenizer input_ids lengths). Eqs. 1-3
            # expected counts -> Gumbel-key full ordering; the shared downstream
            # fixed-token-budget truncation (identical for every method) is the
            # fixed-budget adapter. Frozen params per docs/published_method_fidelity_gate.md.
            dom_arr = np.asarray(domains)
            tw = np.maximum(np.asarray(tok_counts, dtype=float), 1.0)  # the SAME tokenizer
            # counts the budget consumes - token weights are real, not a proxy
            return list(quadmix_published_core(infl_pplq, feats, n, domains=dom_arr,
                                               token_weights=tw, seed=SEED))
        if method == "quadmix":        # QuaDMix-STYLE PROXY (quality buckets + farthest-first),
            # NOT the published sampler; retained under the "style proxy" label only.
            # The main text protocol uses a fair token budget inside every domain.  Build a
            # complete deterministic order independently per domain, so the generic
            # STRATIFY truncation below preserves that exact protocol.  Reference-PPL is
            # the available text-quality signal and hashed n-grams provide coverage.
            out = []
            dom_arr = np.asarray(domains)
            for d_i, d in enumerate(pool_doms):
                idx = np.where(dom_arr == d)[0]
                local = quadmix(
                    infl_pplq[idx], feats[idx], len(idx), alpha=0.5,
                    seed=SEED * 1009 + d_i,
                )
                out.extend(int(idx[j]) for j in local)
            return out
        if method == "if_mates":       # influence Top-K (shares the reference-PPL influence)
            return list(if_select(pool, n, influence=influence))
        if method == "dmf":            # basic dynamic multi-signal fusion (no upgrades; the surpassed control)
            basic = MultiActorConsole([("redundancy", RedundancySignal()), ("influence", InfluenceSignal())])
            return list(np.argsort(-basic.importance(pool, scores=np.stack([minmax(red), minmax(influence)], axis=0))))
        if method == "balance":
            return BudgetSelector(lam=LAM).select(pool, imp_static, n, features=feats)
        if method in ("mmdataselect", "fixed_fusion"):
            # Frozen fixed-fusion baseline: authenticity gate followed by the
            # preregistered influence/coverage weighting.
            thr = float(np.quantile(auth, AUTH_Q))
            imp_f = imp_dynamic.copy()
            imp_f[auth < thr] = -1e9       # drop garbage (truncation/template/cross-domain) up front
            return BudgetSelector(lam=LAM).select(pool, imp_f, n, features=feats)
        if method == "mmds_adiv":          # modality-ADAPTIVE diversity: broad modalities (image) get
            thr = float(np.quantile(auth, AUTH_Q))   # more diversity, narrow ones (code/table) get less
            imp_f = imp_dynamic.copy()
            imp_f[auth < thr] = -1e9
            order = []
            for d in pool_doms:                      # select within each modality with its own lam
                d_idx = [i for i in range(n) if pool[i].domain == d]
                Fd = feats[d_idx]
                msim = float((Fd @ Fd.T).mean())     # mean self-similarity: high = narrow distribution
                lam_d = float(np.clip(0.9 * (1.0 - msim), 0.15, 0.65))  # narrow -> low diversity
                sub = BudgetSelector(lam=lam_d).select([pool[i] for i in d_idx], imp_f[d_idx], len(d_idx), features=Fd)
                order += [d_idx[j] for j in sub]
            return order
        if method in ("mmds_lam100", "mmds_lam085", "mmds_lam065"):  # diversity-strength sweep (auth-gated)
            lam = {"mmds_lam100": 1.0, "mmds_lam085": 0.85, "mmds_lam065": 0.65}[method]
            thr = float(np.quantile(auth, AUTH_Q))
            imp_f = imp_dynamic.copy()
            imp_f[auth < thr] = -1e9
            if lam >= 1.0:
                return list(np.argsort(-imp_f))
            return BudgetSelector(lam=lam).select(pool, imp_f, n, features=feats)
        # ---- improved variants: soft fusion (no hard filter / no forced diversity),
        #      borrowing distribution-matching from DSIR + keeping our deep quality signals ----
        if method == "dmf_auth":           # dmf + soft authenticity as a 3rd channel
            return list(np.argsort(-_fuse([influence, red, auth], [0.40, 0.15, 0.45])))
        if method == "mmds_dist":          # quality + distribution-match + redundancy
            return list(np.argsort(-_fuse([influence, distmatch, red], [0.45, 0.45, 0.10])))
        if method == "mmds_v2":            # best-of-all: quality + dist-match + soft authenticity + redundancy
            return list(np.argsort(-_fuse([influence, distmatch, auth, red], [0.34, 0.30, 0.26, 0.10])))
        if method == "mmds_v2div":         # mmds_v2 + light diversity tie-break
            return BudgetSelector(lam=0.25).select(pool, _fuse([influence, distmatch, auth, red], [0.34, 0.30, 0.26, 0.10]), n, features=feats)
        if method == "mmds_v2flt":         # mmds_v2 + soft authenticity gate (drop only the bottom garbage decile)
            imp = _fuse([influence, distmatch, auth, red], [0.34, 0.30, 0.26, 0.10])
            thr = float(np.quantile(auth, 0.10))
            imp[auth < thr] = -1e9
            return list(np.argsort(-imp))
        if method in ("mmds_v3", "mmds_v3lo"):  # mmdataselect recipe + distribution-match channel
            lam = LAM if method == "mmds_v3" else 0.35
            imp = _fuse([influence, distmatch, red], [0.45, 0.35, 0.20])  # quality + dist-match + density
            thr = float(np.quantile(auth, AUTH_Q))
            imp[auth < thr] = -1e9                                        # keep authenticity prefilter
            return BudgetSelector(lam=lam).select(pool, imp, n, features=feats)  # keep diversity (image needs it)
        if method == "mmds_rankauth":      # ablation: authenticity as a ranking maximand (narrows distribution)
            return BudgetSelector(lam=LAM).select(pool, imp_auth, n, features=feats)
        if method == "mmds_noauth":        # ablation: no authenticity at all (influence x diversity)
            return BudgetSelector(lam=LAM).select(pool, imp_dynamic, n, features=feats)
        if method == "authenticity_only":  # authenticity channel alone (Top-K)
            return list(np.argsort(-auth))
        raise ValueError(method)

    if METHOD_V3:
        from mmdataselect.fusion.methodv3_text_terminal_adapter import (
            freeze_text_pair_manifest,
            run_methodv3_text_terminal_adapter,
        )

        def _declared_methods(env_name):
            values = tuple(
                value.strip()
                for value in os.environ.get(env_name, "").split(",")
                if value.strip()
            )
            if not values:
                raise ValueError(f"{env_name} must pre-register a non-empty portfolio")
            return values

        reference_methods = _declared_methods("METHOD_V3_TEXT_REFERENCE_METHODS")
        challenger_methods = _declared_methods("METHOD_V3_TEXT_CHALLENGER_METHODS")
        table_baselines = _declared_methods("METHOD_V3_TEXT_TABLE_BASELINES")
        skipped_methods = _declared_methods("METHOD_V3_TEXT_SKIPPED_METHODS")
        if set(reference_methods) & set(challenger_methods):
            raise ValueError("reference and challenger portfolios must be disjoint")
        if set(METHODS) != set(reference_methods) | set(challenger_methods):
            raise ValueError("METHODS must exactly equal the two frozen v3 portfolios")
        if set(METHODS) & set(skipped_methods):
            raise ValueError("a skipped text baseline cannot also be scheduled")
        if set(table_baselines) != set(METHODS) | set(skipped_methods):
            raise ValueError(
                "table-baseline registry must equal scheduled plus skipped methods"
            )
        max_portfolio = int(os.environ.get("METHOD_V3_TEXT_MAX_PORTFOLIO", "8"))
        if max_portfolio < 2 or len(METHODS) > max_portfolio:
            raise ValueError(
                "frozen text portfolio exceeds METHOD_V3_TEXT_MAX_PORTFOLIO; "
                "refusing unbounded CPU model residency"
            )
        forbidden = {"base", "noselect", "full", "mmds_adapt"}
        if set(METHODS) & forbidden:
            raise ValueError("base/full/noselect/recursive controller are not pair candidates")

        all_split_ids = {}
        for split_name, split in (
            ("construction", refs_by_dom),
            ("V1", v1_by_dom),
            ("V2", v2_by_dom),
            ("REPORT", held_by_dom),
        ):
            all_split_ids[split_name] = {
                domain: [str(record.id) for record in split[domain]] for domain in sorted(split)
            }
        flat_split_sets = {
            name: {record_id for values in domains_map.values() for record_id in values}
            for name, domains_map in all_split_ids.items()
        }
        split_names = tuple(flat_split_sets)
        for left_index, left in enumerate(split_names):
            for right in split_names[left_index + 1 :]:
                if flat_split_sets[left] & flat_split_sets[right]:
                    raise ValueError(f"text v3 splits overlap: {left} vs {right}")
        split_manifest = {
            "schema_version": "omniselect.methodv3-text-splits.v1",
            "seed": SEED,
            "split_ordered_ids": all_split_ids,
            "split_ordered_ids_sha256": {
                name: _canonical_sha(values) for name, values in all_split_ids.items()
            },
            "pairwise_disjoint": True,
            "historical_status": "DEVELOPMENT_ONLY_HISTORICAL_HELDOUT_ALREADY_OBSERVED",
        }
        split_manifest["split_manifest_sha256"] = _canonical_sha(split_manifest)

        effective_domain_caps = (
            {domain: max(CTX, (dom_budget[domain] // CTX) * CTX) for domain in pool_doms}
            if STRATIFY
            else None
        )
        effective_token_cap = (
            sum(effective_domain_caps.values())
            if effective_domain_caps is not None
            else max(CTX, (budget // CTX) * CTX)
        )
        pool_path = os.path.join(_REPO, "data/processed/qpool_train.jsonl")
        pool_sha256 = hashlib.sha256(open(pool_path, "rb").read()).hexdigest()
        runner_impl_sha256 = hashlib.sha256(
            open(os.path.abspath(__file__), "rb").read()
        ).hexdigest()
        selector_component_paths = {
            "external_baselines": "src/mmdataselect/selectors/external_baselines.py",
            "budget_selector": "src/mmdataselect/selectors/budget_select.py",
            "fusion_console": "src/mmdataselect/fusion/console.py",
            "signal_authenticity": "src/mmdataselect/signals/authenticity.py",
            "signal_influence": "src/mmdataselect/signals/influence.py",
            "signal_redundancy": "src/mmdataselect/signals/redundancy.py",
            "text_transfers": "src/mmdataselect/selectors/text_transfers.py",
        }
        selector_component_sha256 = {
            name: hashlib.sha256(
                open(os.path.join(_REPO, relative_path), "rb").read()
            ).hexdigest()
            for name, relative_path in selector_component_paths.items()
        }
        selector_bundle_sha256 = _canonical_sha(selector_component_sha256)
        lmeval_enabled = int(os.environ.get("LM_EVAL", "0"))
        lmeval_tasks = (
            tuple(
                task.strip()
                for task in os.environ.get(
                    "LMEVAL_TASKS",
                    "arc_easy,arc_challenge,hellaswag,openbookqa",
                ).split(",")
                if task.strip()
            )
            if lmeval_enabled
            else ()
        )
        lmeval_limit = int(os.environ.get("LMEVAL_LIMIT", "0"))
        lmeval_batch_size = int(os.environ.get("LMEVAL_BS", "16"))
        frozen_config = {
            "methods": METHODS,
            "method_v3": 1,
            "reference_methods": reference_methods,
            "challenger_methods": challenger_methods,
            "table_baselines": table_baselines,
            "skipped_methods": skipped_methods,
            "skipped_method_reasons": {
                "el2n": "classification-error score has no native autoregressive-text protocol",
                "grand": "classification-gradient norm has no native autoregressive-text protocol",
                "ccs": "depends on classification-style EL2N difficulty and pruning strata",
            },
            "max_portfolio": max_portfolio,
            "stratify": int(STRATIFY),
            "infl": INFL_KIND,
            "infl_kind": INFL_KIND,
            "budget_frac": BUDGET_FRAC,
            "effective_domain_caps": effective_domain_caps,
            "effective_token_cap": effective_token_cap,
            "passes": PASSES,
            "ctx": CTX,
            "batch_size": BS,
            "train_mode": TRAIN_MODE,
            "ref_model": REF_MODEL,
            "ref_model_revision": REF_MODEL_REVISION,
            "tokenizer_model": REF_MODEL,
            "tokenizer_revision": REF_MODEL_REVISION,
            "only_domain": ONLY,
            "pool_n": n,
            "pool_domains": pool_doms,
            "lmeval": lmeval_enabled,
            "lmeval_tasks": lmeval_tasks,
            "lmeval_limit": lmeval_limit,
            "lmeval_batch_size": lmeval_batch_size,
            "ft_lr": FT_LR,
            "ft_steps_cap": FT_STEPS_CAP,
            "ft_freeze": FT_FREEZE,
            "delta": METHOD_V3_DELTA,
            "text_clip": METHOD_V3_TEXT_CLIP,
            "text_embed_batch_size": int(
                os.environ.get("TEXT_EMBED_BS", "32")
            ),
            "report_all_candidates": int(
                os.environ.get("REPORT_ALL_CANDIDATES", "0")
            ),
            "lmeval_all_candidates": int(
                os.environ.get("LMEVAL_ALL_CANDIDATES", "0")
            ),
            "save_all_candidate_models": int(
                os.environ.get("SAVE_ALL_CANDIDATE_MODELS", "0")
            ),
            "text_transfer_protocols": (
                __import__(
                    "mmdataselect.selectors.text_transfers",
                    fromlist=["TEXT_TRANSFER_PROTOCOLS"],
                ).TEXT_TRANSFER_PROTOCOLS
                if set(METHODS) & text_transfer_methods
                else {}
            ),
        }
        config_sha256 = _canonical_sha(frozen_config)
        run_id = os.environ.get("RUN_ID", "")
        tag = "-".join(
            [
                f"run_id={run_id}" if run_id else "run_id=unset",
                f"stratify={int(STRATIFY)}",
                f"infl={INFL_KIND}",
                f"train={TRAIN_MODE}",
                "method_v3=1",
            ]
        )
        out_dir = os.path.join(_REPO, "outputs", "experiment", tag, f"seed_{SEED}")
        os.makedirs(out_dir, exist_ok=True)
        checkpoint_paths = {}

        def _run_lmeval(model):
            if not lmeval_enabled:
                return None
            import lm_eval
            from lm_eval.models.huggingface import HFLM

            lm_model = HFLM(
                pretrained=model,
                tokenizer=TOK,
                batch_size=lmeval_batch_size,
            )
            raw_result = lm_eval.simple_evaluate(
                model=lm_model,
                tasks=list(lmeval_tasks),
                limit=lmeval_limit or None,
                verbosity="ERROR",
            )
            metrics = {}
            for task in lmeval_tasks:
                task_result = raw_result["results"].get(task, {})
                metrics[task] = round(
                    float(
                        task_result.get(
                            "acc_norm,none",
                            task_result.get("acc,none", float("nan")),
                        )
                    ),
                    6,
                )
            return metrics

        v1_records = [record for domain in doms for record in v1_by_dom[domain]]
        v2_records = [record for domain in doms for record in v2_by_dom[domain]]

        def _domain_balanced_nll(nll, counts, records):
            by_domain = []
            for domain in doms:
                indices = [i for i, record in enumerate(records) if record.domain == domain]
                total = sum(counts[i] for i in indices)
                if total <= 0:
                    raise ValueError(f"domain {domain!r} has no valid prediction tokens")
                by_domain.append(sum(nll[i] * counts[i] for i in indices) / total)
            return float(np.mean(by_domain))

        # Phase A is outcome-free: construct every ordering and group exact token
        # streams before any V1 model is trained.  Phase B fits each equivalence
        # class once and retains only the current best reference/challenger states.
        candidate_specs, stream_groups = [], {}
        for method in METHODS:
            started = time.time()
            order = ranking(method)
            if len(order) != n or len(set(int(value) for value in order)) != n:
                raise ValueError(f"{method} did not produce an exact full ordering")
            if STRATIFY:
                selected = []
                for domain in pool_doms:
                    domain_order = [index for index in order if pool[index].domain == domain]
                    selected += rank_to_budget(domain_order, tok_counts, dom_budget[domain])
                blocks, token_stream_sha256 = make_exact_blocks_by_domain(
                    selected, pool, TOK, effective_domain_caps
                )
            else:
                selected = rank_to_budget(order, tok_counts, budget)
                blocks, token_stream_sha256 = make_exact_blocks(
                    selected, pool, TOK, effective_token_cap
                )
            selected_ids = [str(pool[index].id) for index in selected]
            selection_order_sha256 = _canonical_sha(selected_ids)
            selection_set_sha256 = _canonical_sha(sorted(selected_ids))
            role = "reference" if method in reference_methods else "challenger"
            spec = {
                "method": method,
                "role": role,
                "n_sel": len(selected),
                "tok_sel_raw": sum(tok_counts[index] for index in selected),
                "effective_train_tokens": effective_token_cap,
                "selection_indices": [int(value) for value in selected],
                "selected_record_ids": selected_ids,
                "selection_order_sha256": selection_order_sha256,
                "selection_set_sha256": selection_set_sha256,
                "token_stream_sha256": token_stream_sha256,
                "selector_impl_sha256": selector_bundle_sha256,
                "selector_metadata": selection_metadata.get(method, {}),
                "selector_config_sha256": _canonical_sha(
                    {
                        "method": method,
                        "selector_bundle_sha256": selector_bundle_sha256,
                        **frozen_config,
                    }
                ),
                "selection_secs": round(time.time() - started, 3),
            }
            candidate_specs.append(spec)
            stream_groups.setdefault(token_stream_sha256, []).append(spec)
            del blocks

        fits_by_stream, candidate_rows = {}, []
        best_by_role = {"reference": None, "challenger": None}
        for token_stream_sha256, members in stream_groups.items():
            representative = members[0]
            if STRATIFY:
                blocks, replayed_stream_sha256 = make_exact_blocks_by_domain(
                    representative["selection_indices"], pool, TOK, effective_domain_caps
                )
            else:
                blocks, replayed_stream_sha256 = make_exact_blocks(
                    representative["selection_indices"], pool, TOK, effective_token_cap
                )
            if replayed_stream_sha256 != token_stream_sha256:
                raise ValueError("selection-equivalence stream replay changed before fit")
            model, training_manifest = train_mini_v3(
                blocks, dev, total_tokens=effective_token_cap * PASSES
            )
            del blocks
            v1_nll, v1_token_counts = per_record_mean_nll(model, TOK, v1_records, dev)
            v1_score = _domain_balanced_nll(v1_nll, v1_token_counts, v1_records)
            report_ppl_all = None
            lmeval_all = None
            if os.environ.get("REPORT_ALL_CANDIDATES", "0") == "1":
                report_ppl_all = {
                    domain: round(
                        heldout_ppl(model, TOK, held_by_dom[domain], dev), 6
                    )
                    for domain in doms
                }
                if os.environ.get("LMEVAL_ALL_CANDIDATES", "0") == "1":
                    lmeval_all = _run_lmeval(model)
            checkpoint_path = None
            if os.environ.get("SAVE_ALL_CANDIDATE_MODELS", "0") == "1":
                checkpoint_path = os.path.join(
                    out_dir,
                    "checkpoints",
                    representative["method"],
                )
                os.makedirs(checkpoint_path, exist_ok=False)
                model.save_pretrained(checkpoint_path, safe_serialization=True)
                TOK.save_pretrained(checkpoint_path)
            equivalent = {
                "model": None,
                "v1_nll": v1_nll,
                "v1_token_counts": v1_token_counts,
                "training_manifest": training_manifest,
                "fit_owner": representative["method"],
                "report_ppl": report_ppl_all,
                "lmeval": lmeval_all,
                "checkpoint_path": checkpoint_path,
            }
            fits_by_stream[token_stream_sha256] = equivalent
            for spec in members:
                row = {
                    **spec,
                    "selection_equivalence_owner": representative["method"],
                    "v1_domain_balanced_mean_nll": v1_score,
                    "v1_ordered_record_ids_sha256": split_manifest[
                        "split_ordered_ids_sha256"
                    ]["V1"],
                    "v1_mean_nll": v1_nll,
                    "v1_prediction_token_counts": v1_token_counts,
                    "training_manifest": training_manifest,
                    "report_ppl": report_ppl_all,
                    "report_gmean_ppl": (
                        math.exp(
                            float(
                                np.mean(
                                    [
                                        math.log(max(value, 1e-12))
                                        for value in report_ppl_all.values()
                                    ]
                                )
                            )
                        )
                        if report_ppl_all is not None
                        else None
                    ),
                    "lmeval": lmeval_all,
                    "checkpoint_path": checkpoint_path,
                }
                if checkpoint_path is not None:
                    checkpoint_paths[spec["method"]] = checkpoint_path
                row["v1_evidence_sha256"] = _canonical_sha(
                    {
                        "method": row["method"],
                        "ordered_record_ids_sha256": row[
                            "v1_ordered_record_ids_sha256"
                        ],
                        "mean_nll": v1_nll,
                        "prediction_token_counts": v1_token_counts,
                    }
                )
                candidate_rows.append(row)
                incumbent = best_by_role[row["role"]]
                # Stable registry order is the frozen tie rule; an exact tie never
                # displaces the earlier candidate.
                if incumbent is None or v1_score < incumbent[
                    "v1_domain_balanced_mean_nll"
                ]:
                    best_by_role[row["role"]] = row
                print(
                    f"  [v3/V1] {row['method']:14} role={row['role']:10} "
                    f"score={v1_score:.6f} fit_owner={representative['method']}"
                )

            retained_streams = {
                row["token_stream_sha256"]
                for row in best_by_role.values()
                if row is not None
            }
            # A newly better stream can evict an older state.  At this boundary
            # there are never more than two resident CPU models.
            for stream_sha256, fit in fits_by_stream.items():
                if (
                    stream_sha256 != token_stream_sha256
                    and stream_sha256 not in retained_streams
                    and fit["model"] is not None
                ):
                    fit["model"] = None
            if token_stream_sha256 in retained_streams:
                equivalent["model"] = model.to("cpu")
            else:
                del model
            if dev == "cuda":
                torch.cuda.empty_cache()

        resident_count = sum(fit["model"] is not None for fit in fits_by_stream.values())
        if resident_count > 2:
            raise RuntimeError("text v3 retained more than two CPU model states")

        reference_row = best_by_role["reference"]
        challenger_row = best_by_role["challenger"]
        if reference_row is None or challenger_row is None:
            raise ValueError("V1 did not freeze one reference and one challenger")

        pair_manifest = freeze_text_pair_manifest(
            reference_arm=reference_row["method"],
            challenger_arm=challenger_row["method"],
            seed=SEED,
            pool_sha256=pool_sha256,
            v1_split_sha256=split_manifest["split_ordered_ids_sha256"]["V1"],
            reference_selector_sha256=reference_row["selector_config_sha256"],
            challenger_selector_sha256=challenger_row["selector_config_sha256"],
            reference_selection_sha256=reference_row["selection_order_sha256"],
            challenger_selection_sha256=challenger_row["selection_order_sha256"],
            reference_v1_evidence_sha256=reference_row["v1_evidence_sha256"],
            challenger_v1_evidence_sha256=challenger_row["v1_evidence_sha256"],
            effective_token_cap=effective_token_cap,
        )

        v2_evidence = {}
        for row in (reference_row, challenger_row):
            equivalent = fits_by_stream[row["token_stream_sha256"]]
            model = equivalent["model"].to(dev)
            nll, counts = per_record_mean_nll(model, TOK, v2_records, dev)
            equivalent["model"] = model.to("cpu")
            v2_evidence[row["method"]] = {"mean_nll": nll, "token_counts": counts}
            if dev == "cuda":
                torch.cuda.empty_cache()

        v2_ids = [str(record.id) for record in v2_records]
        v2_domains = [record.domain for record in v2_records]
        if reference_row["token_stream_sha256"] == challenger_row["token_stream_sha256"]:
            decision_record = {
                "mode": "method_v3",
                "metric": "mean_token_nll",
                "decision": "KEEP_REFERENCE",
                "switched": False,
                "selected_arm": reference_row["method"],
                "no_switch_reason": "IDENTICAL_TRAINING_TOKEN_STREAM",
                "pair_manifest_sha256": pair_manifest["pair_manifest_sha256"],
            }
        else:
            reference_evidence = v2_evidence[reference_row["method"]]
            challenger_evidence = v2_evidence[challenger_row["method"]]
            if reference_evidence["token_counts"] != challenger_evidence["token_counts"]:
                raise ValueError("paired models produced different V2 token counts")
            decision_record = run_methodv3_text_terminal_adapter(
                pair_manifest,
                ordered_record_ids=v2_ids,
                domains=v2_domains,
                token_counts=reference_evidence["token_counts"],
                reference_mean_nll=reference_evidence["mean_nll"],
                challenger_mean_nll=challenger_evidence["mean_nll"],
                delta=METHOD_V3_DELTA,
                clip=METHOD_V3_TEXT_CLIP,
            )
        selected_method = decision_record["selected_arm"]
        selected_row = next(row for row in (reference_row, challenger_row) if row["method"] == selected_method)
        selected_model = fits_by_stream[selected_row["token_stream_sha256"]]["model"].to(dev)
        report_ppl = selected_row.get("report_ppl")
        if report_ppl is None:
            report_ppl = {
                domain: round(
                    heldout_ppl(selected_model, TOK, held_by_dom[domain], dev), 6
                )
                for domain in doms
            }
        lmeval = selected_row.get("lmeval")
        if lmeval is None:
            lmeval = _run_lmeval(selected_model)
        if not checkpoint_paths:
            checkpoint_path = os.path.join(
                out_dir,
                "checkpoints",
                selected_method,
            )
            os.makedirs(checkpoint_path, exist_ok=False)
            selected_model.save_pretrained(checkpoint_path, safe_serialization=True)
            TOK.save_pretrained(checkpoint_path)
            checkpoint_paths[selected_method] = checkpoint_path
        selected_checkpoint_path = checkpoint_paths.get(
            selected_method, selected_row.get("checkpoint_path")
        )
        if selected_checkpoint_path is not None:
            checkpoint_paths["mmds_adapt"] = selected_checkpoint_path
        del selected_model

        result_row = {
            "method": "mmds_adapt",
            "picked": selected_method,
            "ppl": report_ppl,
            "gmean_ppl": math.exp(
                float(np.mean([math.log(max(value, 1e-12)) for value in report_ppl.values()]))
            ),
            "lmeval": lmeval,
            "n_sel": selected_row["n_sel"],
            "effective_train_tokens": effective_token_cap,
            "selection_order_sha256": selected_row["selection_order_sha256"],
            "selection_set_sha256": selected_row["selection_set_sha256"],
            "token_stream_sha256": selected_row["token_stream_sha256"],
            "initial_state_sha256": selected_row["training_manifest"]["initial_state_sha256"],
            "training_block_order_sha256": selected_row["training_manifest"]["training_block_order_sha256"],
            "checkpoint_path": selected_checkpoint_path,
        }
        payload = {
            "arm": "text",
            "seed": SEED,
            "execution_phase": "method_v3_terminal",
            "paper_scope": "main" if STRATIFY else "proxy_only",
            "config": {**frozen_config, "config_sha256": config_sha256, "run_id": run_id},
            "data_manifest": {
                "pool_sha256": pool_sha256,
                "heldout_sha256": hashlib.sha256(
                    open(os.path.join(_REPO, "data/processed/qpool_heldout.jsonl"), "rb").read()
                ).hexdigest(),
                "influence_cache_file": os.path.basename(icache),
                "influence_cache_sha256": hashlib.sha256(
                    open(icache, "rb").read()
                ).hexdigest(),
                "text_transfer_cache_file": (
                    os.path.basename(text_transfer_cache)
                    if text_transfer_cache
                    else None
                ),
                "text_transfer_cache_sha256": (
                    hashlib.sha256(open(text_transfer_cache, "rb").read()).hexdigest()
                    if text_transfer_cache
                    else None
                ),
            },
            "split_manifest": split_manifest,
            "pair_manifest": pair_manifest,
            "candidate_v1_evidence": candidate_rows,
            "v2_evidence": {
                "ordered_record_ids": v2_ids,
                "domains": v2_domains,
                "reference": v2_evidence[reference_row["method"]],
                "challenger": v2_evidence[challenger_row["method"]],
            },
            "decision_record": decision_record,
            "selection_manifest": {
                key: selected_row[key]
                for key in (
                    "selection_indices",
                    "selected_record_ids",
                    "selection_order_sha256",
                    "selection_set_sha256",
                    "token_stream_sha256",
                )
            },
            "training_manifest": selected_row["training_manifest"],
            "impl_sha256": {
                "runner": runner_impl_sha256,
                "selector_bundle": selector_bundle_sha256,
                "gate": hashlib.sha256(
                    open(
                        os.path.join(
                            _REPO,
                            "src/mmdataselect/fusion/paired_text_logloss_gate.py",
                        ),
                        "rb",
                    ).read()
                ).hexdigest(),
                "adapter": hashlib.sha256(
                    open(
                        os.path.join(
                            _REPO,
                            "src/mmdataselect/fusion/methodv3_text_terminal_adapter.py",
                        ),
                        "rb",
                    ).read()
                ).hexdigest(),
            },
            "selector_component_sha256": selector_component_sha256,
            "results": [result_row],
        }
        payload["methodv3_impl_sha256"] = dict(payload["impl_sha256"])
        payload["pairing_manifest"] = {
            "schema_version": "omniselect.methodv3-text-pairing.v1",
            "pair_manifest_sha256": pair_manifest["pair_manifest_sha256"],
            "reference_arm": reference_row["method"],
            "challenger_arm": challenger_row["method"],
            "same_effective_token_cap": True,
            "effective_token_cap": effective_token_cap,
            "same_initial_state": (
                reference_row["training_manifest"]["initial_state_sha256"]
                == challenger_row["training_manifest"]["initial_state_sha256"]
            ),
            "same_training_block_order": (
                reference_row["training_manifest"]["training_block_order_sha256"]
                == challenger_row["training_manifest"]["training_block_order_sha256"]
            ),
            "same_optimizer_config": (
                reference_row["training_manifest"]["optimizer_config_sha256"]
                == challenger_row["training_manifest"]["optimizer_config_sha256"]
            ),
            "pool_sha256": pool_sha256,
            "split_manifest_sha256": split_manifest["split_manifest_sha256"],
        }
        if not all(
            payload["pairing_manifest"][key]
            for key in (
                "same_effective_token_cap",
                "same_initial_state",
                "same_training_block_order",
                "same_optimizer_config",
            )
        ):
            raise ValueError("text pair violated the frozen same-fit contract")
        payload["pairing_manifest"]["pairing_manifest_sha256"] = _canonical_sha(
            payload["pairing_manifest"]
        )
        payload["artifact_sha256"] = _canonical_sha(payload)
        import tempfile as _tf

        fd, temporary = _tf.mkstemp(dir=out_dir, suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
        result_path = os.path.join(out_dir, "results.json")
        os.replace(temporary, result_path)
        from mmdataselect.utils.repro_bundle import write_repro_bundle

        write_repro_bundle(
            out_dir,
            repo_root=_REPO,
            runner_path=os.path.abspath(__file__),
            arm="text",
            dataset="five-domain-text",
            seed=SEED,
            config=payload["config"],
            result_path=result_path,
            selections={
                **{
                    row["method"]: row["selection_indices"]
                    for row in candidate_rows
                },
                "mmds_adapt": selected_row["selection_indices"],
            },
            text_records=pool,
            split_manifest=split_manifest,
            checkpoint_paths=checkpoint_paths,
            input_paths={
                "pool": pool_path,
                "heldout": os.path.join(
                    _REPO, "data/processed/qpool_heldout.jsonl"
                ),
                "influence_cache": icache,
                **(
                    {"text_transfer_cache": text_transfer_cache}
                    if text_transfer_cache
                    else {}
                ),
            },
        )
        print(
            f"[v3/terminal] {decision_record['decision']} -> {selected_method}; "
            f"report gmean PPL={result_row['gmean_ppl']:.6f}"
        )
        print(f"saved -> {out_dir}/results.json")
        return 0

    results = []
    for method in METHODS:
        t0 = time.time()
        if method == "base":   # finetune-mode floor: the pretrained base, no training
            model = AutoModelForCausalLM.from_pretrained(REF_MODEL).to(dev)
            ppl = {d: round(heldout_ppl(model, TOK, held_by_dom[d], dev), 3) for d in doms}
            results.append({"method": "base", "n_sel": 0, "tok_sel": 0, "steps": 0,
                            "high_frac": 0.0, "set_redundancy": 0.0, "ppl": ppl, "secs": round(time.time() - t0, 1)})
            print(f"  {'base':14} (pretrained, no train) ppl={ppl} ({results[-1]['secs']}s)")
            del model
            continue
        order = ranking(method)
        if method == "noselect":
            sel = order
        elif STRATIFY:  # per-modality fair budget: cut the method's own ranking within each domain
            sel = []
            for d in pool_doms:
                d_order = [i for i in order if pool[i].domain == d]
                sel += rank_to_budget(d_order, tok_counts, dom_budget[d])
        else:
            sel = rank_to_budget(order, tok_counts, budget)
        sub = [pool[i] for i in sel]
        blocks = make_blocks([r.text for r in sub], TOK)
        ntok_sel = sum(tok_counts[i] for i in sel)
        model, steps = train_mini(blocks, dev, total_tokens=ntok_sel * PASSES)
        ppl = {d: round(heldout_ppl(model, TOK, held_by_dom[d], dev), 3) for d in doms}
        ppl_ctrl = {d: round(heldout_ppl(model, TOK, ctrl_by_dom[d], dev), 3) for d in doms}
        lmeval = None
        if os.environ.get("LM_EVAL", "0") == "1":
            # standard-harness accuracy on the trained artifact (scale-up report metric);
            # runs per method so the controller row can reuse its winner's numbers.
            import lm_eval
            from lm_eval.models.huggingface import HFLM
            _lm = HFLM(pretrained=model, tokenizer=TOK,
                       batch_size=int(os.environ.get("LMEVAL_BS", "16")))
            _tasks = os.environ.get("LMEVAL_TASKS", "arc_easy,arc_challenge,hellaswag,openbookqa").split(",")
            _lim = int(os.environ.get("LMEVAL_LIMIT", "0"))
            _res = lm_eval.simple_evaluate(model=_lm, tasks=_tasks, limit=_lim or None,
                                           verbosity="ERROR")
            lmeval = {}
            for t in _tasks:
                r_ = _res["results"].get(t, {})
                lmeval[t] = round(float(r_.get("acc_norm,none", r_.get("acc,none", float("nan")))), 4)
            print(f"    [lm-eval] {method}: {lmeval}")
        hi = sum(1 for i in sel if pool[i].meta.get("quality") == "high")
        row = {
            "method": method, "n_sel": len(sel), "tok_sel": ntok_sel, "steps": steps,
            "high_frac": round(hi / max(1, len(sel)), 3),
            "set_redundancy": round(set_redundancy(sub), 4),
            "ppl": ppl, "ppl_ctrl": ppl_ctrl, "lmeval": lmeval, "secs": round(time.time() - t0, 1),
            "sel_sha256_12": _h.sha256(
                str(sorted(pool[i].id for i in sel)).encode()
            ).hexdigest()[:12],
        }
        results.append(row)
        print(f"  {method:14} n={len(sel):4} hi%={row['high_frac']:.2f} red={row['set_redundancy']:.3f} ppl={ppl} ({row['secs']}s)")
        del model

    # ---- voting-based adaptive controller on the TEXT arm: the portfolio is the baseline
    # selections themselves (fixed, validation-independent constructions -> Prop 2 applies
    # directly); adjudication = geometric-mean PPL on the ADJUDICATION split, disjoint from
    # the REPORT split; the winner's report-split PPL is then reported like any other row.
    PORTFOLIO = [r for r in results if r["method"] not in ("noselect", "base") and "ppl_ctrl" in r]
    if len(PORTFOLIO) >= 2:
        def _score(row):   # lower geometric-mean adjudication PPL = better
            return float(np.mean([math.log(max(row["ppl_ctrl"][d], 1e-9)) for d in doms]))
        win = min(PORTFOLIO, key=_score)
        rnd = next((r for r in PORTFOLIO if r["method"] == "random"), None)
        kap = (_score(rnd) - _score(win)) if rnd is not None else float("nan")
        row = {"method": "mmds_adapt", "n_sel": win["n_sel"], "tok_sel": win["tok_sel"],
               "steps": win["steps"], "high_frac": win["high_frac"],
               "set_redundancy": win["set_redundancy"], "ppl": win["ppl"],
               "ppl_ctrl": win["ppl_ctrl"], "lmeval": win.get("lmeval"), "secs": 0.0, "picked": win["method"],
               "kappa_hat_logppl": round(kap, 4)}
        results.append(row)
        print(f"  {'mmds_adapt':14} picked '{win['method']}' (adjudication log-ppl {_score(win):.4f}, kappa_hat={kap:.4f})")

    print("\n================ EXPERIMENT (per-modality held-out PPL) ================")
    hdr = f"{'method':<15}{'n':>5}{'hi%':>6}{'set_red':>9}" + "".join(f"{d+'_ppl':>12}" for d in doms)
    print(hdr)
    for r in results:
        print(f"{r['method']:<15}{r['n_sel']:>5}{r['high_frac']:>6.2f}{r['set_redundancy']:>9.3f}" + "".join(f"{r['ppl'][d]:>12}" for d in doms))

    # isolated per-config output (audit: the global-mix proxy run must never clobber a
    # main-batch seed): outputs/experiment/{tags}/seed_{SEED}/results.json, atomic write.
    import hashlib as _h
    import tempfile as _tf
    _tag = "-".join([
        f"stratify={os.environ.get('STRATIFY', '0')}",
        f"infl={os.environ.get('INFL_KIND', 'grad')}",
        f"train={TRAIN_MODE}",
        f"lmeval={os.environ.get('LM_EVAL', '0')}",
    ])
    _run_id = os.environ.get("RUN_ID", "")
    if _run_id:
        _tag = f"run_id={_run_id}-" + _tag
    out_dir = os.path.join(_REPO, "outputs", "experiment", _tag, f"seed_{SEED}")
    os.makedirs(out_dir, exist_ok=True)
    _pool_p = os.path.join(_REPO, "data/processed/qpool_train.jsonl")
    payload = {
        "arm": "text", "seed": SEED, "tag": _tag,
        "config": {"budget_frac": BUDGET_FRAC, "passes": PASSES, "hid": HID, "layers": LAYERS,
                   "ctx": CTX, "n": n, "methods": METHODS, "stratify": os.environ.get("STRATIFY", "0"),
                   "infl_kind": os.environ.get("INFL_KIND", "grad"), "train_mode": TRAIN_MODE,
                   "lmeval_tasks": os.environ.get("LMEVAL_TASKS", ""),
                   "run_id": _run_id,
                   "pool_sha256_12": _h.sha256(open(_pool_p, "rb").read()).hexdigest()[:12],
                   "code_sha256_12": _h.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]},
        "results": results,
    }
    fd, _tmp = _tf.mkstemp(dir=out_dir, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(_tmp, os.path.join(out_dir, "results.json"))
    print(f"\nsaved -> {out_dir}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
