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
        m = AutoModelForCausalLM.from_pretrained(REF_MODEL).to(dev).train()
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
    TOK = AutoTokenizer.from_pretrained(REF_MODEL)
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
    for d, rs in by_dom_all.items():
        half = max(1, len(rs) // 2)
        refs_by_dom[d] = rs[:half]      # clean per-modality reference for grad-alignment
        rest = rs[half:]
        q = max(1, len(rest) // 2)
        ctrl_by_dom[d] = rest[:q]       # controller ADJUDICATION split (disjoint from report)
        held_by_dom[d] = rest[q:]       # disjoint eval set for held-out ppl (REPORT split)
    doms = sorted(held_by_dom)
    print(f"device={dev} | pool={n} ({total_tok} tok) | budget={budget} tok | heldout={ {d: len(v) for d, v in held_by_dom.items()} }")
    print(f"mini-LM: hid={HID} layers={LAYERS} ctx={CTX} | passes={PASSES} | train_mode={TRAIN_MODE} | methods={METHODS}")
    if os.environ.get("CONFIG_PROBE", "0") == "1":
        # fail-closed pre-flight (TEXT_LANE_HARD_STOP item 3): the lane verifies this
        # header carries the intended train_mode BEFORE the formal loop starts.
        print("[config-probe] exiting before any scoring/training")
        return 0

    # ---- influence variants (cached): grad-align (primary) + raw-loss + ppl-quality ----
    # cache key = pool content hash + reference model + CTX + INFL_KIND (audit: a new
    # pool must never silently reuse the pilot pool's influence cache)
    import hashlib as _h
    _pool_sha = _h.sha256(open(os.path.join(_REPO, "data/processed/qpool_train.jsonl"), "rb").read()).hexdigest()[:12]
    _ck = f"{_pool_sha}_{REF_MODEL.split('/')[-1]}_c{CTX}_{os.environ.get('INFL_KIND', 'grad')}"
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
        ref = AutoModelForCausalLM.from_pretrained(REF_MODEL).to(dev)
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

    def ranking(method):
        if method == "noselect":
            return list(range(n))
        if method == "random":
            return list(rng.permutation(n))
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
                    f"only={ONLY}"])
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
        if method == "mmdataselect":       # authenticity as PREREQUISITE FILTER, then influence x diversity
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
