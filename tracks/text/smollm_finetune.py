"""Text track: fine-tune SmolLM2-135M (or 360M) on selected records of the five-domain pool, score held-out NLL.

Selection is under a token budget per domain (50% of each domain's tokens). Every candidate is
fine-tuned on an exact token stream of the same length (AdamW 2e-5, OneCycle, batch 16, context
512, two passes), and per-record mean NLL is recorded on each held-out split. The utility is the
negative geometric mean over domains of the token-weighted perplexity. Signals are the text
authenticity heuristics, reference-model loss (pplq) influence of SmolLM2-135M and compression-ratio
redundancy. The learner smollm2_360m fine-tunes SmolLM2-360M (same tokenizer) with the same stream.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.Data import text as txdata
from omniselect.core.selection.console import MultiActorConsole
from omniselect.core.selection.text_transfers import coverage_token_order, herding_token_order
from omniselect.core.signals import AuthenticitySignal, InfluenceSignal, RedundancySignal, hashed_features, minmax
from tracks.common.downstream import Learner, resolve_device, stage_plan
from tracks.common.experiment import SkipCell
from tracks.common.task import Signals, TaskData, Track, TrackConfig
from tracks.text.pool import split_heldout, stratified_cut

FROZEN_TEXT_MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
LEARNER_MODELS = {
    "smollm2_135m": ("HuggingFaceTB/SmolLM2-135M", FROZEN_TEXT_MODEL_REVISION),
    "smollm2_360m": ("HuggingFaceTB/SmolLM2-360M", "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"),
}
LEARNER_ALIASES = {"smollm_135m": "smollm2_135m", "smollm_360m": "smollm2_360m"}
TEXT_METHODS = ("random", "influence_only", "coverage_text", "fixed_fusion", "herding_text", "density_text",
                "quadmix_pub", "dmf_pub", "mmds_adapt")
SMOKE = {"pool_per_domain": 30, "held_per_domain": 40, "ctx": 128, "batch_size": 4, "passes": 1.0,
         "less_epochs": 1, "less_rank": 8, "less_alpha": 32.0, "less_dim": 1024, "less_warmup_frac": 0.1,
         "dmf_probe_tokens": 2048, "methods": ("random", "influence_only", "coverage_text", "fixed_fusion",
                                               "mmds_adapt")}


@dataclass
class TextConfig(TrackConfig):
    """Text constants: reference model, schedule, token budget, smoke subsampling."""

    ref_model: str = "HuggingFaceTB/SmolLM2-135M"
    ref_revision: str = FROZEN_TEXT_MODEL_REVISION
    ctx: int = 512
    batch_size: int = 16
    ft_lr: float = 2e-5
    passes: float = 2.0
    ft_steps_cap: int = 0
    stratify: bool = True
    embed_batch: int = 32
    dmf_probe_tokens: int = 150000
    screen_passes: float = 1.0        # low-fidelity screening: passes over the subsampled token caps
    less_rank: int = 128              # LESS: LoRA rank, scale alpha / rank, lr, warm-up epochs and share
    less_alpha: float = 512.0
    less_lr: float = 2e-5
    less_epochs: int = 4
    less_warmup_frac: float = 0.05
    less_batch: int = 8
    less_dim: int = 8192              # LESS: count-sketch dimension of the gradient features
    less_grad_batch: int = 16         # LESS: records per batched forward of the per-record gradients
    pool_per_domain: int = 0          # > 0 keeps the first records of each domain (smoke)
    held_per_domain: int = 0
    grid_protocols: tuple[str, ...] = ("v2",)


def state_sha256(model) -> str:
    """sha256 over parameter names, dtypes, shapes and raw bytes of a model state dict."""
    import torch

    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        if tensor.dtype == torch.bfloat16:
            dtype_name, shape = "bfloat16", tuple(tensor.shape)
            raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        else:
            arr = tensor.numpy()
            dtype_name, shape, raw = str(arr.dtype), tuple(arr.shape), arr.tobytes(order="C")
        h.update(name.encode("utf-8") + b"\0")
        h.update(dtype_name.encode("ascii") + b"\0")
        h.update(str(shape).encode("ascii") + b"\0")
        h.update(raw)
    return h.hexdigest()


class TextLM:
    """Tokenizer and model factory bound to one pinned model (the reference model unless given)."""

    def __init__(self, tcfg: TextConfig, device: str, model_name: str = "", revision: str = ""):
        from transformers import AutoTokenizer

        self.tcfg, self.device = tcfg, device
        self.model_name, self.revision = model_name or tcfg.ref_model, revision or tcfg.ref_revision
        try:
            self.tok = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        except OSError as exc:
            raise SkipCell(f"{self.model_name}@{self.revision[:12]} unavailable: {exc}") from exc
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def model(self):
        from transformers import AutoModelForCausalLM

        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_name, revision=self.revision)
        except OSError as exc:
            raise SkipCell(f"{self.model_name}@{self.revision[:12]} unavailable: {exc}") from exc
        return model.to(self.device)

    def token_count(self, text: str) -> int:
        return len(self.tok(text, truncation=True, max_length=self.tcfg.ctx).input_ids)

    def per_record_mean_nll(self, model, records) -> tuple[np.ndarray, np.ndarray]:
        """Mean token NLL and prediction-token count of each record (truncated to ctx)."""
        import torch

        model.eval()
        values, counts = [], []
        with torch.no_grad():
            for r in records:
                enc = self.tok(r.text, truncation=True, max_length=self.tcfg.ctx, return_tensors="pt")
                ids = enc.to(self.device)["input_ids"]
                if ids.size(1) < 2:
                    values.append(0.0)
                    counts.append(0)
                    continue
                values.append(float(model(input_ids=ids, labels=ids).loss.item()))
                counts.append(int(ids.size(1) - 1))
        return np.asarray(values, dtype=float), np.asarray(counts, dtype=np.int64)

    def exact_blocks(self, selected: list[int], records, domain_caps: dict[str, int]):
        """Per-domain token streams truncated to each domain's cap, reshaped into ctx blocks."""
        import torch

        chunks: list[int] = []
        for domain in sorted(domain_caps):
            ids: list[int] = []
            for index in (i for i in selected if records[i].domain == domain):
                ids.extend(self.tok(records[index].text).input_ids + [self.tok.eos_token_id])
                if len(ids) >= domain_caps[domain]:
                    break
            chunks.extend(ids[: domain_caps[domain]])
        usable = (len(chunks) // self.tcfg.ctx) * self.tcfg.ctx
        array = np.asarray(chunks[:usable], dtype=np.int64)
        if len(array) == 0:
            raise ValueError("selected token stream is shorter than one context block")
        stream_sha = hashlib.sha256(array.astype("<i8").tobytes()).hexdigest()
        return torch.from_numpy(array.reshape(-1, self.tcfg.ctx)), stream_sha

    def finetune(self, blocks, total_tokens: int, seed: int, lr: float):
        """AdamW (betas 0.9, 0.95, wd 0.1) with OneCycle over total_tokens / (ctx * batch) steps."""
        import torch

        torch.manual_seed(seed)
        model = self.model().train()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
        steps = max(1, int(total_tokens / (self.tcfg.ctx * self.tcfg.batch_size)))
        if self.tcfg.ft_steps_cap > 0:
            steps = min(steps, self.tcfg.ft_steps_cap)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
        generator = torch.Generator().manual_seed(seed)
        step, n = 0, blocks.size(0)
        while step < steps:
            perm = torch.randperm(n, generator=generator)
            for start in range(0, n, self.tcfg.batch_size):
                if step >= steps:
                    break
                batch = blocks[perm[start:start + self.tcfg.batch_size]].to(self.device)
                loss = model(input_ids=batch, labels=batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
        return model.eval(), steps


class TextLearner(Learner):
    """Fine-tunes a fresh SmolLM2 per subset on an exact-length token stream. Scores are per record."""

    def __init__(self, plan, lm: TextLM, pool, split_records: dict[str, list], domain_caps, seed, tcfg,
                 name: str = "smollm2_135m"):
        super().__init__(plan=plan, keep_models=False, name=name)
        self.lm, self.pool, self.split_records, self.domain_caps, self.seed, self.tcfg = (
            lm, pool, split_records, domain_caps, seed, tcfg)

    def caps_and_passes(self, stage: str) -> tuple[dict[str, int], float]:
        """Domain token caps and passes of a fit at ``stage`` (the screen stage scales both)."""
        fid = self.fidelity(stage)
        if "token_fraction" not in fid:
            return self.domain_caps, float(self.tcfg.passes)
        ctx = self.tcfg.ctx
        caps = {d: max(ctx, (int(c * float(fid["token_fraction"])) // ctx) * ctx) for d, c in self.domain_caps.items()}
        return caps, float(fid["passes"])

    def cost_factor(self, stage: str) -> float:
        """Passes of the stage over the reported passes (1 for every stage but screen)."""
        return self.caps_and_passes(stage)[1] / float(self.tcfg.passes)

    def fit(self, subset, stage):
        caps, passes = self.caps_and_passes(stage)
        blocks, stream_sha = self.lm.exact_blocks([int(i) for i in subset], self.pool, caps)
        total = int(sum(caps.values()) * passes)
        model, steps = self.lm.finetune(blocks, total, self.seed, self.tcfg.ft_lr)
        model.stream_sha256 = stream_sha
        model.steps = steps
        return model

    def score(self, model, split):
        records = self.split_records[split]
        nll, counts = self.lm.per_record_mean_nll(model, records)
        return {"nll": nll, "n_tokens": counts, "domain": np.array([r.domain for r in records]),
                "record_id": np.array([r.id for r in records])}

    def save(self, model, directory: Path, name: str) -> str:
        directory.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(directory, safe_serialization=True)
        self.lm.tok.save_pretrained(directory)
        return str(directory)


def mean_pooled_embeddings(lm: TextLM, records, tcfg: TextConfig) -> np.ndarray:
    """L2-normalized mean-pooled last hidden states of the frozen reference model (the transfer features)."""
    import torch

    model = lm.model().eval()
    feats = np.zeros((len(records), int(model.config.hidden_size)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(records), tcfg.embed_batch):
            batch = records[start:start + tcfg.embed_batch]
            enc = lm.tok([r.text for r in batch], padding=True, truncation=True, max_length=tcfg.ctx,
                         return_tensors="pt").to(lm.device)
            hidden = model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                 use_cache=False).last_hidden_state.float()
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp_min(1e-12)
            feats[start:start + len(batch)] = pooled.cpu().numpy()
    return feats


def _cached(path: str, compute):
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    arrays = compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)
    return arrays


class TextTrack(Track):
    """Five-domain text pool with SmolLM2-135M fine-tuning."""

    name = "text"
    utility = "neg_gmean_ppl"
    config_cls = TextConfig
    default_learner = "smollm2_135m"
    sorted_training = False
    datasets = {"five_domain": {"budget_frac": 0.5, "grid_weights": ((0, .5, .5), (.34, .33, .33), (.5, .5, 0)),
                                "grid_q": (0.25,), "grid_lam": (0.5,), "methods": TEXT_METHODS, "smoke": SMOKE,
                                "coop_rhos": ("1.3",), "coop_ungated": ("herding",)}}

    def config(self, dataset, learner=None, seed=0, smoke=False, **overrides):
        """TrackConfig with the learner name normalized (smollm_135m and smollm_360m are accepted)."""
        learner = LEARNER_ALIASES.get(learner, learner)
        if "learner" in overrides:
            overrides["learner"] = LEARNER_ALIASES.get(overrides["learner"], overrides["learner"])
        cfg = super().config(dataset, learner=learner, seed=seed, smoke=smoke, **overrides)
        if cfg.learner not in LEARNER_MODELS:
            raise ValueError(f"text learner must be one of {tuple(LEARNER_MODELS)}, got {cfg.learner!r}")
        return cfg

    def omni_overrides(self, protocol: str, tcfg: TextConfig) -> dict[str, Any]:
        base = {"cache.by_subset_hash": True, "fidelity.scoring_equals_reported": True,
                "gate.text_clip": 1.0, "synthesis.coordinate_ascent": False}
        if protocol == "canonical":
            return {**base, "gate.kind": "lcb", "gate.split": "rank", "gate.pair_split": "con",
                    "gate.k_challengers": 1, "screening.kind": "off", "synthesis.consensus": False}
        return base          # v2 presets: the preset gate (margin on V_rank, audit on V_conf)

    def load(self, tcfg: TextConfig, ocfg) -> TaskData:
        pool, held, pool_path = txdata.load_pool(tcfg.data_root)
        if tcfg.pool_per_domain > 0:
            pool = _first_per_domain(pool, tcfg.pool_per_domain)
        if tcfg.held_per_domain > 0:
            held = _first_per_domain(held, tcfg.held_per_domain)
        device = resolve_device(tcfg.device)
        lm = TextLM(tcfg, device)
        tok_counts = [lm.token_count(r.text) for r in pool]
        domains = [r.domain for r in pool]
        dom_budget = {d: int(tcfg.budget_frac * sum(t for t, dd in zip(tok_counts, domains) if dd == d))
                      for d in sorted(set(domains))}
        domain_caps = {d: max(tcfg.ctx, (b // tcfg.ctx) * tcfg.ctx) for d, b in dom_budget.items()}
        splits_txt = split_heldout(held, tcfg.seed, ocfg.splits.mode, sizes=tuple(ocfg.splits.text_sizes))
        val = splits_txt.con + splits_txt.rank + splits_txt.conf
        from omniselect.core.adjudication.splits import ValidationSplits

        a, b = len(splits_txt.con), len(splits_txt.con) + len(splits_txt.rank)
        vsplits = ValidationSplits(con=np.arange(0, a), rank=np.arange(a, b), conf=np.arange(b, len(val)))
        tags = np.array([r.meta.get("quality", "high") if isinstance(r.meta, dict) else "high" for r in pool],
                        dtype=object)
        mechanisms = [str(r.meta.get("noise", "")) if isinstance(r.meta, dict) else "" for r in pool]

        def cut(order: list[int]) -> list[int]:
            if tcfg.stratify:
                return stratified_cut(order, domains, tok_counts, dom_budget)
            from tracks.text.pool import rank_to_budget

            return rank_to_budget(order, tok_counts, int(tcfg.budget_frac * sum(tok_counts)))

        return TaskData(
            pool_ids=[r.id for r in pool], val_ids=[r.id for r in val], test_ids=[r.id for r in splits_txt.report],
            tags=tags, splits=vsplits, budget=int(sum(dom_budget.values())), utility=self.utility, records=pool,
            budget_cut=cut, mechanisms=mechanisms,
            arrays={"lm": lm, "pool": pool, "tok_counts": tok_counts, "domains": domains, "dom_budget": dom_budget,
                    "domain_caps": domain_caps, "splits": splits_txt, "device": device, "pool_path": pool_path},
            provenance={"pool_sha256": txdata.pool_sha256(pool_path), "ref_model": tcfg.ref_model,
                        "ref_revision": tcfg.ref_revision, "dom_budget": dom_budget, "domain_caps": domain_caps,
                        "layout": splits_txt.layout},
        )

    def signals(self, data: TaskData, tcfg: TextConfig, ocfg) -> Signals:
        a = data.arrays
        pool, lm = a["pool"], a["lm"]
        ids_key = hashlib.sha256(json.dumps([r.id for r in pool]).encode()).hexdigest()[:12]
        key = f"{ids_key}_{tcfg.ref_model.split('/')[-1]}_rev{tcfg.ref_revision[:12]}_c{tcfg.ctx}"
        base = os.path.join(tcfg.data_root, "processed")

        def loss_arrays():
            model = lm.model()
            nll, _ = lm.per_record_mean_nll(model, pool)
            return {"loss": nll}

        def embed_arrays():
            return {"features": mean_pooled_embeddings(lm, pool, tcfg)}

        influence = -_cached(os.path.join(base, f"text_influence_pplq_{key}.npz"), loss_arrays)["loss"]
        transfer = _cached(os.path.join(base, f"text_transfer_{key}.npz"), embed_arrays)["features"]
        auth = AuthenticitySignal().score(pool)
        red = RedundancySignal().score(pool)
        feats = hashed_features(pool)
        console = MultiActorConsole(
            [("redundancy", RedundancySignal()), ("influence", InfluenceSignal())],
            weights=np.log(np.array([1 - tcfg.w_infl, tcfg.w_infl]) + 1e-9),
            conflict_gate=True, anneal=0.6, group_key="domain", trust_region=0.5, ema_beta=0.3, min_weight=0.02,
        )
        imp_dyn = console.importance(pool, scores=np.stack([minmax(red), minmax(influence)], axis=0), progress=0.5)
        reference = a["splits"].reference
        adapt_extras, adapt_saved = {}, {}
        if {"el2n_adapt", "grand_adapt", "ccs_adapt"} & set(tcfg.methods):
            from tracks.text.score_adapt import frozen_text_scores

            payload = frozen_text_scores(lm, pool, tcfg, a["pool_path"])
            payload.update({"tokens": a["tok_counts"],
                            "domains": a["domains"] if tcfg.stratify else ["all"] * len(pool),
                            "budgets": a["dom_budget"] if tcfg.stratify else {"all": data.budget}})
            adapt_extras["adapt_scores"] = payload
            adapt_saved = {"el2n_adapt": payload["el2n"], "grand_adapt": payload["grand"]}
        return Signals(
            auth=auth, influence=influence, redundancy=red, features=feats,
            reference_ids=[r.id for r in reference],
            reference_source="v_con" if ocfg.splits.mode == "three_way" else "heldout_reference_half",
            extras={"imp_dyn": imp_dyn, "texts": [r.text for r in pool], "ids": [r.id for r in pool],
                    "less_inputs": _less_inputs(lm, pool, reference, tcfg),
                    "dsir_target_texts": [r.text for r in reference], "transfer_features": transfer, **adapt_extras},
            saved={"transfer_features": transfer, **adapt_saved},
        )

    def alignment(self, data: TaskData, sig: Signals, tcfg: TextConfig, ocfg):
        """Alignment = the LESS score of each pool record, computed once and shared with the LESS member."""
        from benchmark.Methods.LESS.method import less_cached

        inputs = sig.extras["less_inputs"]
        keep = ocfg.coverage.space != "feature"
        scores, _ = less_cached(inputs, tcfg.seed, keep_sketches=keep)
        sketches = inputs["cache"].get("sketches") if keep else None
        return {"scores": np.asarray(scores, dtype=float), "phi": sketches,
                "err": None if sketches is None else np.ones((len(scores), 1)), "source": "LESS score"}

    def cooperative_inputs(self, data: TaskData, sig: Signals, tcfg: TextConfig, ocfg) -> dict[str, Any]:
        """Cosine geometry of the frozen-LM transfer features, the token-budgeted gate and selector orders.

        The gate keeps the records of a score order inside rho times each domain's token budget (the
        budget cut with scaled budgets). herding and kcenter inside the gate are the per-domain token
        orders of herding_text and coverage_text. Cleanliness features: text authenticity, compression
        ratio, length in tokens and the neighbour distances. V_con records are embedded here.
        """
        from tracks.text.pool import rank_to_budget

        a = data.arrays
        domains, tok_counts, dom_budget, lm = a["domains"], a["tok_counts"], a["dom_budget"], a["lm"]
        transfer = sig.extras["transfer_features"]
        tdomains = domains if tcfg.stratify else ["all"] * data.n
        tbudgets = dom_budget if tcfg.stratify else {"all": int(sum(dom_budget.values()))}

        def budget_gate(order: list[int], rho: float) -> list[int]:
            if tcfg.stratify:
                scaled = {d: int(round(rho * b)) for d, b in dom_budget.items()}
                return stratified_cut(order, domains, tok_counts, scaled)
            return rank_to_budget(order, tok_counts, int(round(rho * tcfg.budget_frac * sum(tok_counts))))

        def token_order(rule):
            def order(subset: np.ndarray) -> list[int]:
                fn = herding_token_order if rule == "herding" else coverage_token_order
                local = fn(transfer[subset], [tdomains[int(i)] for i in subset], [tok_counts[int(i)] for i in subset],
                           tbudgets)
                return [int(subset[j]) for j in local]
            return order

        con = a["splits"].con
        return {
            "veto_features": transfer, "metric": "cosine", "selector_features": transfer,
            "con_veto_features": mean_pooled_embeddings(lm, con, tcfg),
            "budget_gate": budget_gate, "token_orders": {"herding": token_order("herding"),
                                                         "kcenter": token_order("kcenter")},
            "cleanliness": {"kind": "text",
                            "pool": {"authenticity": sig.auth, "compression_ratio": sig.redundancy,
                                     "length_tokens": np.asarray(tok_counts, dtype=float)},
                            "con": {"authenticity": AuthenticitySignal().score(con),
                                    "compression_ratio": RedundancySignal().score(con),
                                    "length_tokens": np.asarray([lm.token_count(r.text) for r in con], dtype=float)}},
        }

    def strategy_overrides(self, data: TaskData, sig: Signals, tcfg: TextConfig, ocfg) -> dict:
        a = data.arrays
        n = data.n
        domains, tok_counts, dom_budget = a["domains"], a["tok_counts"], a["dom_budget"]
        transfer = sig.extras["transfer_features"]
        tdomains = domains if tcfg.stratify else ["all"] * n
        tbudgets = dom_budget if tcfg.stratify else {"all": int(sum(dom_budget.values()))}

        def random(ctx, k):
            return [int(i) for i in np.random.default_rng(tcfg.seed).permutation(n)]

        def influence_only(ctx, k):
            return [int(i) for i in np.argsort(-sig.influence, kind="stable")]

        def auth_only(ctx, k):
            return [int(i) for i in np.argsort(-sig.auth, kind="stable")]

        def auth_bottom(ctx, k):
            return [int(i) for i in np.argsort(sig.auth, kind="stable")]

        def coverage_text(ctx, k):
            return coverage_token_order(transfer, tdomains, tok_counts, tbudgets)

        def herding_text(ctx, k):
            return herding_token_order(transfer, tdomains, tok_counts, tbudgets)

        def density_text(ctx, k):
            from benchmark.Methods.Density.method import density_select

            return density_select(transfer, n, seed=tcfg.seed)

        def quadmix_pub(ctx, k):
            from benchmark.Methods.QuaDMix.method import quadmix_published_core

            tw = np.maximum(np.asarray(tok_counts, dtype=float), 1.0)
            return quadmix_published_core(sig.influence, sig.features, n, domains=np.asarray(domains),
                                          token_weights=tw, seed=tcfg.seed)

        def fixed_fusion(ctx, k):
            from benchmark.Methods.FixedFusion.method import fixed_fusion as ff

            return ff(data.records, sig.extras["imp_dyn"], sig.auth, n, auth_q=tcfg.auth_q, lam=tcfg.lam,
                      features=sig.features, gate=ocfg.fixes.fixed_fusion_gate)

        def dmf_pub(ctx, k):
            return _dmf_pub_text(data, sig, tcfg, ocfg)

        return {"random": random, "influence_only": influence_only, "auth_only": auth_only,
                "auth_bottom": auth_bottom, "coverage_text": coverage_text, "herding_text": herding_text,
                "density_text": density_text, "quadmix_pub": quadmix_pub, "fixed_fusion": fixed_fusion,
                "dmf_pub": dmf_pub}

    def learner(self, data: TaskData, tcfg: TextConfig, ocfg) -> TextLearner:
        a = data.arrays
        splits_txt = a["splits"]
        total = int(sum(a["domain_caps"].values()) * tcfg.passes)
        steps = max(1, int(total / (tcfg.ctx * tcfg.batch_size)))
        schedule = {"steps": steps, "lr": tcfg.ft_lr}
        frac = float(ocfg.screening.low_fidelity_fraction)
        low_total = int(sum(a["domain_caps"].values()) * frac * tcfg.screen_passes)
        low = {"steps": max(1, int(low_total / (tcfg.ctx * tcfg.batch_size))), "passes": tcfg.screen_passes,
               "token_fraction": frac}
        plan = stage_plan(ocfg, scoring=schedule, reported=schedule,
                          seeds={"con": tcfg.seed, "rank": tcfg.seed, "report": tcfg.seed}, low=low)
        split_records = {"con": splits_txt.con, "rank": splits_txt.rank, "conf": splits_txt.conf,
                         "test": splits_txt.report}
        lm = a["lm"]
        name, revision = LEARNER_MODELS[tcfg.learner]
        if (name, revision) != (lm.model_name, lm.revision):
            lm = TextLM(tcfg, a["device"], name, revision)
        return TextLearner(plan, lm, a["pool"], split_records, a["domain_caps"], tcfg.seed, tcfg, name=tcfg.learner)


def _less_inputs(lm: TextLM, pool, reference, tcfg: TextConfig) -> dict:
    """Inputs of the LESS strategy: the reference SmolLM2, the pool texts and the target (reference sample) texts."""
    from benchmark.Methods.LESS.method import LessConfig

    cfg = LessConfig(rank=tcfg.less_rank, alpha=tcfg.less_alpha, lr=tcfg.less_lr, epochs=tcfg.less_epochs,
                     warmup_frac=tcfg.less_warmup_frac, batch_size=tcfg.less_batch, dim=tcfg.less_dim, ctx=tcfg.ctx,
                     grad_batch=tcfg.less_grad_batch)
    return {"model_factory": lm.model, "tokenizer": lm.tok, "texts": [r.text for r in pool],
            "target_texts": [r.text for r in reference], "target_domains": [r.domain for r in reference],
            "device": lm.device, "config": cfg}


def _first_per_domain(records, per_domain: int):
    counts: dict[str, int] = {}
    out = []
    for r in records:
        if counts.get(r.domain, 0) < per_domain:
            out.append(r)
            counts[r.domain] = counts.get(r.domain, 0) + 1
    return out


def _dmf_pub_text(data: TaskData, sig: Signals, tcfg: TextConfig, ocfg) -> list[int]:
    """DMF Eqs. 6 to 8 with rewards -mean_d log PPL of a short probe fine-tune on the reference sample."""
    a = data.arrays
    lm, pool = a["lm"], a["pool"]
    reference = a["splits"].reference
    cache: dict[str, float] = {}

    def reward(order: list[int]) -> float:
        selected = data.budget_cut(order)
        key = hashlib.sha256(json.dumps(selected).encode()).hexdigest()
        if key not in cache:
            blocks, _ = lm.exact_blocks(selected, pool, a["domain_caps"])
            model, _ = lm.finetune(blocks, tcfg.dmf_probe_tokens, tcfg.seed, tcfg.ft_lr)
            nll, counts = lm.per_record_mean_nll(model, reference)
            doms = np.array([r.domain for r in reference])
            per = [float((nll[doms == d] * counts[doms == d]).sum() / max(counts[doms == d].sum(), 1))
                   for d in sorted(set(doms.tolist()))]
            cache[key] = -float(np.mean(per))
        return cache[key]

    scores = np.stack([minmax(sig.auth), minmax(sig.influence), minmax(sig.redundancy)], axis=0)
    theta = np.ones(3) / 3
    actor = np.array([reward([int(i) for i in np.argsort(-scores[j], kind="mergesort")]) for j in range(3)])
    best_order, best = None, -np.inf
    for _ in range(6):
        order = [int(i) for i in np.argsort(-(theta @ scores), kind="mergesort")]
        value = reward(order)
        if value > best:
            best, best_order = value, order
        projected = np.maximum(theta + 0.5 * (actor - actor.mean()), 0.0)
        theta = projected / projected.sum() if projected.sum() > 1e-12 else np.ones(3) / 3
    return best_order
