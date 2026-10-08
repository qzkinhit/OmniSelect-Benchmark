"""LESS (Xia et al., ICML 2024): LoRA gradient features of training records against target-set gradients.

A LoRA adapter (rank ``rank``, scale alpha / rank, on the attention q and v projections, factors in
the dtype of the base weight) is trained on a seeded ``warmup_frac`` share of the pool for ``epochs``
epochs with AdamW. After every epoch the per-record LoRA gradient of each pool record is
Adam-preconditioned with that epoch's optimizer state, (b1 m + (1 - b1) g) / (sqrt(b2 v + (1 - b2) g^2)
+ eps), and the per-record gradient of each target record is kept as is. Gradients are cast to
float32, L2-normalized and projected to ``dim`` coordinates by a seeded count sketch, all on the
model's device, and only the sketch leaves the device. The score of a pool record is max over target
domains of the sum over epochs of the cosine with the mean target gradient of the domain. The order
sorts the scores (ties by pool index).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from omniselect.core.portfolio.registry import SelectionContext, register


@dataclass
class LessConfig:
    """LESS settings. The defaults follow the paper except the count sketch and the smaller LoRA targets."""

    rank: int = 128
    alpha: float = 512.0
    lr: float = 2e-5
    epochs: int = 4
    warmup_frac: float = 0.05
    batch_size: int = 8
    dim: int = 8192
    ctx: int = 512
    targets: tuple[str, ...] = ("q_proj", "v_proj")
    grad_batch: int = 16              # records per batched forward of the per-record gradients (1: one at a time)


def add_lora(model, rank: int, alpha: float, targets: Sequence[str]) -> list:
    """Wrap every nn.Linear whose name ends with a target in a LoRA adapter, freeze the base. Returns LoRA params."""
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear):
            super().__init__()
            self.base = base
            self.scale = float(alpha) / int(rank)
            weight = base.weight
            init = torch.randn(int(rank), base.in_features, device=weight.device) / math.sqrt(base.in_features)
            # the factors take the base weight's dtype (bfloat16 when transformers loads the config dtype)
            self.lora_a = nn.Parameter(init.to(dtype=weight.dtype))
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, int(rank), device=weight.device,
                                                   dtype=weight.dtype))

            self.capture = False          # keep the inputs and output gradients of the branch
            self.saved: dict = {}

        def forward(self, x):
            out = self.base(x)
            h = x.to(self.lora_a.dtype)
            z = h @ self.lora_a.T
            branch = (z @ self.lora_b.T) * self.scale
            if self.capture and branch.requires_grad:
                self.saved = {"h": h.detach(), "z": z.detach()}
                branch.register_hook(lambda g: self.saved.__setitem__("g", g.detach()))
            return out + branch.to(out.dtype)

    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                wrapped = LoRALinear(child)
                setattr(module, child_name, wrapped)
                params.extend([wrapped.lora_a, wrapped.lora_b])
    return params


class CountSketch:
    """Seeded count sketch R^p -> R^dim: coordinate j goes to bucket h(j) with sign s(j).

    ``bucket`` and ``sign`` come from default_rng(seed). ``__call__`` projects a numpy vector,
    ``project`` a torch vector on its own device with index_add_ (the same function, float32).
    """

    def __init__(self, p: int, dim: int, seed: int):
        rng = np.random.default_rng(seed)
        self.bucket = rng.integers(0, int(dim), size=int(p))
        self.sign = rng.choice(np.array([-1.0, 1.0]), size=int(p))
        self.dim = int(dim)
        self._device_cache: dict[str, Any] = {}

    def __call__(self, v: np.ndarray) -> np.ndarray:
        """Sketch of v, accumulated from float32 input and returned as float32."""
        weights = self.sign * np.asarray(v, dtype=np.float32)
        return np.bincount(self.bucket, weights=weights, minlength=self.dim).astype(np.float32)

    def _on(self, device):
        import torch

        key = str(device)
        if key not in self._device_cache:
            self._device_cache[key] = (torch.as_tensor(self.bucket, dtype=torch.long, device=device),
                                       torch.as_tensor(self.sign, dtype=torch.float32, device=device))
        return self._device_cache[key]

    def project(self, v):
        """Sketch of the float32 torch vector ``v`` (P,) or of each row of ``v`` (B, P) on v's device."""
        import torch

        bucket, sign = self._on(v.device)
        out = torch.zeros(v.shape[:-1] + (self.dim,), dtype=torch.float32, device=v.device)
        return out.index_add_(v.dim() - 1, bucket, v * sign)


def lora_modules(model) -> list:
    """The LoRA-wrapped linear layers of ``model`` in the order of their parameters in ``add_lora``."""
    return [m for m in model.modules() if hasattr(m, "lora_a") and hasattr(m, "capture")]


def per_sample_lora_grads(model, tok, texts: Sequence[str], ctx: int, device: str):
    """Exact per-record gradients of the LoRA parameters from one batched forward and backward.

    The records are right-padded to a common length with an attention mask, each record's loss is
    its mean token negative log-likelihood over its own predicted tokens (as ``_record_loss``), and
    the backward pass runs on the sum of the record losses. Records do not interact (causal attention
    with the padding mask, no batch statistics), so the output gradient G_b of a LoRA branch at
    record b is the gradient of record b's loss. With branch = scale (h A^T) B^T, the per-record
    gradients are dB_b = scale sum_t G_bt^T z_bt and dA_b = scale sum_t (G_bt B)^T h_bt, computed in
    float32. Returns a float32 tensor (B, P) in the order of ``_flat_grad`` and the mask of records
    with at least two tokens.
    """
    import torch
    import torch.nn.functional as F

    ids_list = [tok(t, truncation=True, max_length=int(ctx), return_tensors="pt")["input_ids"][0] for t in texts]
    valid = [len(x) >= 2 for x in ids_list]
    modules = lora_modules(model)
    n_params = sum(m.lora_a.numel() + m.lora_b.numel() for m in modules)
    out = torch.zeros(len(texts), n_params, dtype=torch.float32, device=device)
    rows = [i for i, ok in enumerate(valid) if ok]
    if not rows:
        return out, valid
    length = max(len(ids_list[i]) for i in rows)
    pad = getattr(tok, "pad_token_id", None) or 0
    ids = torch.full((len(rows), length), int(pad), dtype=torch.long)
    mask = torch.zeros((len(rows), length), dtype=torch.long)
    for j, i in enumerate(rows):
        ids[j, : len(ids_list[i])] = ids_list[i]
        mask[j, : len(ids_list[i])] = 1
    ids, mask = ids.to(device), mask.to(device)
    for m in modules:
        m.capture, m.saved = True, {}
    try:
        logits = model(input_ids=ids, attention_mask=mask).logits
        total = 0.0
        for j, i in enumerate(rows):          # float32 logits of one record at a time bound the memory
            n = len(ids_list[i])
            total = total + F.cross_entropy(logits[j, : n - 1].float(), ids[j, 1:n], reduction="mean")
        total.backward()
        blocks = []
        for m in modules:
            g = m.saved["g"].float() * m.scale
            dB = torch.einsum("bto,btr->bor", g, m.saved["z"].float())
            dA = torch.einsum("btr,bti->bri", g @ m.lora_b.detach().float(), m.saved["h"].float())
            blocks += [dA.reshape(len(rows), -1), dB.reshape(len(rows), -1)]
        out[torch.as_tensor(rows, device=device)] = torch.cat(blocks, dim=1)
    finally:
        for m in modules:
            m.capture, m.saved = False, {}
        for p in model.parameters():
            p.grad = None
    return out, valid


def _flat_grad(params):
    """Concatenated parameter gradients as one float32 tensor on the parameters' device."""
    import torch

    return torch.cat([p.grad.detach().reshape(-1).to(torch.float32) for p in params])


def _record_loss(model, tok, text: str, ctx: int, device: str):
    enc = tok(text, truncation=True, max_length=int(ctx), return_tensors="pt")
    ids = enc["input_ids"].to(device)
    if ids.size(1) < 2:
        return None
    return model(input_ids=ids, labels=ids).loss


def _features(g, sketch: CountSketch, m=None, v=None, betas=(0.9, 0.999), eps: float = 1e-8):
    """Adam preconditioning (when m, v are given), unit normalization, count sketch and normalization of a
    float32 gradient block (B, P) or vector (P,) on its device."""
    import torch

    if g.dim() == 2 and g.shape[0] > 4:        # rows in chunks of 4 bound the temporaries of a (B, P) block
        return torch.cat([_features(g[s:s + 4], sketch, m, v, betas, eps) for s in range(0, g.shape[0], 4)])
    b1, b2 = betas
    if m is not None:
        g = (b1 * m + (1 - b1) * g) / (torch.sqrt(b2 * v + (1 - b2) * g * g) + eps)
    g = g / (torch.linalg.vector_norm(g, dim=-1, keepdim=True) + 1e-12)
    z = sketch.project(g)
    return z / (torch.linalg.vector_norm(z, dim=-1, keepdim=True) + 1e-12)


def gradient_features(model, tok, params, texts: Sequence[str], ctx: int, device: str, sketch: CountSketch,
                      adam_state: dict | None = None, betas=(0.9, 0.999), eps: float = 1e-8,
                      keep_on_device: bool = False, grad_batch: int = 1, info: dict | None = None) -> np.ndarray:
    """Normalized, sketched per-record LoRA gradients (Adam-preconditioned when ``adam_state`` is given).

    With ``grad_batch`` > 1 the per-record gradients of ``grad_batch`` records come from one batched
    forward and backward (``per_sample_lora_grads``). With 1, or when the batched path raises, each
    record runs its own forward and backward. Preconditioning with the stored moments (float32 tensors
    on the model's device), the count sketch and both normalizations run on the model's device in
    float32. Each block moves its sketches to the CPU, or all of them at the end with
    ``keep_on_device``. ``info`` receives the path used ("batched" or "batch1").
    """
    import torch

    m = v = None
    if adam_state is not None:
        m, v = adam_state["exp_avg"], adam_state["exp_avg_sq"]
    out = np.zeros((len(texts), sketch.dim), dtype=np.float32)
    kept: list[tuple[int, object]] = []
    path = "batched" if int(grad_batch) > 1 else "batch1"
    if path == "batched":
        try:
            for start in range(0, len(texts), int(grad_batch)):
                block, valid = per_sample_lora_grads(model, tok, texts[start:start + int(grad_batch)], ctx, device)
                with torch.no_grad():
                    z = _features(block, sketch, m, v, betas, eps)
                idx = [start + j for j, ok in enumerate(valid) if ok]
                rows = z[[j for j, ok in enumerate(valid) if ok]]
                if keep_on_device:
                    kept.append((idx, rows))
                elif idx:
                    out[idx] = rows.cpu().numpy()
        except (RuntimeError, NotImplementedError, KeyError) as exc:
            path = f"batch1 (batched path raised {type(exc).__name__}: {str(exc)[:160]})"
            kept, out = [], np.zeros((len(texts), sketch.dim), dtype=np.float32)
    if path != "batched":
        for i, text in enumerate(texts):
            for p in params:
                p.grad = None
            loss = _record_loss(model, tok, text, ctx, device)
            if loss is None:
                continue
            loss.backward()
            with torch.no_grad():
                z = _features(_flat_grad(params), sketch, m, v, betas, eps)
            if keep_on_device:
                kept.append(([i], z[None]))
            else:
                out[i] = z.cpu().numpy()
        for p in params:
            p.grad = None
    if kept:
        idx = [i for block_idx, _ in kept for i in block_idx]
        if idx:
            out[idx] = torch.cat([rows for block_idx, rows in kept if len(block_idx)]).cpu().numpy()
    if info is not None:
        info["path"] = path
    return out


def less_scores(model_factory, tok, pool_texts: Sequence[str], target_texts: Sequence[str],
                target_domains: Sequence[str], device: str, cfg: LessConfig, seed: int = 0,
                sketch_out: dict | None = None) -> tuple[np.ndarray, dict]:
    """LESS score of every pool record and a summary (warm-up size, epochs, LoRA size, seconds, per-record time)."""
    import time

    import torch

    start = time.perf_counter()
    torch.manual_seed(seed)
    model = model_factory()
    params = add_lora(model, cfg.rank, cfg.alpha, cfg.targets)
    n_params = int(sum(p.numel() for p in params))
    sketch = CountSketch(n_params, cfg.dim, seed)
    rng = np.random.default_rng(seed)
    n_warm = max(1, int(round(cfg.warmup_frac * len(pool_texts))))
    warm = [pool_texts[int(i)] for i in rng.permutation(len(pool_texts))[:n_warm]]
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    domains = np.asarray(target_domains).astype(str)
    names = sorted(set(domains.tolist()))
    scores = np.zeros((len(names), len(pool_texts)))
    pool_sum = np.zeros((len(pool_texts), cfg.dim), dtype=np.float32) if sketch_out is not None else None
    secs = {"warmup": 0.0, "pool_features": 0.0, "target_features": 0.0}
    paths: dict = {}
    for epoch in range(int(cfg.epochs)):
        t0 = time.perf_counter()
        model.train()
        order = rng.permutation(len(warm))
        for s in range(0, len(order), cfg.batch_size):
            opt.zero_grad()
            losses = [loss for loss in (_record_loss(model, tok, warm[int(j)], cfg.ctx, device)
                                        for j in order[s:s + cfg.batch_size]) if loss is not None]
            if losses:
                torch.stack(losses).mean().backward()
                opt.step()
        model.eval()
        secs["warmup"] += time.perf_counter() - t0
        state = {key: torch.cat([opt.state[p][key].detach().reshape(-1).to(torch.float32) for p in params])
                 for key in ("exp_avg", "exp_avg_sq")} if params[0] in opt.state else None
        t0 = time.perf_counter()
        train = gradient_features(model, tok, params, pool_texts, cfg.ctx, device, sketch, adam_state=state,
                                  grad_batch=cfg.grad_batch, info=paths)
        secs["pool_features"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        target = gradient_features(model, tok, params, target_texts, cfg.ctx, device, sketch, keep_on_device=True,
                                   grad_batch=cfg.grad_batch, info=paths)
        secs["target_features"] += time.perf_counter() - t0
        for d, name in enumerate(names):
            mean = target[domains == name].mean(axis=0)
            scores[d] += train @ (mean / (np.linalg.norm(mean) + 1e-12))
        if pool_sum is not None:
            pool_sum += train
    epochs = max(int(cfg.epochs), 1)
    summary = {"warmup_records": n_warm, "epochs": int(cfg.epochs), "lora_params": n_params, "dim": cfg.dim,
               "domains": names, "device": str(device), "seconds": time.perf_counter() - start,
               "grad_batch": int(cfg.grad_batch), "gradient_path": paths.get("path"),
               "stage_seconds": secs,
               "per_record_secs_pool": secs["pool_features"] / (epochs * max(len(pool_texts), 1)),
               "per_record_secs_target": secs["target_features"] / (epochs * max(len(target_texts), 1))}
    if sketch_out is not None:
        sketch_out["pool"] = pool_sum
    return scores.max(axis=0), summary


def less_cached(inputs: dict, seed: int, keep_sketches: bool = False) -> tuple[np.ndarray, dict]:
    """LESS scores of the text pool computed once per cell and kept in ``inputs['cache']``.

    The alignment channel of protocol v2.1 and the LESS member read the same result. With
    ``keep_sketches`` the sum over epochs of the pool sketches is kept as ``inputs['cache']['sketches']``
    (the gradient coverage space of the text track).
    """
    cache = inputs.setdefault("cache", {})
    if "scores" not in cache or (keep_sketches and "sketches" not in cache):
        cfg = inputs.get("config") or LessConfig()
        out: dict = {} if keep_sketches else None
        scores, summary = less_scores(inputs["model_factory"], inputs["tokenizer"], inputs["texts"],
                                      inputs["target_texts"], inputs["target_domains"], inputs["device"], cfg,
                                      seed=seed, sketch_out=out)
        cache.update({"scores": scores, "summary": summary})
        if out is not None:
            cache["sketches"] = out["pool"]
    return cache["scores"], cache["summary"]


@register("less", method_dir="LESS", family="new", fidelity="published-core transfer",
          requires=("less_inputs",), display="LESS")
def less_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """Full LESS order of the text pool (the track cuts it to the token budget)."""
    inputs: dict[str, Any] = ctx.extras["less_inputs"]
    shared = "scores" in inputs.get("cache", {})
    scores, summary = less_cached(inputs, ctx.seed)
    summary = {**summary, "shared_with_alignment_channel": shared}
    inputs["summary"] = summary
    info = ctx.extras.get("strategy_info")
    if isinstance(info, dict):
        info["less"] = summary
    return [int(i) for i in np.argsort(-scores, kind="stable")]
