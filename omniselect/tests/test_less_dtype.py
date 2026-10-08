"""LESS under a bfloat16 base model: LoRA factors follow the base dtype, gradient features are float32."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from benchmark.Methods.LESS.method import (  # noqa: E402
    CountSketch,
    LessConfig,
    add_lora,
    gradient_features,
    less_scores,
)


class _Tok:
    def __call__(self, text, truncation=True, max_length=32, return_tensors="pt"):
        return {"input_ids": torch.tensor([[ord(c) % 64 for c in text][:max_length]])}


def _tiny_llama(dtype):
    config = transformers.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64)
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(config).to(dtype)


def test_lora_on_a_bfloat16_linear_runs_forward_and_backward():
    block = torch.nn.Module()
    block.q_proj = torch.nn.Linear(8, 8).to(torch.bfloat16)
    params = add_lora(block, rank=2, alpha=4.0, targets=("q_proj",))
    lora = block.q_proj
    assert lora.lora_a.dtype == torch.bfloat16 and lora.lora_b.dtype == torch.bfloat16
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    out = lora(x)
    assert out.dtype == torch.bfloat16 and torch.equal(out, lora.base(x))    # lora_b starts at zero
    out.float().pow(2).sum().backward()
    assert all(p.grad is not None and p.grad.dtype == torch.bfloat16 for p in params)
    assert lora.base.weight.grad is None


def test_gradient_features_are_float32_under_bfloat16():
    model = _tiny_llama(torch.bfloat16)
    params = add_lora(model, rank=4, alpha=8.0, targets=("q_proj", "v_proj"))
    sketch = CountSketch(int(sum(p.numel() for p in params)), 128, seed=0)
    feats = gradient_features(model, _Tok(), params, ["abcabc", "xyzxyz"], 32, "cpu", sketch)
    assert feats.dtype == np.float32 and np.all(np.isfinite(feats))
    assert np.allclose(np.linalg.norm(feats, axis=1), 1.0, atol=1e-5)
    assert sketch(np.ones(sketch.bucket.shape[0], dtype=np.float64)).dtype == np.float32
    scores, summary = less_scores(lambda: _tiny_llama(torch.bfloat16), _Tok(), ["abcabc", "xyzxyz", "abab", "qq qq"],
                                  ["abc", "xyz"], ["a", "b"], "cpu",
                                  LessConfig(rank=4, alpha=8.0, lr=1e-2, epochs=1, warmup_frac=0.5, batch_size=2,
                                             dim=64, ctx=32), seed=0)
    assert np.all(np.isfinite(scores)) and summary["epochs"] == 1


def _reference_cpu_features(model, tok, params, texts, ctx, device, sketch, adam_state=None, betas=(0.9, 0.999),
                            eps=1e-8, keep_on_device=False, grad_batch=1, info=None):
    """The CPU feature path of commit 56e5128 (gradients moved to the CPU, numpy Adam and numpy sketch)."""
    out = np.zeros((len(texts), sketch.dim), dtype=np.float32)
    b1, b2 = betas
    m = v = None
    if adam_state is not None:
        m = adam_state["exp_avg"].detach().to(torch.float32).cpu().numpy()
        v = adam_state["exp_avg_sq"].detach().to(torch.float32).cpu().numpy()
    for i, text in enumerate(texts):
        for p in params:
            p.grad = None
        ids = tok(text, truncation=True, max_length=int(ctx), return_tensors="pt")["input_ids"].to(device)
        if ids.size(1) < 2:
            continue
        model(input_ids=ids, labels=ids).loss.backward()
        g = torch.cat([p.grad.detach().reshape(-1).to(torch.float32).cpu() for p in params]).numpy()
        if m is not None:
            g = ((b1 * m + (1 - b1) * g) / (np.sqrt(b2 * v + (1 - b2) * g * g) + eps)).astype(np.float32)
        g = g / np.float32(np.linalg.norm(g) + 1e-12)
        z = sketch(g)
        out[i] = z / np.float32(np.linalg.norm(z) + 1e-12)
    return out


def _toy_pool(n=64):
    rng = np.random.default_rng(7)
    letters = "abcdefghijklmnopqrstuvwxyz"
    return ["".join(rng.choice(list(letters[: 4 + i % 20]), size=12 + i % 9)) for i in range(n)]


def test_device_sketch_matches_the_numpy_sketch():
    sketch = CountSketch(5000, 256, seed=3)
    v = np.random.default_rng(0).normal(size=5000).astype(np.float32)
    ref = sketch(v)
    dev = sketch.project(torch.as_tensor(v)).numpy()
    assert np.allclose(dev, ref, rtol=1e-4, atol=1e-5 * np.abs(ref).max())


def test_device_features_equal_the_cpu_reference(monkeypatch):
    import benchmark.Methods.LESS.method as less

    model = _tiny_llama(torch.bfloat16)
    params = add_lora(model, rank=4, alpha=8.0, targets=("q_proj", "v_proj"))
    with torch.no_grad():
        for p in params:
            p.add_(0.01 * torch.randn_like(p))            # a nonzero adapter
    sketch = CountSketch(int(sum(p.numel() for p in params)), 128, seed=1)
    texts = _toy_pool(16)
    state = {"exp_avg": torch.full((sketch.bucket.shape[0],), 1e-4),
             "exp_avg_sq": torch.full((sketch.bucket.shape[0],), 1e-6)}
    new = gradient_features(model, _Tok(), params, texts, 32, "cpu", sketch, adam_state=state)
    ref = _reference_cpu_features(model, _Tok(), params, texts, 32, "cpu", sketch, adam_state=state)
    assert new.dtype == np.float32
    assert np.allclose(new, ref, rtol=1e-4, atol=1e-4 * np.abs(ref).max())
    batched = gradient_features(model, _Tok(), params, texts, 32, "cpu", sketch, keep_on_device=True)
    assert np.allclose(batched, _reference_cpu_features(model, _Tok(), params, texts, 32, "cpu", sketch),
                       rtol=1e-4, atol=1e-4)

    pool = _toy_pool(64)
    cfg = LessConfig(rank=4, alpha=8.0, lr=1e-2, epochs=2, warmup_frac=0.25, batch_size=4, dim=128, ctx=32,
                     grad_batch=1)
    targets, domains = ["abcabcabc", "defdefdef", "ghighi", "abab"], ["x", "x", "y", "y"]
    scores_new, summary = less_scores(lambda: _tiny_llama(torch.bfloat16), _Tok(), pool, targets, domains, "cpu",
                                      cfg, seed=0)
    monkeypatch.setattr(less, "gradient_features", _reference_cpu_features)
    scores_ref, _ = less_scores(lambda: _tiny_llama(torch.bfloat16), _Tok(), pool, targets, domains, "cpu", cfg,
                                seed=0)
    assert np.allclose(scores_new, scores_ref, rtol=1e-4, atol=1e-5)
    k = 16
    assert set(np.argsort(-scores_new, kind="stable")[:k]) == set(np.argsort(-scores_ref, kind="stable")[:k])
    assert summary["per_record_secs_pool"] > 0 and summary["per_record_secs_target"] > 0


def _batch1_grads(model, params, texts, ctx=32):
    rows = []
    for text in texts:
        for p in params:
            p.grad = None
        ids = _Tok()(text, max_length=ctx)["input_ids"]
        model(input_ids=ids, labels=ids).loss.backward()
        rows.append(torch.cat([p.grad.detach().reshape(-1).to(torch.float32) for p in params]))
    for p in params:
        p.grad = None
    return torch.stack(rows)


def _adapted(dtype):
    model = _tiny_llama(dtype)
    params = add_lora(model, rank=4, alpha=8.0, targets=("q_proj", "v_proj"))
    with torch.no_grad():
        for p in params:
            p.add_(0.05 * torch.randn_like(p))
    return model, params


def test_batched_per_record_gradients_equal_the_batch1_path():
    from benchmark.Methods.LESS.method import per_sample_lora_grads

    texts = ["abcabcabcabcab", "xyz", "hello there general", "q", "mnopmnopmnopmnopmnop", "abab"]  # 4 lengths, 1 short
    model, params = _adapted(torch.float32)
    block, valid = per_sample_lora_grads(model, _Tok(), texts, 32, "cpu")
    assert valid == [True, True, True, False, True, True] and block.dtype == torch.float32
    assert torch.all(block[3] == 0)
    ref = _batch1_grads(model, params, [t for t, ok in zip(texts, valid) if ok])
    got = block[torch.tensor(valid)]
    rel = torch.linalg.vector_norm(got - ref, dim=1) / torch.linalg.vector_norm(ref, dim=1)
    assert float(rel.max()) < 1e-4
    # bfloat16: the batched and single-record kernels round differently, agreement is at bfloat16 precision
    model16, params16 = _adapted(torch.bfloat16)
    block16, _ = per_sample_lora_grads(model16, _Tok(), texts, 32, "cpu")
    ref16 = _batch1_grads(model16, params16, [t for t, ok in zip(texts, valid) if ok])
    diff16 = block16[torch.tensor(valid)] - ref16
    rel16 = torch.linalg.vector_norm(diff16, dim=1) / torch.linalg.vector_norm(ref16, dim=1)
    assert float(rel16.max()) < 3e-2


def test_batched_sketches_and_top16_match_the_cpu_reference(monkeypatch):
    import benchmark.Methods.LESS.method as less

    model, params = _adapted(torch.float32)
    sketch = CountSketch(int(sum(p.numel() for p in params)), 128, seed=1)
    texts = _toy_pool(24)
    state = {"exp_avg": torch.full((sketch.bucket.shape[0],), 1e-4),
             "exp_avg_sq": torch.full((sketch.bucket.shape[0],), 1e-6)}
    info: dict = {}
    batched = gradient_features(model, _Tok(), params, texts, 32, "cpu", sketch, adam_state=state, grad_batch=8,
                                info=info)
    ref = _reference_cpu_features(model, _Tok(), params, texts, 32, "cpu", sketch, adam_state=state)
    assert info["path"] == "batched"
    assert np.allclose(batched, ref, rtol=1e-4, atol=1e-4 * np.abs(ref).max())

    pool = _toy_pool(64)
    cfg = LessConfig(rank=4, alpha=8.0, lr=1e-2, epochs=2, warmup_frac=0.25, batch_size=4, dim=128, ctx=32,
                     grad_batch=16)
    targets, domains = ["abcabcabc", "defdefdef", "ghighi", "abab"], ["x", "x", "y", "y"]
    scores_new, summary = less_scores(lambda: _tiny_llama(torch.bfloat16), _Tok(), pool, targets, domains, "cpu",
                                      cfg, seed=0)
    assert summary["gradient_path"] == "batched" and summary["grad_batch"] == 16
    monkeypatch.setattr(less, "gradient_features", _reference_cpu_features)
    scores_ref, _ = less_scores(lambda: _tiny_llama(torch.bfloat16), _Tok(), pool, targets, domains, "cpu", cfg,
                                seed=0)
    k = 16
    assert set(np.argsort(-scores_new, kind="stable")[:k]) == set(np.argsort(-scores_ref, kind="stable")[:k])


def test_batched_path_falls_back_to_batch1(monkeypatch):
    import benchmark.Methods.LESS.method as less

    def broken(*args, **kwargs):
        raise RuntimeError("vmap-like failure")

    model, params = _adapted(torch.float32)
    sketch = CountSketch(int(sum(p.numel() for p in params)), 64, seed=0)
    monkeypatch.setattr(less, "per_sample_lora_grads", broken)
    info: dict = {}
    feats = less.gradient_features(model, _Tok(), params, _toy_pool(5), 32, "cpu", sketch, grad_batch=4, info=info)
    assert info["path"].startswith("batch1") and "RuntimeError" in info["path"]
    assert np.allclose(np.linalg.norm(feats, axis=1), 1.0, atol=1e-5)
