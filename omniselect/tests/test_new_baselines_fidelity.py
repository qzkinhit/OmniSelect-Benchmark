"""Fidelity checks of the R5 baselines: each shows the defining mechanism of its paper on controlled data.

- GLISTER: with clean validation labels, the Taylor gains avoid pool records with flipped labels.
- GRAD-MATCH: the OMP-weighted subset gradient matches the full training gradient better than a
  random subset of the same size, and OMP recovers a two-atom target exactly.
- InfoMax: the redundancy term removes near-duplicates that importance top-k keeps.
- Clean oracle: draws only tag-clean records while they last.
- LESS: LoRA starts at the base model, the count sketch keeps inner products, the score runs end to end.
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmark.Methods._gradients import LastLayer, append_bias, gram
from benchmark.Methods.CleanOracle.method import clean_oracle
from benchmark.Methods.GLISTER.method import glister
from benchmark.Methods.GRADMATCH.method import gradmatch, omp_kernel
from benchmark.Methods.InfoMax.method import infomax


def _noisy_softmax_problem(seed: int = 0, n: int = 600, n_val: int = 200, flip: float = 0.3):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(4, 6)) * 3
    y = rng.integers(0, 4, n)
    X = centers[y] + rng.normal(size=(n, 6))
    yv = rng.integers(0, 4, n_val)
    Xv = centers[yv] + rng.normal(size=(n_val, 6))
    flipped = rng.random(n) < flip
    obs = np.where(flipped, (y + 1 + rng.integers(0, 3, n)) % 4, y)
    from sklearn.linear_model import LogisticRegression

    from benchmark.Methods._gradients import softmax_theta

    head = LogisticRegression(max_iter=50).fit(X, obs)
    theta = softmax_theta(head.coef_.T, head.intercept_, head.classes_, 4)
    model = LastLayer("softmax", append_bias(X), obs, theta, append_bias(Xv), yv, labels=obs)
    return model, flipped


def test_glister_avoids_flipped_labels():
    model, flipped = _noisy_softmax_problem()
    k = 200
    sel = glister(model, k, rounds=10)
    assert len(sel) == k == len(set(sel))
    assert flipped[sel].mean() < 0.5 * flipped.mean()


def test_gradmatch_matches_the_full_gradient_better_than_random():
    model, _ = _noisy_softmax_problem(seed=1)
    k = 80
    chosen, weights = gradmatch(model, k, seed=0)
    assert len(chosen) == k == len(set(chosen))
    errors = model.errors(model.phi, model.target)
    labels = model.labels
    rng = np.random.default_rng(0)

    def residual(idx, w):
        total = 0.0
        for c in np.unique(labels):
            members = np.flatnonzero(labels == c)
            full = model.phi[members].T @ errors[members]
            mask = labels[idx] == c
            part = model.phi[idx[mask]].T @ (errors[idx[mask]] * w[mask][:, None])
            total += float(np.linalg.norm(full - part) ** 2)
        return total

    idx = np.asarray(chosen)
    rand = rng.choice(len(labels), size=k, replace=False)
    scale = len(labels) / k
    assert residual(idx, weights) < residual(rand, np.full(k, scale))


def test_omp_kernel_recovers_two_atoms():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(40, 30))
    A /= np.linalg.norm(A, axis=0)
    y = 2.0 * A[:, 4] + 0.5 * A[:, 17]
    K = A.T @ A
    support, x = omp_kernel(K, A.T @ y, float(y @ y), budget=5, lam=1e-8)
    assert sorted(support[:2]) == [4, 17] and len(support) == 2
    assert np.allclose(sorted(x), [0.5, 2.0], atol=1e-4)


def test_gradmatch_regression_group_and_kernel():
    rng = np.random.default_rng(4)
    phi = append_bias(rng.normal(size=(300, 5)))
    target = phi @ rng.normal(size=(6, 2)) + 0.1 * rng.normal(size=(300, 2))
    model = LastLayer("squared", phi, target, np.zeros((6, 2)), phi[:50], target[:50])
    chosen, weights = gradmatch(model, 30, seed=1)
    assert len(chosen) == 30 == len(set(chosen)) and np.all(weights >= 0)
    e = model.errors(phi, target)
    g = np.einsum("nd,nm->ndm", phi, e).reshape(300, -1)
    assert np.allclose(gram(phi[:7], e[:7], phi[:5], e[:5]), g[:7] @ g[:5].T)


def test_infomax_drops_near_duplicates():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(100, 8))
    dups = base[:50] + 1e-3 * rng.normal(size=(50, 8))
    X = np.vstack([base, dups])
    importance = np.concatenate([rng.random(100) * 0.2 + 0.8, np.full(50, 0.95)])   # duplicates look important
    k = 60
    sel, x = infomax(importance, X, k, beta=1.0, knn=5)
    top = np.argsort(-importance, kind="stable")[:k]
    is_dup = np.arange(150) >= 100
    assert len(sel) == k == len(set(sel)) and abs(x.sum() - k) < 1e-6
    assert is_dup[sel].sum() < is_dup[top].sum()


def test_clean_oracle_reads_tags():
    tags = np.array(["high", "flip", "high", "dup", "high", "hard"])
    sel = clean_oracle(tags, 3, seed=0)
    assert sorted(sel) == [0, 2, 4]
    more = clean_oracle(tags, 5, seed=0)
    assert set(more[:3]) == {0, 2, 4} and len(set(more)) == 5
    assert len(clean_oracle(tags, 2, seed=0, full_order=True)) == 6


def test_less_lora_sketch_and_scores():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from benchmark.Methods.LESS.method import CountSketch, LessConfig, add_lora, less_scores

    sketch = CountSketch(5000, 2048, seed=0)
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=5000), rng.normal(size=5000)
    b = 0.8 * a + 0.6 * b
    assert abs(sketch(a) @ sketch(b) - a @ b) < 0.1 * np.linalg.norm(a) * np.linalg.norm(b)

    config = transformers.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64)

    def factory():
        torch.manual_seed(0)
        return transformers.LlamaForCausalLM(config)

    class Tok:
        def __call__(self, text, truncation=True, max_length=32, return_tensors="pt"):
            ids = [ord(c) % 64 for c in text][:max_length]
            return {"input_ids": torch.tensor([ids])}

    model = factory()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    before = model(input_ids=ids).logits.detach()
    params = add_lora(model, rank=4, alpha=8.0, targets=("q_proj", "v_proj"))
    assert len(params) == 8 and torch.allclose(model(input_ids=ids).logits, before)
    pool = ["abcabcabc", "xyzxyzxyz", "abcabcabd", "hello world", "abababab", "qqqqqqqq"]
    scores, summary = less_scores(factory, Tok(), pool, ["abcabc", "xyzxyz"], ["a", "b"], "cpu",
                                  LessConfig(rank=4, alpha=8.0, lr=1e-2, epochs=2, warmup_frac=0.5, batch_size=2,
                                             dim=256, ctx=32), seed=0)
    assert scores.shape == (6,) and np.all(np.isfinite(scores)) and summary["epochs"] == 2
