"""Mechanism and fairness checks for explicitly named non-classification adapters."""
from __future__ import annotations

import numpy as np
import pytest

from benchmark.Methods.CCS.method import ccs
from benchmark.Methods.ScoreAdapt.method import (
    ADAPT_METHODS, ccs_by_score, ccs_token_order, squared_error_scores,
)
from omniselect.core.portfolio.membership import membership
from omniselect.core.portfolio.registry import SelectionContext, get, missing_inputs
from tracks.text.pool import stratified_cut


def test_forecast_gradient_norm_matches_autograd():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(10)
    # Two explicit constant columns represent DLinear's two separate biases.
    phi = np.c_[rng.normal(size=(7, 3)), np.ones(7), rng.normal(size=(7, 3)), np.ones(7)]
    theta = torch.tensor(rng.normal(size=(8, 4)), dtype=torch.float64, requires_grad=True)
    target = rng.normal(size=(7, 4))
    prediction = torch.tensor(phi) @ theta
    scores = squared_error_scores(phi, prediction.detach().numpy(), target)
    expected = []
    for i in range(len(phi)):
        loss = 0.5 * (prediction[i] - torch.tensor(target[i])).square().sum()
        gradient, = torch.autograd.grad(loss, theta, retain_graph=True)
        expected.append(gradient.norm().item())
    np.testing.assert_allclose(scores["grand"], expected, rtol=1e-12)
    np.testing.assert_allclose(scores["el2n"], np.linalg.norm(prediction.detach().numpy() - target, axis=1))


def test_token_head_scores_match_individual_token_autograd_and_length_normalization():
    torch = pytest.importorskip("torch")
    from tracks.text.score_adapt import token_error_gradient_scores

    torch.manual_seed(4)
    head = torch.nn.Linear(4, 6)
    hidden = torch.randn(5, 4)
    target = torch.tensor([1, 3, 4, 0, 2])
    logits = head(hidden)
    result = token_error_gradient_scores(logits, hidden, target, head_has_bias=True)
    errors, norms = [], []
    for i in range(5):
        loss = torch.nn.functional.cross_entropy(logits[i:i+1], target[i:i+1])
        grads = torch.autograd.grad(loss, tuple(head.parameters()), retain_graph=True)
        norms.append(sum(g.square().sum() for g in grads))
        truth = torch.nn.functional.one_hot(target[i], 6)
        errors.append((logits[i].softmax(-1) - truth).square().sum())
    expected = torch.stack((torch.stack(errors).mean().sqrt(), torch.stack(norms).mean().sqrt()))
    torch.testing.assert_close(result, expected)
    repeated = token_error_gradient_scores(logits.repeat(2, 1), hidden.repeat(2, 1), target.repeat(2),
                                          head_has_bias=True)
    torch.testing.assert_close(result, repeated)
    # The declared token RMS gradient is deliberately not the document-mean gradient.
    mean_loss = torch.nn.functional.cross_entropy(logits, target)
    document_grads = torch.autograd.grad(mean_loss, tuple(head.parameters()))
    document_norm = sum(g.square().sum() for g in document_grads).sqrt()
    assert not torch.isclose(result[1], document_norm)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ccs_record_core_equals_existing_classification_implementation(seed):
    rng = np.random.default_rng(23)
    logits = rng.normal(size=(400, 5))
    proba = np.exp(logits - logits.max(1, keepdims=True))
    proba /= proba.sum(1, keepdims=True)
    labels = rng.integers(0, 5, size=400)
    error = np.linalg.norm(proba - np.eye(5)[labels], axis=1)
    assert ccs_by_score(error, 120, seed=seed) == ccs(proba, labels, 120, is_logits=False, seed=seed)


def test_ccs_token_cutoff_stratification_and_existing_budget_cut():
    n = 500
    domains = ["a"] * n + ["b"] * n
    difficulty = np.tile(np.linspace(0, 1, n), 2)
    tokens = np.tile(np.arange(n) % 19 + 15, 2)
    budgets = {d: int(tokens[:n].sum() * 0.5) for d in ("a", "b")}
    order = ccs_token_order(difficulty, domains, tokens, budgets, seed=1)
    assert len(order) == 2*n and set(order) == set(range(2*n))
    assert order == ccs_token_order(difficulty, domains, tokens, budgets, seed=1)
    selection = stratified_cut(order, domains, list(tokens), budgets)
    for domain in budgets:
        chosen = [i for i in selection if domains[i] == domain]
        assert tokens[chosen].sum() >= budgets[domain]
        assert tokens[chosen[:-1]].sum() < budgets[domain]
        assert np.all(np.asarray(chosen) % n < int(n * 0.9))
        # The selected profile spans the retained difficulty range.
        assert difficulty[chosen].min() < 0.1 and difficulty[chosen].max() > 0.8


def test_ccs_does_not_restore_discarded_hard_records_to_fill_token_budget():
    tokens = [1] * 90 + [1000] * 10
    with pytest.raises(ValueError, match="cannot supply"):
        ccs_token_order(np.arange(100), ["a"] * 100, tokens, {"a": 5000})


def test_registered_adapters_are_explicit_and_read_no_validation_callback():
    n = 100
    calls = []
    ctx = SelectionContext(n=n, seed=2, features=np.ones((n, 1)),
                           construction_gain=lambda _: calls.append("unexpected"), extras={
                               "adapt_scores": {"el2n": np.arange(n), "grand": np.arange(n)[::-1],
                                                "info": {"source": "fixture"}}, "strategy_info": {}})
    for name in ADAPT_METHODS:
        assert missing_inputs(get(name), ctx) == []
        selected = get(name).fn(ctx, 30)
        assert len(selected) == len(set(selected)) == 30
        assert ctx.extras["strategy_info"][name]["adaptation"]
    assert not calls
    assert get("el2n_adapt").fn(ctx, 30) != get("grand_adapt").fn(ctx, 30)
    for track in ("text", "timeseries"):
        rows = membership(track, "unified_v22")
        assert not ADAPT_METHODS & {r.name for r in rows if r.included}
        assert not {"el2n", "grand", "ccs"} & {r.name for r in rows if r.included}


def test_adapters_require_explicit_scores():
    ctx = SelectionContext(n=10, seed=0, features=np.ones((10, 2)))
    for name in ADAPT_METHODS:
        assert missing_inputs(get(name), ctx) == ["adapt_scores"]


def test_text_forward_and_cache_are_bound_to_pool_content(tmp_path):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from tracks.text.score_adapt import frozen_text_scores

    class Encoding(dict):
        def to(self, device):
            return Encoding({key: value.to(device) for key, value in self.items()})

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(9, 4)

        def forward(self, input_ids, **_):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Core()
            self.head = torch.nn.Linear(4, 9, bias=False)

        def get_output_embeddings(self):
            return self.head

    def tokenize(text, **_):
        return Encoding(input_ids=torch.tensor([[int(c) for c in text]]))

    pool_path = tmp_path / "pool.jsonl"
    pool_path.write_text("registered fixture content")
    lm = SimpleNamespace(model=Model, model_name="fixture", revision="fixture-v1", device="cpu", tok=tokenize)
    config = SimpleNamespace(ctx=8, data_root=str(tmp_path))
    records = [SimpleNamespace(id="a", text="12345"), SimpleNamespace(id="b", text="01234")]
    first = frozen_text_scores(lm, records, config, str(pool_path))
    assert first["el2n"].shape == first["grand"].shape == (2,)

    def fail_model():
        raise AssertionError("new forward requested")

    lm.model = fail_model
    cached = frozen_text_scores(lm, records, config, str(pool_path))
    np.testing.assert_array_equal(first["el2n"], cached["el2n"])
    pool_path.write_text("different pool content, unchanged record IDs")
    with pytest.raises(AssertionError, match="new forward"):
        frozen_text_scores(lm, records, config, str(pool_path))


def test_forecast_standalone_preserves_random_pairing_and_saves_all_outputs(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from tracks.common.experiment import resolve_configs, run_cell
    from tracks.timeseries import dlinear

    raw = np.sin(np.arange(1000) / 11) + 0.1 * np.cos(np.arange(1000) / 3)
    monkeypatch.setattr(dlinear.tsdata, "load_series", lambda *_: (raw, "synthetic-series"))
    track = dlinear.ForecastTrack()
    common = {"pool_n": 40, "val_n": 30, "test_n": 20, "L": 8, "H": 2, "epochs": 1,
              "scoring_epochs": 1, "reference_epochs": 1, "last_layer_warm": 1, "device": "cpu",
              "knn": 3, "methods": "random"}
    omni = {"store.per_unit_scope": "all", "save_models": "all"}
    baseline_cfg, cfg = resolve_configs(track, "ETTh1", learner=None, seed=1, protocol="v2_2", smoke=False,
                                        track_overrides=common, omni_overrides=omni)
    run_cell(track, baseline_cfg, cfg, out_root=tmp_path, batch="original", standalone=True)
    adapt_cfg = baseline_cfg.with_overrides(methods="random,el2n_adapt,grand_adapt,ccs_adapt")
    run_cell(track, adapt_cfg, cfg, out_root=tmp_path, batch="adapt", standalone=True)
    old = tmp_path / "original/timeseries/ETTh1/dlinear/seed_1"
    new = tmp_path / "adapt/timeseries/ETTh1/dlinear/seed_1"
    import json
    import pathlib
    assert json.loads((old / "splits.json").read_text()) == json.loads((new / "splits.json").read_text())
    # Locate the driver's indexed candidate directories rather than assuming their slug.
    old_random = next(p.parent for p in old.rglob("scores.json") if p.parent.name == "random")
    new_random = next(p.parent for p in new.rglob("scores.json") if p.parent.name == "random")
    with np.load(old_random / "selection.npz") as left, np.load(new_random / "selection.npz") as right:
        np.testing.assert_array_equal(left["idx"], right["idx"])
    assert json.loads((old_random / "scores.json").read_text())["utility"] == \
        json.loads((new_random / "scores.json").read_text())["utility"]
    for name in ("random", "el2n_adapt", "grand_adapt", "ccs_adapt"):
        folder = next(p.parent for p in new.rglob("scores.json") if p.parent.name == name)
        score = json.loads((folder / "scores.json").read_text())
        assert pathlib.Path(score["model_path"]).is_file()
        assert set(score["utility"]) == {"con", "rank", "conf", "test"}
        assert list(folder.rglob("*.npz"))
