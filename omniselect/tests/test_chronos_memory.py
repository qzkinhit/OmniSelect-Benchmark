"""Chronos keeps the train-once fit on CPU and bounds inference batches."""
from types import SimpleNamespace

import numpy as np
import pytest

from omniselect.core.adjudication.cache import FitCache
from tracks.common.downstream import StagePlan
from tracks.timeseries.chronos import ChronosForecaster
from tracks.timeseries.dlinear import ForecastConfig, ForecastLearner, predict


def test_chronos_cpu_cache_preserves_predictions_and_fit_identity(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from tracks.timeseries import chronos

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(7)
    x = rng.normal(size=(23, 4)).astype(np.float32)
    y = rng.normal(size=(23, 2)).astype(np.float32)
    calls, batches = [], []

    class TinyQuantileModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, context):
            batches.append(len(context))
            return SimpleNamespace(quantile_preds=self.linear(context).unsqueeze(1))

    def train(X, Y, dev, seed, epochs, tcfg, val=None):
        calls.append((X.copy(), seed, epochs))
        torch.manual_seed(seed)
        model = TinyQuantileModel().to(dev)
        # Simulate the unused gradients left by the last training step.
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        return ChronosForecaster(model.eval(), 2)

    monkeypatch.setattr(chronos, "train_chronos", train)
    plan = StagePlan(stages={s: {"epochs": 3, "seed": 17} for s in ("con", "rank", "report")},
                     reported={"epochs": 3}, schedule_key="epochs")
    cfg = ForecastConfig(dataset="ETTh1", chronos_predict_batch=5)
    learner = ForecastLearner(plan, "chronos_small", x, y,
                              {s: (x, y, np.arange(len(x))) for s in ("con", "rank", "test")}, device, cfg)
    cache = FitCache(learner, enabled=True)
    entry = cache.evaluate([3, 1, 2], "con", ["con"])
    assert max(batches) == 5 and len(calls) == 1
    assert all(p.device.type == "cpu" and p.grad is None for p in entry.model.model.parameters())
    before = {k: v.clone() for k, v in entry.model.state_dict().items()}
    report = cache.evaluate([1, 2, 3], "report", ["rank", "test"])
    assert report is entry and len(calls) == 1 and calls[0][1:] == (17, 3)
    assert np.array_equal(calls[0][0], x[[1, 2, 3]])
    for name, value in report.model.state_dict().items():
        assert torch.equal(value, before[name]) and value.device.type == "cpu"
    expected = predict(entry.model.to(device), x, device)
    entry.model.to("cpu")
    np.testing.assert_allclose(entry.per_unit["test"]["prediction"], expected, rtol=1e-6, atol=1e-7)
    path = learner.save(entry.model, tmp_path, "elected")
    assert all(torch.equal(v, before[k]) for k, v in torch.load(path, weights_only=True).items())


def test_chronos_score_offloads_after_failure(monkeypatch):
    from tracks.timeseries import dlinear

    destinations = []
    model = SimpleNamespace(to=lambda device: destinations.append(device))
    x, y = np.ones((3, 4)), np.ones((3, 2))
    cfg = ForecastConfig(dataset="ETTh1", chronos_predict_batch=2)
    learner = ForecastLearner(None, "chronos_small", x, y, {"test": (x, y, np.arange(3))}, "cuda", cfg)

    def fail(*args):
        raise RuntimeError("inference failure")

    monkeypatch.setattr(dlinear, "predict", fail)
    with pytest.raises(RuntimeError, match="inference failure"):
        learner.score(model, "test")
    assert destinations == ["cuda", "cpu"]
