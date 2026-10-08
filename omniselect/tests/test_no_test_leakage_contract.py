"""No test split reaches the controller: test outputs are requested only by reported fits.

The driver is run on a small TEP cell under both protocols with FitCache.evaluate wrapped. Every
request of the 'test' split must come from the 'report' stage, and every report request must come
after the last controller request (con, rank, conf, member), so the election is fixed before any
test output exists.
"""
from __future__ import annotations

import pytest

from omniselect.core.adjudication import cache as cache_module
from tracks.common.experiment import load_track, resolve_configs, run_cell


@pytest.mark.parametrize("protocol", ["canonical", "v2"])
def test_controller_never_requests_test_outputs(monkeypatch, tmp_path, protocol):
    calls: list[tuple[str, tuple]] = []
    original = cache_module.FitCache.evaluate

    def spy(self, subset, stage, splits):
        calls.append((stage, tuple(splits)))
        return original(self, subset, stage, splits)

    monkeypatch.setattr(cache_module.FitCache, "evaluate", spy)
    track = load_track("process")
    tcfg, ocfg = resolve_configs(track, "tep21", learner="mlp", seed=0, protocol=protocol, smoke=True,
                                 track_overrides={"pool_n": 200, "val_n": 200, "test_n": 150, "mlp_max_iter": 20},
                                 omni_overrides={})
    run_cell(track, tcfg, ocfg, out_root=tmp_path, batch="leak", cli=["test"])
    assert calls, "the driver made no learner requests"
    for stage, splits in calls:
        if "test" in splits:
            assert stage == "report", (stage, splits)
    first_report = next(i for i, (stage, _) in enumerate(calls) if stage == "report")
    assert all(stage == "report" for stage, _ in calls[first_report:])
