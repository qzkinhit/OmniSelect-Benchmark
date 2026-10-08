"""Thin wrappers that load one task's data the way its track does, for notebooks and audits.

``load_task(track, dataset, seed, protocol, smoke)`` resolves the configs exactly as the driver
does and returns the track's TaskData (pool, validation and test arrays, tags, splits). Nothing is
trained and nothing is written. The loaders themselves live in benchmark/Data.
"""
from __future__ import annotations

from typing import Any

from tracks.common.experiment import load_track, resolve_configs
from tracks.common.task import TaskData


def load_task(track: str, dataset: str, seed: int = 0, protocol: str = "v2", smoke: bool = False,
              **track_overrides: Any) -> TaskData:
    """TaskData of one cell without signals, learners or a run record."""
    t = load_track(track)
    tcfg, ocfg = resolve_configs(t, dataset, learner=None, seed=seed, protocol=protocol, smoke=smoke,
                                 track_overrides=track_overrides, omni_overrides={})
    return t.load(tcfg, ocfg)
