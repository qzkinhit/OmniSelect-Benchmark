"""Entry points of the restructured benchmark: shell syntax, task inventory, pins, protocol overrides."""
from __future__ import annotations

import subprocess
from pathlib import Path

from omniselect.config.config import OmniSelectConfig
from tracks.common.experiment import TRACKS, load_track, main, resolve_configs

ROOT = Path(__file__).resolve().parents[2]


def test_shell_entrypoints_have_valid_syntax():
    paths = [ROOT / "benchmark" / "run_all.sh", *sorted((ROOT / "benchmark" / "MethodsRunScript").glob("*/run.sh"))]
    for extra in ("run.sh", "run_omniselect/run.sh", "run_omniselect/run_cell.sh", "run_omniselect/run_experiment.sh"):
        if (ROOT / extra).is_file():
            paths.append(ROOT / extra)
    subprocess.run(["bash", "-n", *map(str, paths)], check=True)


def test_task_inventory_covers_the_twelve_tasks_and_the_clip_instance():
    inventory = {name: sorted(load_track(name).datasets) for name in TRACKS}
    assert inventory["vision"] == ["cifar100", "cifar100n", "cifar10_clip"]
    assert inventory["native"] == ["cifar10", "imagenet100"]
    assert inventory["timeseries"] == ["ETTh1", "ETTh2", "ETTm1", "daisy_cstr", "daisy_steamgen"]
    assert inventory["process"] == ["tep21"]
    assert inventory["tabular"] == ["electricity"]
    assert inventory["text"] == ["five_domain"]


def test_vision_sources_and_encoder_are_pinned():
    source = (ROOT / "benchmark" / "Data" / "vision.py").read_text()
    for revision in ("3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", "aadb3af77e9048adbea6b47c21a81e47dd092ae5",
                     "0b2714987fa478483af9968de7c934580d0bb9a2"):
        assert revision in source


def test_protocol_overrides_of_native_and_text():
    native = load_track("native")
    _, ocfg = resolve_configs(native, "cifar10", learner=None, seed=0, protocol="canonical", smoke=True,
                              track_overrides={}, omni_overrides={})
    assert ocfg.gate.kind == "argmax" and ocfg.splits.mode == "rank_only" and ocfg.cache.by_subset_hash
    text = load_track("text")
    _, ocfg = resolve_configs(text, "five_domain", learner=None, seed=0, protocol="canonical", smoke=True,
                              track_overrides={}, omni_overrides={})
    assert ocfg.gate.kind == "lcb" and ocfg.gate.pair_split == "con" and ocfg.gate.split == "rank"
    _, ocfg = resolve_configs(text, "five_domain", learner=None, seed=0, protocol="v2", smoke=True,
                              track_overrides={}, omni_overrides={})
    assert ocfg.gate.kind == "margin" and ocfg.gate.split == "rank" and ocfg.gate.k_challengers == 3
    assert ocfg.gate.audit and ocfg.gate.audit_split == "conf" and ocfg.splits.mode == "three_way"


def test_cli_overrides_reach_both_configs():
    track = load_track("process")
    tcfg, ocfg = resolve_configs(track, "tep21", learner="rf", seed=3, protocol="v2", smoke=False,
                                 track_overrides={"pool_n": "500"}, omni_overrides={"gate.kind": "lcb"})
    assert tcfg.pool_n == 500 and tcfg.learner == "rf" and tcfg.seed == 3
    assert ocfg.gate.kind == "lcb" and ocfg.seed == 3
    assert OmniSelectConfig.from_dict(ocfg.to_dict()) == ocfg


def test_missing_dataset_is_a_skip_with_exit_code_3(tmp_path):
    code = main(["--track", "native", "--dataset", "cifar10", "--smoke", "--out", str(tmp_path),
                 "--track-set", f"data_root={tmp_path / 'nodata'}"])
    assert code == 3
