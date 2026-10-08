"""tools/gate_replay.py on a synthetic cell written in the 96af9ab run-record schema."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from tools import gate_replay


def _write_cell(cell: Path, conf_correct: dict[str, np.ndarray], u_rank: dict[str, float], roles: dict[str, str],
                u_test: dict[str, float]) -> None:
    (cell / "candidates").mkdir(parents=True)
    n = len(next(iter(conf_correct.values())))
    target = np.arange(n) % 4
    for name, correct in conf_correct.items():
        pu = cell / "candidates" / name / "per_unit"
        pu.mkdir(parents=True)
        prediction = np.where(correct > 0, target, (target + 1) % 4)
        np.savez(pu / "conf.npz", target=target, prediction=prediction, correct=correct.astype(np.float32))
    rows = [{"name": k, "role": roles[k], "scoring_u_rank": u_rank[k], "scoring_u_con": None, "u_test": u_test[k]}
            for k in conf_correct]
    (cell / "leaderboard.json").write_text(json.dumps({"utility": "accuracy", "rows": rows,
                                                      "candidate_dirs": {k: k for k in conf_correct}}))
    (cell / "splits.json").write_text(json.dumps({"counts": {"con": 10, "rank": 10, "conf": n}, "s0": 1.0}))
    # 96af9ab config: no eprocess_weighting, gate kind margin
    (cell / "config.json").write_text(json.dumps({
        "protocol": "v2", "omniselect": {"gate": {"kind": "margin", "k_challengers": 3, "delta": 0.05, "eps": 0.0,
                                                  "margin_frac": 0.015, "n_boot": 1000, "p_beat_min": 0.9,
                                                  "text_clip": 1.0, "ts_block_steps": 0, "pair_split": "rank"}},
        "track": {"track": "vision", "dataset": "cifar100", "learner": "clip_vitb32", "seed": 0}}))
    (cell / "decision.json").write_text(json.dumps({"gate": {"kind": "margin"}, "elected": "random"}))


def test_replay_recomputes_every_gate(tmp_path):
    n = 400
    rng = np.random.default_rng(0)
    half = (rng.random(n) < 0.5).astype(float)
    better = half.copy()
    better[half == 0] = (rng.random(int((half == 0).sum())) < 0.5)   # about 75 percent correct
    conf = {"random": half, "herding": half.copy(), "fuse_a": better, "fuse_b": half.copy()}
    u_rank = {"random": 0.50, "herding": 0.49, "fuse_a": 0.505, "fuse_b": 0.60}
    roles = {"random": "reference", "herding": "reference", "fuse_a": "challenger", "fuse_b": "challenger"}
    u_test = {"random": 0.40, "herding": 0.41, "fuse_a": 0.45, "fuse_b": 0.39}
    cell = tmp_path / "R1" / "vision" / "cifar100" / "clip_vitb32" / "seed_0"
    _write_cell(cell, conf, u_rank, roles, u_test)

    result = gate_replay.replay_cell(cell)
    gates = result["gates"]
    assert result["unit_split"] == "conf" and result["reference"] == "random"
    assert result["challengers"] == ["fuse_b", "fuse_a"]
    assert gates["retain"]["elected"] == "random"
    assert gates["argmax"]["elected"] == "fuse_b" and gates["margin"]["elected"] == "fuse_b"
    # fuse_b equals random on conf, fuse_a gains about 0.25: the statistical gates adopt fuse_a
    for kind in ("lcb", "bootstrap", "eprocess"):
        assert gates[kind]["elected"] == "fuse_a", kind
        assert gates[kind]["k_tested"] == 2
    assert gates["eprocess"]["certified"] and gates["margin"]["certified"] is False
    assert result["audit"]["reference"] == "random" and [e["challenger"] for e in result["audit"]["entries"]] == [
        "fuse_b", "fuse_a"]
    ep = gates["eprocess"]["tests"]
    assert len(ep) == 2 and not ep[0]["adopted"] and ep[1]["adopted"]
    assert ep[1]["statistics"]["stop_index"] is not None
    assert abs(gates["eprocess"]["gain_over_random"] - 0.05) < 1e-12

    rows, table = gate_replay.replay_batch(tmp_path / "R1", tmp_path / "out")
    sweep = {gate_replay.sweep_name(f) for f in gate_replay.MARGIN_SWEEP}
    assert {r["gate"] for r in rows} == set(gate_replay.KINDS) | {"recorded"} | sweep
    with open(tmp_path / "out" / "gates_R1.csv") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == len(rows)
    summary = {t["gate"]: t for t in table if t["task"] == "all"}
    assert abs(summary["eprocess"]["mean_gain_over_random"] - 0.05) < 1e-12
    assert summary["retain"]["adoption_rate"] == 0.0
    assert summary["eprocess"]["certified_adoptions"] == 1 and summary["margin"]["certified_adoptions"] == 0
    per_task = {t["gate"]: t for t in table if t["task"] == "vision/cifar100"}
    reads = per_task["eprocess"]["units_read_median"]
    assert reads == gates["eprocess"]["units_read"] >= ep[1]["statistics"]["stop_index"]   # K = 3 against K = 2
    assert per_task["margin"]["adoptions"] == 1 and per_task["margin"]["units_read_median"] is None
    assert (tmp_path / "out" / "gates_summary_R1.md").read_text().startswith("# Gate replay")


def test_margin_sweep(tmp_path):
    n = 200
    half = (np.arange(n) % 2).astype(float)
    conf = {"random": half, "fuse_a": half.copy(), "fuse_b": half.copy()}
    u_rank = {"random": 0.50, "fuse_a": 0.51, "fuse_b": 0.52}      # +2% and +4% of the reference
    roles = {"random": "reference", "fuse_a": "challenger", "fuse_b": "challenger"}
    u_test = {"random": 0.40, "fuse_a": 0.42, "fuse_b": 0.39}
    cell = tmp_path / "R1" / "vision" / "cifar100" / "clip_vitb32" / "seed_0"
    _write_cell(cell, conf, u_rank, roles, u_test)
    gates = gate_replay.replay_cell(cell)["gates"]
    expected = {"margin@0": "fuse_b", "margin@0.005": "fuse_b", "margin@0.015": "fuse_b",
                "margin@0.03": "fuse_b", "margin@0.05": "random"}
    assert {k: gates[k]["elected"] for k in expected} == expected
    assert gates["margin@0.05"]["k_tested"] == 2 and not gates["margin@0.05"]["adopted"]
    rows, table = gate_replay.replay_batch(tmp_path / "R1", tmp_path / "out")
    per_task = {t["gate"]: t for t in table if t["task"] == "vision/cifar100"}
    assert per_task["margin@0.05"]["adoptions"] == 0 and per_task["margin@0"]["adoptions"] == 1
    assert abs(per_task["margin@0.05"]["mean_elected_test_utility"] - 0.40) < 1e-12
    only = gate_replay.replay_cell(cell, margin_sweep=(0.1,))["gates"]
    assert "margin@0.1" in only and "margin@0.005" not in only
