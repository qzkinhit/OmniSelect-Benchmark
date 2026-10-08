"""Queue templates of the paper's batches, the queue runner and the proxy-ranking tool."""
from __future__ import annotations

import json
import re
from pathlib import Path

from run_omniselect import run_queue
from run_omniselect.make_queues import PAPER_BATCHES, RECORD_BATCH, TASK_LABELS, build
from tools import proxy_ranking

LINE = re.compile(r"^python -m tracks\.common\.experiment .*   # cell=\S+ name=\S+$")
ROOT = Path(__file__).resolve().parents[2]


def test_paper_queues_have_unique_cells_and_the_paper_seed_layout():
    for batch, (protocol, _) in PAPER_BATCHES.items():
        lines = [x for x in build(batch) if not x.startswith("#")]
        assert lines and all(LINE.match(x) for x in lines), batch
        assert all(f"--protocol {protocol} --batch {RECORD_BATCH[batch]} " in x for x in lines), batch
        cells = [re.search(r"cell=(\S+)", x).group(1) for x in lines]
        names = [re.search(r"name=(\S+)", x).group(1) for x in lines]
        assert len(cells) == len(set(cells)) == len(set(names)) == 12 * 3, batch
        assert not any(c.startswith("/") for c in cells)
        reported = json.loads((ROOT / "results/paper/seeds_reported.json").read_text())["seeds"]
        for (track, dataset), label in TASK_LABELS.items():
            task_lines = [x for x in lines if f"--track {track} --dataset {dataset} " in x]
            assert sorted(int(re.search(r"--seed (\d+)", x).group(1)) for x in task_lines) == reported[label]
        heavy = [x for x in lines if "--track text" in x or "--track native" in x]
        assert sorted({int(re.search(r"--seed (\d+)", x).group(1)) for x in heavy}) == [0, 1, 2]
        assert all("selection_device=cuda" in x for x in lines if "--track native" in x)
        committed = (ROOT / "run_omniselect" / "queues" / f"{batch}.txt").read_text().splitlines()
        assert committed == build(batch), f"queues/{batch}.txt is stale, run python -m run_omniselect.make_queues"


def test_robustness_queue_and_results_root():
    lines = [x for x in build("robustness", protocol="v2") if not x.startswith("#")]
    assert sum("robustness/ratio_" in x for x in lines) == 7 * 3 * 3
    assert all(" --protocol v2 " in x and "store.per_unit_scope=finalists" in x for x in lines)
    assert any("--learner chronos_small" in x and "/chronos_small/" in x for x in lines)
    assert not any("--learner clip_vitb32" in x for x in lines)
    cells = [re.search(r"cell=(\S+)", x).group(1) for x in lines]
    assert len(cells) == len(set(cells))
    moved = [x for x in build("main_v2_3", root="/data/runs", seeds="3,4,5", heavy_seeds="3,4,5") if not x.startswith("#")]
    assert all("--out /data/runs" in x and "cell=/data/runs/main/" in x for x in moved)
    assert sorted({int(re.search(r"--seed (\d+)", x).group(1)) for x in moved}) == [3, 4, 5]


def test_queue_runner_skips_complete_cells_and_counts_exit_codes(tmp_path):
    done = tmp_path / "cells" / "a"
    done.mkdir(parents=True)
    (done / "decision.json").write_text("{}")
    queue_file = tmp_path / "q.txt"
    queue_file.write_text("\n".join([
        "# comment",
        f"exit 1   # cell={done} name=complete",
        "exit 0   # name=ok",
        "exit 3   # name=unavailable",
        "exit 2",
    ]) + "\n")
    jobs = run_queue.parse_queue(queue_file)
    assert [j.name for j in jobs] == ["complete", "ok", "unavailable", "q_5"]
    assert run_queue.is_complete(jobs[0], tmp_path) and not run_queue.is_complete(jobs[1], tmp_path)
    code = run_queue.main([str(queue_file), "--jobs", "2", "--log-dir", str(tmp_path / "logs")])
    assert code == 1
    assert (tmp_path / "logs" / "ok.log").is_file() and not (tmp_path / "logs" / "complete.log").exists()


def _cell(root: Path, learner: str, u_rank: list[float], u_test: list[float], elected: str) -> None:
    cell = root / "vision" / "cifar100" / learner / "seed_0"
    (cell / "candidates").mkdir(parents=True)
    rows = [{"name": n, "sel_sha12": f"sha{n}", "u_rank": r, "u_test": t}
            for n, r, t in zip("abcd", u_rank, u_test)]
    (cell / "leaderboard.json").write_text(json.dumps({"rows": rows}))
    (cell / "decision.json").write_text(json.dumps({"elected": elected}))
    (cell / "config.json").write_text(json.dumps({"track": {"track": "vision", "dataset": "cifar100",
                                                            "seed": 0, "learner": learner}}))


def test_proxy_ranking_pairs_learners(tmp_path):
    _cell(tmp_path / "cheap", "clip_vitb32", [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], "d")
    _cell(tmp_path / "exp", "resnet18_scratch", [0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.4, 0.3], "a")
    assert proxy_ranking.main(["--cheap", str(tmp_path / "cheap"), "--expensive", str(tmp_path / "exp"),
                               "--out", str(tmp_path / "out")]) == 0
    row = (tmp_path / "out" / "proxy_ranking.csv").read_text().splitlines()
    assert len(row) == 2
    result = proxy_ranking.compare(tmp_path / "cheap" / "vision" / "cifar100" / "clip_vitb32" / "seed_0",
                                   tmp_path / "exp" / "vision" / "cifar100" / "resnet18_scratch" / "seed_0")
    assert result["kendall_tau_rank"] == -1.0 and abs(result["regret_of_cheap_election"] + 0.1) < 1e-12
    assert result["same_election"] is False and result["common_candidates"] == 4
