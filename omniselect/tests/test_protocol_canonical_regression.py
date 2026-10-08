"""Canonical protocol regression on a small TEP fixture with pinned selections and election.

The pinned values were produced by the 2026-07 runner scripts/run_tep_experiment.py at commit
cd7a6c7 (POOL_N=400 VAL_N=300 TEST_N=300 MODEL=mlp PAIRED_RNG=1 SEED=0) and reproduced by the
new driver with identical selections, test macro-F1 and a 26-entry leaderboard.
Since score ties are broken by ascending pool index (stable sorts, docs/DESIGN.md),
auth_only (kNN agreement counts, heavily tied) has a new hash and the election moved from dmf
(03e90759cace) to vote_ensemble(top3). The other strategy hashes are the 2026-07 values. The hashes
depend on numpy and scikit-learn only. The election also depends on the MLP.
"""
from __future__ import annotations

import json

import pytest

from tracks.common.experiment import load_track, resolve_configs, run_cell

PINNED_SEL = {
    "full": "dbccfb4a53a1", "random": "467b014cdbb7", "coreset": "74a1403f98d1", "auth_only": "a7744654dba8",
    "influence_only": "3e07fb8e82ee", "mmdataselect": "ae2d4b57ec34", "herding": "6c1bf58930e2",
    "el2n": "bad54c94a497", "grand": "3b31e6472417", "ccs": "4207d9b5abba", "density": "1bbd05a7372b",
    "quadmix_pub": "308ef75012b5",
}
PINNED_ELECTED = "vote_ensemble(top3)"
PINNED_ELECTED_SEL = "1313e9427f57"


@pytest.mark.slow
def test_canonical_tep_fixture_matches_the_2026_07_runner(tmp_path):
    track = load_track("process")
    methods = tuple(PINNED_SEL) + ("dmf_pub", "mmds_adapt")
    tcfg, ocfg = resolve_configs(track, "tep21", learner="mlp", seed=0, protocol="canonical", smoke=False,
                                 track_overrides={"pool_n": 400, "val_n": 300, "test_n": 300, "methods": methods},
                                 omni_overrides={})
    run_cell(track, tcfg, ocfg, out_root=tmp_path, batch="r0fixture", cli=["test"])
    cell = tmp_path / "r0fixture" / "process" / "tep21" / "mlp" / "seed_0"
    rows = json.loads((cell / "metrics.json").read_text())["rows"]
    for name, sha in PINNED_SEL.items():
        assert rows[name]["sel_sha12"] == sha, name
    decision = json.loads((cell / "decision.json").read_text())
    assert decision["elected"] == PINNED_ELECTED
    assert decision["elected_sel_sha12"] == PINNED_ELECTED_SEL
    board = json.loads((cell / "leaderboard.json").read_text())["rows"]
    assert len([r for r in board if r["scoring_u_rank"] is not None]) == 26
