import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results_canonical/text/fixed_primary_20260724/seed_0/results.json"
)


def _canonical_sha(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def test_registered_fixed_primary_text_result_is_intact():
    payload = json.loads(RESULT.read_text())
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "2d18b22619f9bee399735b814e841482e9cad7cc8481d894cfa61493ff001ab5"
    )
    without_sha = dict(payload)
    recorded = without_sha.pop("artifact_sha256")
    assert _canonical_sha(without_sha) == recorded

    candidates = {
        row["method"]: row["report_gmean_ppl"]
        for row in payload["candidate_v1_evidence"]
    }
    assert candidates["random"] == 11.073388180049143
    assert candidates["herding_text"] == 11.044150249228661
    assert candidates["quadmix_pub"] == 11.143291732226436
    assert payload["results"][0]["picked"] == "herding_text"
    assert payload["results"][0]["gmean_ppl"] == candidates["herding_text"]
