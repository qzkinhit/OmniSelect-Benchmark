import hashlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mmdataselect.fusion.methodv3_text_terminal_adapter import (
    freeze_text_pair_manifest,
    run_methodv3_text_terminal_adapter,
)
from mmdataselect.fusion.paired_text_logloss_gate import (
    freeze_text_pair,
    run_paired_text_logloss_gate,
)
from scripts.run_experiment import _state_sha256, split_methodv3_text_controller_records


def _sha(label):
    return hashlib.sha256(label.encode()).hexdigest()


def test_state_sha256_supports_bfloat16_without_changing_weights():
    model = torch.nn.Linear(3, 2).to(dtype=torch.bfloat16)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    digest_a = _state_sha256(model)
    digest_b = _state_sha256(model)
    assert len(digest_a) == 64 and digest_a == digest_b
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def _manifest():
    return freeze_text_pair_manifest(
        reference_arm="zip",
        challenger_arm="mmds_v3",
        seed=0,
        pool_sha256=_sha("pool"),
        v1_split_sha256=_sha("v1"),
        reference_selector_sha256=_sha("zip-code"),
        challenger_selector_sha256=_sha("mmds-code"),
        reference_selection_sha256=_sha("zip-selection"),
        challenger_selection_sha256=_sha("mmds-selection"),
        reference_v1_evidence_sha256=_sha("zip-v1"),
        challenger_v1_evidence_sha256=_sha("mmds-v1"),
        effective_token_cap=4096,
    )


def test_large_record_level_improvement_switches():
    pair = freeze_text_pair("zip", "mmds_v3", _sha("pair"))
    ref = [2.0] * 100
    chal = [0.0] * 100
    result = run_paired_text_logloss_gate(
        pair, ref, chal, [10] * 100, ["a"] * 50 + ["b"] * 50
    )
    assert result.decision == "SWITCH_CERTIFIED"
    assert result.lcb > 0


def test_small_improvement_keeps_reference():
    pair = freeze_text_pair("zip", "mmds_v3", _sha("pair"))
    result = run_paired_text_logloss_gate(
        pair,
        [1.01] * 20,
        [1.0] * 20,
        [10] * 20,
        ["a"] * 10 + ["b"] * 10,
    )
    assert result.decision == "KEEP_REFERENCE"
    assert result.selected_arm == "zip"


def test_bad_inputs_abstain_without_switch():
    pair = freeze_text_pair("zip", "mmds_v3", _sha("pair"))
    result = run_paired_text_logloss_gate(
        pair, [1.0, float("nan")], [0.0, 0.0], [10, 10], ["a", "a"]
    )
    assert result.decision == "ABSTAIN_UNCERTIFIED"
    assert not result.switched


def test_domain_balancing_not_record_count_balancing():
    pair = freeze_text_pair("zip", "mmds_v3", _sha("pair"))
    # Domain a has 90 negative records; domain b has 10 positive records.  Equal
    # record weights would be negative, while domain balance gives zero.
    result = run_paired_text_logloss_gate(
        pair,
        [0.0] * 90 + [2.0] * 10,
        [1.0] * 90 + [0.0] * 10,
        [1] * 100,
        ["a"] * 90 + ["b"] * 10,
    )
    assert abs(result.estimate) < 1e-12
    assert result.decision == "KEEP_REFERENCE"


def test_terminal_binds_ordered_ids_and_rejects_tampered_pair():
    manifest = _manifest()
    ids = [f"id-{i}" for i in range(100)]
    record = run_methodv3_text_terminal_adapter(
        manifest,
        ordered_record_ids=ids,
        domains=["a"] * 50 + ["b"] * 50,
        token_counts=[10] * 100,
        reference_mean_nll=[2.0] * 100,
        challenger_mean_nll=[0.0] * 100,
    )
    assert record["decision"] == "SWITCH_CERTIFIED"
    assert len(record["decision_record_sha256"]) == 64

    bad = dict(manifest)
    bad["reference_arm"] = "random"
    rejected = run_methodv3_text_terminal_adapter(
        bad,
        ordered_record_ids=ids,
        domains=["a"] * 50 + ["b"] * 50,
        token_counts=[10] * 100,
        reference_mean_nll=[2.0] * 100,
        challenger_mean_nll=[0.0] * 100,
    )
    assert rejected["decision"] == "ABSTAIN_UNCERTIFIED"
    assert rejected["selected_arm"] == "random"


def test_duplicate_v2_ids_fail_closed():
    record = run_methodv3_text_terminal_adapter(
        _manifest(),
        ordered_record_ids=["same", "same"],
        domains=["a", "a"],
        token_counts=[10, 10],
        reference_mean_nll=[2.0, 2.0],
        challenger_mean_nll=[0.0, 0.0],
    )
    assert record["decision"] == "ABSTAIN_UNCERTIFIED"


def test_zero_token_counts_and_non_mapping_manifest_fail_closed():
    record = run_methodv3_text_terminal_adapter(
        _manifest(),
        ordered_record_ids=["a", "b"],
        domains=["a", "a"],
        token_counts=[0, 0],
        reference_mean_nll=[2.0, 2.0],
        challenger_mean_nll=[0.0, 0.0],
    )
    assert record["decision"] == "ABSTAIN_UNCERTIFIED"
    malformed = run_methodv3_text_terminal_adapter(
        None,
        ordered_record_ids=["a"],
        domains=["a"],
        token_counts=[1],
        reference_mean_nll=[1.0],
        challenger_mean_nll=[0.0],
    )
    assert malformed["decision"] == "ABSTAIN_UNCERTIFIED"


def test_id_only_v1_v2_split_is_stable_and_disjoint():
    records = [SimpleNamespace(id=f"row-{i}", text=f"secret-{i}") for i in range(20)]
    v1_a, v2_a = split_methodv3_text_controller_records(records, 7)
    # Changing text/outcomes cannot affect the split; only seed and record ID do.
    changed = [SimpleNamespace(id=r.id, text="different") for r in reversed(records)]
    v1_b, v2_b = split_methodv3_text_controller_records(changed, 7)
    assert [r.id for r in v1_a] == [r.id for r in v1_b]
    assert [r.id for r in v2_a] == [r.id for r in v2_b]
    assert {r.id for r in v1_a}.isdisjoint({r.id for r in v2_a})
    assert {r.id for r in v1_a} | {r.id for r in v2_a} == {
        r.id for r in records
    }
