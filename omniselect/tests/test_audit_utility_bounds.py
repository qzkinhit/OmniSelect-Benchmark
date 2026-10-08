"""High MASE and perplexity are valid finite observations, not impossible metrics."""
import json

import pytest

from tools.audit_records import audit_cell, utility_severity


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -float("inf"), 0.01])
def test_negative_mase_nonfinite_or_positive_is_error(value):
    assert utility_severity("neg_mase", value) == "ERROR"


@pytest.mark.parametrize("value", [0.0, -1.0, -5.0])
def test_negative_mase_legal_values_through_heuristic_boundary(value):
    assert utility_severity("neg_mase", value) is None


@pytest.mark.parametrize("value", [-5.000001, -5.472850588464919, -1e12])
def test_negative_mase_below_heuristic_boundary_is_warning(value):
    assert utility_severity("neg_mase", value) == "WARN"


def test_perplexity_has_no_finite_upper_bound_and_accuracy_is_bounded():
    assert utility_severity("neg_gmean_ppl", -200.0) is None
    assert utility_severity("neg_gmean_ppl", -200.01) == "WARN"
    assert utility_severity("neg_gmean_ppl", -0.99) == "ERROR"
    assert utility_severity("accuracy", 1.01) == "ERROR"
    assert utility_severity("accuracy", -0.01) == "ERROR"


@pytest.mark.parametrize("value,expected", [(-5.5, "WARN"), (0.01, "ERROR"), (None, "ERROR")])
def test_cell_audit_preserves_high_error_alert_without_rejecting_record(tmp_path, value, expected):
    # A minimum leaderboard fixture isolates the utility checks from the separate file-integrity checks.
    (tmp_path / "decision.json").write_text(json.dumps({"elected": "candidate"}))
    (tmp_path / "leaderboard.json").write_text(json.dumps(
        {"rows": [{"name": "candidate", "u_test": value, "role": "reference"}]}))
    (tmp_path / "splits.json").write_text(json.dumps({"utility": "neg_mase", "ids": {}}))
    issues = []
    audit_cell(str(tmp_path), issues, set())
    assert issues and {issue[0] for issue in issues} == {expected}
    if expected == "WARN":
        assert "high error" in issues[0][2] and "not a mathematical bound" in issues[0][2]
