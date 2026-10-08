import json

import pytest

from omniselect.utils.io import write_json, write_jsonl


def test_json_writers_reject_nonfinite_values_without_publishing(tmp_path):
    json_path = tmp_path / "value.json"
    with pytest.raises(ValueError):
        write_json({"metric": float("nan")}, str(json_path))
    assert not json_path.exists()

    jsonl_path = tmp_path / "value.jsonl"
    with pytest.raises(ValueError):
        write_jsonl([{"metric": float("inf")}], str(jsonl_path))
    assert not jsonl_path.exists()


def test_json_writers_publish_complete_parseable_files(tmp_path):
    json_path = tmp_path / "value.json"
    assert write_json({"value": 7}, str(json_path)) == str(json_path)
    assert json.loads(json_path.read_text()) == {"value": 7}

    jsonl_path = tmp_path / "value.jsonl"
    assert write_jsonl([{"row": 1}, {"row": 2}], str(jsonl_path)) == 2
    assert [
        json.loads(line) for line in jsonl_path.read_text().splitlines()
    ] == [{"row": 1}, {"row": 2}]


def test_failed_jsonl_iteration_preserves_existing_destination(tmp_path):
    destination = tmp_path / "rows.jsonl"
    destination.write_text('{"original": true}\n')

    def interrupted_rows():
        yield {"row": 1}
        raise RuntimeError("simulated interrupted producer")

    with pytest.raises(RuntimeError, match="interrupted producer"):
        write_jsonl(interrupted_rows(), str(destination))

    assert destination.read_text() == '{"original": true}\n'
    assert list(tmp_path.glob("*.tmp")) == []
