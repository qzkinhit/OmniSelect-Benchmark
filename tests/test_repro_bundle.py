import json
from pathlib import Path

import numpy as np

from mmdataselect.utils.repro_bundle import (
    canonical_sha256,
    save_downstream_checkpoint,
    validate_repro_bundle,
    write_repro_bundle,
)


def test_bundle_persists_selected_data_predictions_and_hashes(tmp_path):
    result = tmp_path / "results.json"
    result.write_text('{"ok": true}\n')
    runner = tmp_path / "runner.py"
    runner.write_text("print('ok')\n")
    checkpoint = tmp_path / "checkpoints" / "random"
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights.bin").write_bytes(b"weights")
    bundle = write_repro_bundle(
        tmp_path,
        repo_root=Path.cwd(),
        runner_path=runner,
        arm="unit",
        dataset="fixture",
        seed=0,
        config={"passes": 2},
        result_path=result,
        selections={"random": [2, 0]},
        selection_source={
            "features": np.arange(12).reshape(3, 4),
            "labels": np.array([4, 5, 6]),
        },
        predictions={"random": {"y_true": [1, 0], "y_pred": [1, 1]}},
        checkpoint_paths={"random": checkpoint},
    )
    assert validate_repro_bundle(bundle) == []
    selected = np.load(bundle / "selected_data" / "random.npz")
    assert selected["labels"].tolist() == [6, 4]
    selection = json.loads((bundle / "selections" / "random.json").read_text())
    assert selection["ordered_sha256"] == canonical_sha256([2, 0])
    checkpoints = json.loads((bundle / "checkpoints.json").read_text())
    assert checkpoints["random"]["files"]["weights.bin"]["sha256"]


def test_bundle_validator_detects_tampering(tmp_path):
    result = tmp_path / "results.json"
    result.write_text("{}")
    runner = tmp_path / "runner.py"
    runner.write_text("pass\n")
    bundle = write_repro_bundle(
        tmp_path,
        repo_root=Path.cwd(),
        runner_path=runner,
        arm="unit",
        dataset="fixture",
        seed=1,
        config={},
        result_path=result,
        selections={"m": [0]},
        selection_source={"x": np.array([[1.0]])},
    )
    (bundle / "selections" / "m.json").write_text("{}")
    assert any("sha256 mismatch" in error for error in validate_repro_bundle(bundle))


def test_save_downstream_checkpoint_serializes_a_fitted_sklearn_model(tmp_path):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(random_state=0).fit(
        [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        [0, 1, 1, 0],
    )
    checkpoint = save_downstream_checkpoint(
        model,
        tmp_path / "checkpoint",
        metadata={"arm": "unit", "method": "random"},
    )
    descriptor = json.loads((checkpoint / "checkpoint.json").read_text())
    assert descriptor["serializer"] == "joblib"
    assert descriptor["metadata"]["method"] == "random"
    assert (checkpoint / "model.joblib").is_file()
