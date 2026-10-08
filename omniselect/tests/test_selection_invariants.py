from pathlib import Path

import numpy as np
import pytest

from omniselect.tools.pairing import (
    exact_kmeans_representatives,
    validate_selection,
)


def test_kmeans_representatives_fill_empty_clusters_to_exact_budget():
    features = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.2, 5.0],
            [9.0, 9.0],
        ]
    )
    # Center 1 is intentionally empty, reproducing the MiniBatchKMeans edge
    # case that previously returned only two representatives for budget three.
    labels = np.array([0, 0, 2, 2, 2])
    centers = np.array([[0.0, 0.0], [2.5, 2.5], [5.0, 5.0]])
    selected = exact_kmeans_representatives(
        features, labels, centers, expected_size=3
    )

    assert selected == [0, 2, 1]
    assert len(selected) == len(set(selected)) == 3


def test_kmeans_representatives_reject_mismatched_or_nonfinite_state():
    features = np.array([[0.0], [1.0], [2.0]])
    labels = np.array([0, 0, 1])
    with pytest.raises(ValueError, match="center count"):
        exact_kmeans_representatives(
            features,
            labels,
            np.array([[0.0], [2.0]]),
            expected_size=3,
        )
    with pytest.raises(ValueError, match="finite"):
        exact_kmeans_representatives(
            np.array([[0.0], [np.nan], [2.0]]),
            labels,
            np.array([[0.0], [1.0], [2.0]]),
            expected_size=3,
        )


def test_all_kmeans_coverage_paths_share_exact_completion():
    root = Path(__file__).resolve().parents[2]
    paths = [
        "benchmark/Methods/Coverage/method.py",
        "tracks/vision/native_resnet.py",
        "omniselect/core/adjudication/controller.py",
    ]
    for relative in paths:
        source = (root / relative).read_text()
        assert "exact_kmeans_representatives(" in source, relative


def test_validate_selection_accepts_integer_ids_and_preserves_order():
    selected = validate_selection(
        np.array([3, 1, 4], dtype=np.int64),
        pool_size=5,
        expected_size=3,
        method="example",
    )
    assert selected == [3, 1, 4]


@pytest.mark.parametrize(
    ("selected", "message"),
    [
        ([0, 0], "duplicate selection ID"),
        ([-1, 1], "outside"),
        ([0, 3], "outside"),
        ([0], "expected exactly 2"),
        ([0.0, 1.0], "not an integer"),
        ([False, 1], "is boolean"),
    ],
)
def test_validate_selection_rejects_invalid_outputs(selected, message):
    with pytest.raises((TypeError, ValueError), match=message):
        validate_selection(
            selected,
            pool_size=3,
            expected_size=2,
            method="broken-selector",
        )
