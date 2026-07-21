"""jsonl round-trip + manifest contract (manifest.json + selected.jsonl)."""
from __future__ import annotations

import os

from mmdataselect.utils.io import read_jsonl, write_jsonl
from mmdataselect.utils.manifest import (
    MANIFEST_NAME,
    SELECTED_NAME,
    read_manifest,
    read_selected,
    write_manifest,
)


def test_jsonl_round_trip(tmp_path):
    rows = [
        {"id": "a", "text": "first", "meta": {"n": 1}},
        {"id": "b", "text": "secondé", "meta": {"n": 2}},  # non-ascii preserved
    ]
    path = os.path.join(str(tmp_path), "sub", "rows.jsonl")
    n = write_jsonl(rows, path)
    assert n == 2
    assert os.path.exists(path)
    back = list(read_jsonl(path))
    assert back == rows


def test_jsonl_skips_blank_lines(tmp_path):
    path = os.path.join(str(tmp_path), "rows.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"id": "a"}\n')
        f.write("\n")  # blank line must be ignored
        f.write("   \n")
        f.write('{"id": "b"}\n')
    back = list(read_jsonl(path))
    assert back == [{"id": "a"}, {"id": "b"}]


def test_write_manifest_emits_both_files_with_full_fields(tmp_path):
    out_dir = str(tmp_path)
    selected_ids = ["r0", "r3", "r4"]
    selected_rows = [
        {"id": "r0", "modality": "text", "domain": "general", "text": "a", "meta": {}},
        {"id": "r3", "modality": "image_text", "domain": "math", "text": "b", "meta": {}},
        {"id": "r4", "modality": "text", "domain": "code", "text": "c", "meta": {}},
    ]
    manifest = write_manifest(
        out_dir,
        experiment_id="exp42",
        method="unit-test",
        n_total=6,
        selected_ids=selected_ids,
        selected_rows=selected_rows,
        extra={"diagnostics": {"keep_ratio": 0.5}, "note": "hi"},
    )

    # Returned manifest carries the full required contract.
    for key in ("experiment_id", "method", "n_total", "n_selected", "selected_ids"):
        assert key in manifest
    assert manifest["experiment_id"] == "exp42"
    assert manifest["method"] == "unit-test"
    assert manifest["n_total"] == 6
    assert manifest["n_selected"] == 3
    assert manifest["selected_ids"] == selected_ids
    # extra fields are merged in verbatim.
    assert manifest["diagnostics"] == {"keep_ratio": 0.5}
    assert manifest["note"] == "hi"

    # Both files land under <out_dir>/manifests/.
    mdir = os.path.join(out_dir, "manifests")
    assert os.path.exists(os.path.join(mdir, MANIFEST_NAME))
    assert os.path.exists(os.path.join(mdir, SELECTED_NAME))

    # Round-trip via the contract readers.
    on_disk = read_manifest(out_dir)
    assert on_disk == manifest
    assert read_selected(out_dir) == selected_rows


def test_write_manifest_without_rows_skips_selected_file(tmp_path):
    out_dir = str(tmp_path)
    write_manifest(
        out_dir,
        experiment_id="exp0",
        method="m",
        n_total=4,
        selected_ids=["a", "b"],
    )
    mdir = os.path.join(out_dir, "manifests")
    assert os.path.exists(os.path.join(mdir, MANIFEST_NAME))
    # selected.jsonl is optional and omitted when no rows are supplied.
    assert not os.path.exists(os.path.join(mdir, SELECTED_NAME))


def test_write_manifest_counts_from_iterator(tmp_path):
    # selected_ids may be any sequence; n_selected is derived from its length.
    ids = [f"id{i}" for i in range(7)]
    manifest = write_manifest(
        str(tmp_path),
        experiment_id="e",
        method="m",
        n_total=10,
        selected_ids=ids,
    )
    assert manifest["n_selected"] == 7
    assert manifest["selected_ids"] == ids
