"""full_data baseline runner — keep the whole pool, emit the shared manifest.

    python baselines/full_data/run_full_data.py --config configs/experiments/<exp>.yaml

Reads the processed UnifiedRecord jsonl named by ``cfg["data"]["processed_path"]``,
keeps every record (upper-bound reference), and writes
``outputs/<exp_id>_full_data/manifests/{manifest.json,selected.jsonl}`` in the same
contract format as ``run_mmdataselect/run_select.py`` so it plugs into run_train / run_eval.
"""
from __future__ import annotations

import argparse
import os
import sys

# Path resolution mirrors run_mmdataselect/run_select.py: repo root is three levels up
# (baselines/full_data/run_full_data.py -> repo), and src is put on sys.path for the
# core utils. The baseline's own select() stays import-light (no torch / no core deps).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmdataselect.budget import Budget  # noqa: E402
from mmdataselect.datatypes import UnifiedRecord  # noqa: E402
from mmdataselect.utils.io import read_jsonl, read_yaml  # noqa: E402
from mmdataselect.utils.logger import get_logger  # noqa: E402
from mmdataselect.utils.manifest import write_manifest  # noqa: E402

from method import select  # noqa: E402

log = get_logger("run_full_data")

METHOD = "full_data"


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="full_data baseline: keep all records -> manifest.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = read_yaml(resolve(args.config))
    exp_id = cfg.get("experiment_id", "exp")
    out_dir = resolve(args.output_dir) if args.output_dir else os.path.join(_REPO, "outputs", f"{exp_id}_{METHOD}")

    processed = resolve(cfg["data"]["processed_path"])
    if not os.path.exists(processed):
        log.error("processed pool not found: %s (run tools/standardize first)", processed)
        return 2
    records = [UnifiedRecord.from_dict(d) for d in read_jsonl(processed)]
    n_total = len(records)
    log.info("loaded %d records from %s", n_total, processed)

    # The budget is read for symmetry/logging only; full_data deliberately ignores it
    # and keeps the entire pool as the upper-bound reference.
    budget = Budget.from_cfg(cfg.get("select", {}))

    ids = [r.id for r in records]
    selected_ids = select(ids)
    selected_rows = [r.to_dict() for r in records]

    assert len(selected_ids) == n_total, "full_data must keep every record"

    write_manifest(
        out_dir,
        experiment_id=exp_id,
        method=METHOD,
        n_total=n_total,
        selected_ids=selected_ids,
        selected_rows=selected_rows,
        extra={"role": "upper_bound", "budget": {"kind": budget.kind, "value": budget.value}},
    )
    log.info("kept %d/%d (full data) -> %s", len(selected_ids), n_total, os.path.join(out_dir, "manifests"))
    print(f"FULL_DATA OK | {len(selected_ids)}/{n_total} | {os.path.join(out_dir, 'manifests')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
