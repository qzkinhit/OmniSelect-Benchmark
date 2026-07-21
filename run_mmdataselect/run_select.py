"""Application-layer entry: select a budget-constrained subset and write a manifest.

    python run_mmdataselect/run_select.py --config configs/experiments/<exp>.yaml

Reads the processed UnifiedRecord jsonl, runs the system's ``select_pool``, and emits
``<output-dir>/manifests/{manifest.json,selected.jsonl}`` in the shared contract format.
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from mmdataselect.api import select_pool  # noqa: E402
from mmdataselect.budget import Budget  # noqa: E402
from mmdataselect.datatypes import UnifiedRecord  # noqa: E402
from mmdataselect.utils.io import read_jsonl, read_yaml  # noqa: E402
from mmdataselect.utils.logger import get_logger  # noqa: E402
from mmdataselect.utils.manifest import write_manifest  # noqa: E402

log = get_logger("run_select")


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Budget-constrained data selection -> manifest.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = read_yaml(resolve(args.config))
    exp_id = cfg.get("experiment_id", "exp")
    out_dir = resolve(args.output_dir) if args.output_dir else os.path.join(_REPO, "outputs", exp_id)

    processed = resolve(cfg["data"]["processed_path"])
    if not os.path.exists(processed):
        log.error("processed pool not found: %s (run tools/standardize first)", processed)
        return 2
    records = [UnifiedRecord.from_dict(d) for d in read_jsonl(processed)]
    log.info("loaded %d records from %s", len(records), processed)

    scfg = cfg.get("select", {})
    budget = Budget.from_cfg(scfg)
    res = select_pool(
        records,
        budget,
        model_name=scfg.get("influence_model"),
        lam=float(scfg.get("lam", 0.5)),
        method=scfg.get("method", "greedy"),
        seed=int(scfg.get("seed", 0)),
    )

    selected_rows = [records[i].to_dict() for i in res.selected_idx]
    write_manifest(
        out_dir,
        experiment_id=exp_id,
        method="mmdataselect",
        n_total=len(records),
        selected_ids=res.selected_ids,
        selected_rows=selected_rows,
        extra={"diagnostics": res.diagnostics, "actor_weights": res.weights, "select_config": scfg},
    )
    log.info("selected %d/%d -> %s", res.diagnostics["n_selected"], len(records), os.path.join(out_dir, "manifests"))
    print(f"SELECT OK | {res.diagnostics['n_selected']}/{len(records)} | diagnostics={res.diagnostics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
