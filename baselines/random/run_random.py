"""Application-layer runner for the random baseline.

    python baselines/random/run_random.py --config configs/experiments/<exp>.yaml

Reads the processed UnifiedRecord jsonl, resolves the budget with the shared
``Budget``, draws a uniform random subset, and emits the same manifest contract
(``manifests/{manifest.json,selected.jsonl}``) as ``run_mmdataselect/run_select.py``
so the baseline plugs into the shared ``run_train`` / ``run_eval``.
"""
from __future__ import annotations

import argparse
import os
import sys

# Repo root is two levels up from baselines/random/ ; expose the core package on sys.path.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmdataselect.budget import Budget  # noqa: E402
from mmdataselect.datatypes import UnifiedRecord  # noqa: E402
from mmdataselect.utils.io import read_jsonl, read_yaml  # noqa: E402
from mmdataselect.utils.manifest import write_manifest  # noqa: E402

from method import select  # noqa: E402


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Random baseline: uniform subset -> manifest.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = read_yaml(resolve(args.config))
    exp_id = cfg.get("experiment_id", "exp")
    out_dir = resolve(args.output_dir) if args.output_dir else os.path.join(_REPO, "outputs", f"{exp_id}_random")

    processed = resolve(cfg["data"]["processed_path"])
    if not os.path.exists(processed):
        print(f"processed pool not found: {processed} (run tools/standardize first)", file=sys.stderr)
        return 2
    records = [UnifiedRecord.from_dict(d) for d in read_jsonl(processed)]
    n_total = len(records)

    scfg = cfg.get("select", {})
    budget = Budget.from_cfg(scfg)
    token_counts = [len((r.text or "").split()) for r in records]
    k = budget.resolve(n_total, token_counts=token_counts)
    seed = int(scfg.get("seed", 0))

    ids = [r.id for r in records]
    selected_ids = select(ids, k, seed=seed)

    by_id = {r.id: r for r in records}
    selected_rows = [by_id[i].to_dict() for i in selected_ids]
    write_manifest(
        out_dir,
        experiment_id=exp_id,
        method="random",
        n_total=n_total,
        selected_ids=selected_ids,
        selected_rows=selected_rows,
        extra={"select_config": scfg, "seed": seed, "budget": {"kind": budget.kind, "value": budget.value}},
    )
    print(f"RANDOM OK | {len(selected_ids)}/{n_total} | seed={seed} -> {os.path.join(out_dir, 'manifests')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
