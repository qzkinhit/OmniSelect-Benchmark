"""IF / MATES baseline runner — influence Top-K -> shared manifest.

    python baselines/if_mates/run_if_mates.py --config configs/experiments/<exp>.yaml

Reads the processed UnifiedRecord jsonl, scores each record by its downstream-model
per-sample influence (via ``mmdataselect.signals.InfluenceSignal``, with a CPU
fallback when torch is unavailable), keeps the Top-K under the shared budget, and
emits ``<output-dir>/manifests/{manifest.json,selected.jsonl}`` in the same contract
format as ``run_mmdataselect/run_select.py`` so the baseline plugs into the shared
``run_train`` / ``run_eval``.

References: MATES (Yu et al., NeurIPS 2024; cxcscmu/MATES); Koh & Liang influence
functions (ICML 2017).
"""
from __future__ import annotations

import argparse
import os
import sys

# Repo root is three levels up: baselines/if_mates/run_if_mates.py -> repo/.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local `method` pkg

from mmdataselect.budget import Budget  # noqa: E402
from mmdataselect.datatypes import UnifiedRecord  # noqa: E402
from mmdataselect.signals import InfluenceSignal  # noqa: E402
from mmdataselect.utils.io import read_jsonl, read_yaml  # noqa: E402
from mmdataselect.utils.logger import get_logger  # noqa: E402
from mmdataselect.utils.manifest import write_manifest  # noqa: E402

from method import select  # noqa: E402

log = get_logger("run_if_mates")


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="IF / MATES influence Top-K -> manifest.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = read_yaml(resolve(args.config))
    exp_id = cfg.get("experiment_id", "exp")
    out_dir = resolve(args.output_dir) if args.output_dir else os.path.join(_REPO, "outputs", f"{exp_id}_if_mates")

    processed = resolve(cfg["data"]["processed_path"])
    if not os.path.exists(processed):
        log.error("processed pool not found: %s (run tools/standardize first)", processed)
        return 2
    records = [UnifiedRecord.from_dict(d) for d in read_jsonl(processed)]
    log.info("loaded %d records from %s", len(records), processed)

    scfg = cfg.get("select", {})
    budget = Budget.from_cfg(scfg)
    token_counts = [len((r.text or "").split()) for r in records]
    k = budget.resolve(len(records), token_counts=token_counts)
    seed = int(scfg.get("seed", 0))

    # Score every record once with the influence signal so we can report both the
    # full influence distribution and the selected-subset mean as diagnostics; the
    # signal handles its own torch path / CPU fallback internally.
    influence_model = scfg.get("influence_model")
    signal = InfluenceSignal(influence_model)
    influence = signal.score(records)

    selected_idx = select(records, k, influence=influence, seed=seed)

    selected_ids = [records[i].id for i in selected_idx]
    selected_rows = [records[i].to_dict() for i in selected_idx]
    sel_scores = [float(influence[i]) for i in selected_idx]
    diagnostics = {
        "n_total": len(records),
        "n_selected": len(selected_idx),
        "keep_ratio": round(len(selected_idx) / len(records), 4) if records else 0.0,
        "mean_influence_pool": round(float(sum(influence) / len(influence)), 4) if len(influence) else 0.0,
        "mean_influence_selected": round(sum(sel_scores) / len(sel_scores), 4) if sel_scores else 0.0,
    }
    write_manifest(
        out_dir,
        experiment_id=exp_id,
        method="if_mates",
        n_total=len(records),
        selected_ids=selected_ids,
        selected_rows=selected_rows,
        extra={
            "diagnostics": diagnostics,
            "if_mates_config": {
                "influence_model": influence_model,
                "seed": seed,
            },
            "reference": "MATES (Yu et al., NeurIPS 2024); Koh & Liang influence functions (ICML 2017)",
        },
    )
    log.info("selected %d/%d -> %s", len(selected_idx), len(records), os.path.join(out_dir, "manifests"))
    print(f"IF/MATES OK | {len(selected_idx)}/{len(records)} | diagnostics={diagnostics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
