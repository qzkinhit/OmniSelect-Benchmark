"""DSIR baseline runner — importance resampling -> shared manifest.

    python baselines/dsir/run_dsir.py --config configs/experiments/<exp>.yaml

Reads the processed UnifiedRecord jsonl, runs DSIR importance resampling toward the
math+code target distribution, and emits ``<output-dir>/manifests/{manifest.json,
selected.jsonl}`` in the same contract format as ``run_mmdataselect/run_select.py``.

Reference: Xie et al., "Data Selection for Language Models via Importance
Resampling (DSIR)", NeurIPS 2023.
"""
from __future__ import annotations

import argparse
import os
import sys

# Repo root is three levels up: baselines/dsir/run_dsir.py -> repo/.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local `method` pkg

from mmdataselect.budget import Budget  # noqa: E402
from mmdataselect.datatypes import UnifiedRecord  # noqa: E402
from mmdataselect.utils.io import read_jsonl, read_yaml  # noqa: E402
from mmdataselect.utils.logger import get_logger  # noqa: E402
from mmdataselect.utils.manifest import write_manifest  # noqa: E402

from method import dsir_select  # noqa: E402

log = get_logger("run_dsir")


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="DSIR importance resampling -> manifest.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = read_yaml(resolve(args.config))
    exp_id = cfg.get("experiment_id", "exp")
    out_dir = resolve(args.output_dir) if args.output_dir else os.path.join(_REPO, "outputs", f"{exp_id}_dsir")

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

    dcfg = cfg.get("dsir", {})
    seed = int(scfg.get("seed", 0))
    texts = [r.text for r in records]
    ids = [r.id for r in records]
    domains = [r.domain for r in records]
    selected_idx, log_w = dsir_select(
        texts,
        ids,
        k,
        domains=domains,
        dim=int(dcfg.get("dim", 4096)),
        ngram=int(dcfg.get("ngram", 2)),
        smoothing=float(dcfg.get("smoothing", 1.0)),
        noise=float(dcfg.get("noise", 1.0)),
        seed=seed,
    )

    n_target = int(sum(1 for d in domains if str(d).lower() in ("math", "code")))
    selected_ids = [records[i].id for i in selected_idx]
    selected_rows = [records[i].to_dict() for i in selected_idx]
    diagnostics = {
        "n_total": len(records),
        "n_selected": len(selected_idx),
        "keep_ratio": round(len(selected_idx) / len(records), 4) if records else 0.0,
        "n_target_domain": n_target,
        "mean_log_weight_pool": round(float(log_w.mean()), 4) if log_w.size else 0.0,
        "mean_log_weight_selected": round(
            float(log_w[selected_idx].mean()), 4
        ) if selected_idx else 0.0,
    }
    write_manifest(
        out_dir,
        experiment_id=exp_id,
        method="dsir",
        n_total=len(records),
        selected_ids=selected_ids,
        selected_rows=selected_rows,
        extra={
            "diagnostics": diagnostics,
            "dsir_config": {
                "dim": int(dcfg.get("dim", 4096)),
                "ngram": int(dcfg.get("ngram", 2)),
                "smoothing": float(dcfg.get("smoothing", 1.0)),
                "noise": float(dcfg.get("noise", 1.0)),
                "seed": seed,
                "target_domains": ["math", "code"],
            },
            "reference": "Xie et al., DSIR, NeurIPS 2023",
        },
    )
    log.info("selected %d/%d -> %s", len(selected_idx), len(records), os.path.join(out_dir, "manifests"))
    print(f"DSIR OK | {len(selected_idx)}/{len(records)} | diagnostics={diagnostics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
