"""ZIP / Entropy-Law baseline runner — redundancy-minimizing select -> manifest.

    python baselines/zip/run_zip.py --config configs/experiments/<exp>.yaml

Reads the processed UnifiedRecord jsonl, runs the three-stage ZIP greedy that keeps
the set compression ratio ``g(D) = Bits(D)/Bits(C(D))`` as low as possible (least
redundant / most informative subset), and emits ``<output-dir>/manifests/{manifest
.json,selected.jsonl}`` in the same contract format as ``run_mmdataselect/run_select
.py``.

Reference: Yin et al., "Entropy Law: The Story Behind Data Compression and LLM
Performance" (ZIP), USTC-StarTeam/ZIP.
"""
from __future__ import annotations

import argparse
import os
import sys

# Repo root is three levels up: baselines/zip/run_zip.py -> repo/.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local `method` pkg

from mmdataselect.budget import Budget  # noqa: E402
from mmdataselect.datatypes import UnifiedRecord  # noqa: E402
from mmdataselect.utils.io import read_jsonl, read_yaml  # noqa: E402
from mmdataselect.utils.logger import get_logger  # noqa: E402
from mmdataselect.utils.manifest import write_manifest  # noqa: E402

from method import compression_ratio, zip_select  # noqa: E402

log = get_logger("run_zip")


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="ZIP / Entropy-Law redundancy selection -> manifest.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = read_yaml(resolve(args.config))
    exp_id = cfg.get("experiment_id", "exp")
    out_dir = resolve(args.output_dir) if args.output_dir else os.path.join(_REPO, "outputs", f"{exp_id}_zip")

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

    zcfg = cfg.get("zip", {})
    seed = int(scfg.get("seed", 0))
    texts = [r.text for r in records]
    ids = [r.id for r in records]
    k1_ratio = float(zcfg.get("k1_ratio", 0.1))
    k2_ratio = float(zcfg.get("k2_ratio", 0.5))
    level = int(zcfg.get("level", 6))
    selected_idx, ratio_trace = zip_select(
        texts,
        ids,
        k,
        k1_ratio=k1_ratio,
        k2_ratio=k2_ratio,
        level=level,
        seed=seed,
    )

    selected_blob = "\n".join(records[i].text or "" for i in selected_idx).encode("utf-8")
    pool_blob = "\n".join(texts).encode("utf-8") if texts else b""
    selected_ids = [records[i].id for i in selected_idx]
    selected_rows = [records[i].to_dict() for i in selected_idx]
    diagnostics = {
        "n_total": len(records),
        "n_selected": len(selected_idx),
        "keep_ratio": round(len(selected_idx) / len(records), 4) if records else 0.0,
        "g_selected": round(compression_ratio(selected_blob, level=level), 4),
        "g_pool": round(compression_ratio(pool_blob, level=level), 4),
        "final_ratio_trace": round(ratio_trace[-1], 4) if ratio_trace else 0.0,
    }
    write_manifest(
        out_dir,
        experiment_id=exp_id,
        method="zip",
        n_total=len(records),
        selected_ids=selected_ids,
        selected_rows=selected_rows,
        extra={
            "diagnostics": diagnostics,
            "zip_config": {
                "k1_ratio": k1_ratio,
                "k2_ratio": k2_ratio,
                "level": level,
                "seed": seed,
            },
            "reference": "Yin et al., Entropy Law / ZIP, USTC-StarTeam/ZIP",
        },
    )
    log.info("selected %d/%d -> %s", len(selected_idx), len(records), os.path.join(out_dir, "manifests"))
    print(f"ZIP OK | {len(selected_idx)}/{len(records)} | diagnostics={diagnostics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
