"""DMF baseline runner — dynamic multi-signal fusion -> shared manifest.

    python baselines/dmf/run_dmf.py --config configs/experiments/<exp>.yaml

Reads the processed UnifiedRecord jsonl, fuses a redundancy + influence signal with
the *base* dynamic multi-signal fusion (all advanced mechanisms off), selects a
budget-constrained subset, and emits ``<output-dir>/manifests/{manifest.json,
selected.jsonl}`` in the same contract format as ``run_mmdataselect/run_select.py``.

DMF is the multi-signal dynamic-fusion comparison our system aims to surpass; this
runner keeps it on the same budget and manifest contract so the two are directly
comparable. ``torch``/``transformers`` are only reached lazily inside the influence
signal and degrade to a CPU proxy when missing, so this runner has no hard ML deps.
"""
from __future__ import annotations

import argparse
import os
import sys

# Repo root is three levels up: baselines/dmf/run_dmf.py -> repo/.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local `method` pkg

import numpy as np  # noqa: E402

from mmdataselect.budget import Budget  # noqa: E402
from mmdataselect.datatypes import UnifiedRecord  # noqa: E402
from mmdataselect.utils.io import read_jsonl, read_yaml  # noqa: E402
from mmdataselect.utils.logger import get_logger  # noqa: E402
from mmdataselect.utils.manifest import write_manifest  # noqa: E402

from method import select  # noqa: E402

log = get_logger("run_dmf")


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="DMF dynamic multi-signal fusion -> manifest.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = read_yaml(resolve(args.config))
    exp_id = cfg.get("experiment_id", "exp")
    out_dir = resolve(args.output_dir) if args.output_dir else os.path.join(_REPO, "outputs", f"{exp_id}_dmf")

    processed = resolve(cfg["data"]["processed_path"])
    if not os.path.exists(processed):
        log.error("processed pool not found: %s (run tools/standardize first)", processed)
        return 2
    records = [UnifiedRecord.from_dict(d) for d in read_jsonl(processed)]
    n_total = len(records)
    log.info("loaded %d records from %s", n_total, processed)

    scfg = cfg.get("select", {})
    budget = Budget.from_cfg(scfg)
    token_counts = [len((r.text or "").split()) for r in records]
    k = budget.resolve(n_total, token_counts=token_counts)

    dcfg = cfg.get("dmf", {})
    seed = int(scfg.get("seed", 0))
    model_name = scfg.get("influence_model") or dcfg.get("influence_model")
    lr = float(dcfg.get("lr", 0.5))
    diversity = bool(dcfg.get("diversity", True))
    lam = float(dcfg.get("lam", scfg.get("lam", 0.5)))

    # Optional held-out probe for one dynamic-weight update. ``holdout_frac`` carves
    # a deterministic tail-fraction of the pool as the probe; 0 (default) skips it,
    # leaving pure base fusion.
    holdout_frac = float(dcfg.get("holdout_frac", 0.0))
    holdout_idx = None
    if 0.0 < holdout_frac < 1.0 and n_total > 1:
        h = max(1, min(n_total - 1, int(round(holdout_frac * n_total))))
        rng = np.random.default_rng(seed)
        holdout_idx = sorted(int(i) for i in rng.choice(n_total, size=h, replace=False))

    selected_idx = select(
        records,
        k,
        model_name=model_name,
        lr=lr,
        diversity=diversity,
        lam=lam,
        holdout_idx=holdout_idx,
        seed=seed,
    )

    selected_ids = [records[i].id for i in selected_idx]
    selected_rows = [records[i].to_dict() for i in selected_idx]
    diagnostics = {
        "n_total": n_total,
        "n_selected": len(selected_idx),
        "keep_ratio": round(len(selected_idx) / n_total, 4) if n_total else 0.0,
        "n_holdout_probe": len(holdout_idx) if holdout_idx else 0,
    }
    write_manifest(
        out_dir,
        experiment_id=exp_id,
        method="dmf",
        n_total=n_total,
        selected_ids=selected_ids,
        selected_rows=selected_rows,
        extra={
            "diagnostics": diagnostics,
            "dmf_config": {
                "signals": ["redundancy", "influence"],
                "lr": lr,
                "diversity": diversity,
                "lam": lam,
                "holdout_frac": holdout_frac,
                "influence_model": model_name,
                "seed": seed,
                "advanced_mechanisms": "off",  # base dynamic multi-signal fusion only
            },
            "select_config": scfg,
            "reference": "dynamic multi-signal data-selection fusion (base variant)",
        },
    )
    log.info("selected %d/%d -> %s", len(selected_idx), n_total, os.path.join(out_dir, "manifests"))
    print(f"DMF OK | {len(selected_idx)}/{n_total} | diagnostics={diagnostics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
