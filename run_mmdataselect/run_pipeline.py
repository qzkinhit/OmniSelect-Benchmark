"""One-shot pipeline: select -> train -> eval, sharing one output dir + manifest.

    python run_mmdataselect/run_pipeline.py --config configs/experiments/<exp>.yaml

Each stage is a separate application-layer entry; train/eval exit code 3 = "skipped"
(missing optional extra) so the pipeline still demonstrates select-only CPU runs.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


def _run(stage: str, config: str, output_dir: str) -> int:
    cmd = [sys.executable, os.path.join(_HERE, f"run_{stage}.py"), "--config", config, "--output-dir", output_dir]
    print(f"\n=== run_{stage} ===")
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description="select -> train -> eval pipeline.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--stages", default="select,train,eval")
    args = ap.parse_args()

    config = args.config if os.path.isabs(args.config) else os.path.join(_REPO, args.config)
    import yaml

    with open(config, "r", encoding="utf-8") as f:
        exp_id = (yaml.safe_load(f) or {}).get("experiment_id", "exp")
    out_dir = args.output_dir or os.path.join(_REPO, "outputs", exp_id)

    for stage in [s.strip() for s in args.stages.split(",") if s.strip()]:
        rc = _run(stage, config, out_dir)
        if rc == 3:
            print(f"[pipeline] {stage} skipped (missing optional extra); continuing.")
            continue
        if rc != 0:
            print(f"[pipeline] {stage} failed with code {rc}; stopping.")
            return rc
    print("\nPIPELINE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
