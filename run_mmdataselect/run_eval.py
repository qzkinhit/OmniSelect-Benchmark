"""Application-layer entry: evaluate the trained checkpoint on benchmarks.

Delegates to tools/eval, which wraps lm-evaluation-harness (language/math) and a
code sandbox. Without the ``eval`` extra this exits code 3 (skipped).

    python run_mmdataselect/run_eval.py --config configs/experiments/<exp>.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)  # so `tools` is importable

from mmdataselect.utils.io import ensure_dir, read_yaml, write_json  # noqa: E402
from mmdataselect.utils.logger import get_logger  # noqa: E402

log = get_logger("run_eval")


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on benchmarks.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = read_yaml(resolve(args.config))
    exp_id = cfg.get("experiment_id", "exp")
    out_dir = resolve(args.output_dir) if args.output_dir else os.path.join(_REPO, "outputs", exp_id)
    ckpt = os.path.join(out_dir, "models")
    tasks = cfg.get("eval", {}).get("tasks", ["arc_easy"])

    try:
        from tools.eval.harness import evaluate_model
    except Exception as e:
        log.error("tools.eval import failed: %s", e)
        return 2

    result = evaluate_model(model_path=ckpt, tasks=tasks, cfg=cfg.get("eval", {}))
    if result is None:
        print("EVAL SKIPPED | missing 'eval' extra (pip install -e \".[eval]\")")
        return 3

    eval_dir = ensure_dir(os.path.join(out_dir, "eval"))
    out_path = os.path.join(eval_dir, "results.json")
    write_json(result, out_path)
    log.info("wrote eval -> %s", out_path)
    print(f"EVAL OK | tasks={tasks} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
