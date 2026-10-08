"""Downstream benchmark check of a text cell: refit the elected subset and run lm-eval tasks.

``python -m tracks.text.lm_eval <cell>`` reads config.json and candidates/<elected>/selection.npz,
fine-tunes SmolLM2 on that subset with the cell's schedule, and evaluates the lm-eval-harness tasks
(arc_easy, arc_challenge, hellaswag, openbookqa by default). The scores are written to
<cell>/lm_eval.json. The canonical text batch ran this check on the elected model only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from omniselect.config.config import OmniSelectConfig
from tracks.common.experiment import load_track


def main(argv: list[str] | None = None) -> int:
    """Refit the elected subset of a text cell and write lm_eval.json."""
    parser = argparse.ArgumentParser(description="lm-eval check of the elected text subset")
    parser.add_argument("cell", type=Path)
    parser.add_argument("--tasks", default="arc_easy,arc_challenge,hellaswag,openbookqa")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    config = json.loads((args.cell / "config.json").read_text())
    decision = json.loads((args.cell / "decision.json").read_text())
    board = json.loads((args.cell / "leaderboard.json").read_text())
    track = load_track("text")
    tcfg = track.config_cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in config["track"].items()})
    ocfg = OmniSelectConfig.from_dict(config["omniselect"])
    data = track.load(tcfg, ocfg)
    learner = track.learner(data, tcfg, ocfg)
    elected_dir = board["candidate_dirs"][decision["elected"]]
    with np.load(args.cell / "candidates" / elected_dir / "selection.npz") as z:
        idx = z["idx"]
    model = learner.fit(idx, "report")
    lm = HFLM(pretrained=model, tokenizer=data.arrays["lm"].tok, batch_size=args.batch_size)
    out = lm_eval.simple_evaluate(model=lm, tasks=args.tasks.split(","), limit=args.limit or None)
    scores = {task: out["results"][task].get("acc_norm,none", out["results"][task].get("acc,none"))
              for task in args.tasks.split(",")}
    (args.cell / "lm_eval.json").write_text(json.dumps({"elected": decision["elected"], "scores": scores}, indent=2))
    print(scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
