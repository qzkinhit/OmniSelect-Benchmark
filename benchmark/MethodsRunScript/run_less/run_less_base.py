"""Standalone LESS run: its selection, the reported fit and the run record, without the controller.

Forwards to tracks.common.experiment with ``--methods less --standalone``. Every other driver
flag (track, dataset, seed, protocol, batch, --set, --track-set, --smoke) passes through. The run
record lands in results_and_logs/<batch>/<track>/<dataset>/<learner>/seed_<s>/ with the same
layout as a controller cell (docs/RUN_RECORD.md), and decision.json records controller off.
"""
from __future__ import annotations

import sys

from tracks.common.experiment import main

METHOD = "less"


if __name__ == "__main__":
    raise SystemExit(main(["--methods", METHOD, "--standalone", *sys.argv[1:]]))
