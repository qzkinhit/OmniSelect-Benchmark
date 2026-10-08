# Local run records

The driver writes every cell here by default (`--out results_and_logs`). Everything in this
directory except this README is ignored by git.

- A cell writes `results_and_logs/<batch>/<track>/<dataset>/<learner>/seed_<s>/` with the files of
  [the run record](../docs/RUN_RECORD.md) and appends one line to
  `results_and_logs/<batch>/index.jsonl` when it finishes. A cell that already holds
  `decision.json` is skipped when a batch or queue is resumed.
- `python -m run_omniselect.run_queue` writes one log per job to `results_and_logs/logs/`.

```bash
python -m omniselect.store.replay results_and_logs/main_v2_3                     # verify stored selections
python -m omniselect.store.recompute results_and_logs/main_v2_3 --metric balanced_accuracy --split test
python -m tools.gate_replay results_and_logs/main_v2_3 --out results_and_logs/summary/gates
python -m run_omniselect.generate_csv --batch results_and_logs/main_v2_3 --out results_and_logs/summary/main_v2_3
```

Use a new batch name for each protocol or code revision. The index and the run record carry the
git sha of the code that produced a cell, and cells from different code states should not be
pooled. The paper's results are described in [../results/README.md](../results/README.md).
