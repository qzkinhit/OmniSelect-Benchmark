# full_data baseline

Keep **every** record in the pool — the upper-bound reference. No selection happens;
`method/full_select.py:select(ids)` returns all ids unchanged, so `n_selected == n_total`.
Any budget-constrained method is measured by how close it gets to this ceiling while
keeping far fewer records.

```
baselines/full_data/
  method/full_select.py   select(ids) -> all ids (identity, pure stdlib)
  run_full_data.py        load processed jsonl -> keep all -> write manifest
```

## Run

```bash
python baselines/full_data/run_full_data.py --config configs/experiments/<exp>.yaml
```

Reads `cfg["data"]["processed_path"]` and writes
`outputs/<experiment_id>_full_data/manifests/{manifest.json,selected.jsonl}` in the
shared manifest contract (`method="full_data"`), so it plugs into the same
`run_train` / `run_eval` as every other method — point `--output-dir` at this folder.

The configured `select.budget` is read for logging symmetry only and is intentionally
ignored: full_data always keeps the entire pool.

Reference: standard *train-on-all-data* upper bound used in data-selection studies.
