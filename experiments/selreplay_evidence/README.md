# Selection replay evidence (raw selected_ids)

`experiments/selection_manifest_verdicts.json` records the PASS/FAIL verdict of a one-time
selection-replay verification (`scripts/validate_selection_manifests.py`) for the pubcore
main-table four arms (vision/CIFAR-100, tep/TEP21, tabular/electricity, timeseries/ETTh1).
That verdict file only stores the outcome, not the underlying data it was computed from.
The raw per-method `selected_ids` lists lived under the gitignored `outputs/` tree and were
never committed.

This directory recovers those raw files from a 2026-07-17 server backup snapshot
(`server_backup_postaudit2_20260717T120138/`) and commits them so the evidence behind the
verdict is actually in the repo, not just the pass/fail summary. Each `results.json` here has,
per method, `selected_ids` (the exact pool-local integer ids chosen), `training_order`, and the
`sel_sha12`/`train_order_sha12` fingerprints that `selection_manifest_verdicts.json` cites.

Scope: only these four arms/datasets (matches the validator's own documented scope, it
explicitly does not cover text/CIFAR-100N/CIFAR-10-original/ImageNet-100/ETTm1/ETTh2/DaISy/
Chronos/TEP-calib2). Going forward, `baselines/deepcore_original/run_original_protocol.py`
saves `selected_ids`/`sel_sha12` directly into its own `results_canonical/.../results.json` for
every method it runs, so future original-protocol runs (CIFAR-10, ImageNet-100, ...) no longer
need a separate replay step to capture this.

Only seed 0's raw `selected_ids`/`training_order` is committed here as a representative
illustration (repo convention: new evidence/showcase additions show seed 0 only). The
`sel_sha12`/`train_order_sha12` fingerprints for seeds 1 and 2 are already public in
`results_canonical/**/seed_{1,2}/results.json`, and their PASS verdicts for all three seeds are
recorded in `experiments/selection_manifest_verdicts.json`.
