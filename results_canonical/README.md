# Canonical small results (whitelisted)

The 16 `results.json` files under this directory are the exact run artifacts behind the
paper's canonical tables (`experiments/canonical_tables.json`, `latex_FINAL_*`):

- `run_id=pubcore-paired-20260716T1754-*` (12 files): the single-code-state paired main
  batch (vision CIFAR-100, time series ETTh1, process TEP, tabular electricity x seeds
  0/1/2; PAIRED_RNG=1, per-seed shared fit seed, `sel_sha12`/`train_order_sha12` per row,
  independent parity re-check 9/9 MATCH).
- `run_id=ctrl-ts2-noproxy-20260717T0922-*` (1 file): the ETTh1 seed-2 controller cell
  re-run after the quadmix style proxy was withdrawn as PROTOCOL_INVALID_DUPLICATE_IDS
  (the controller may not adopt an invalid candidate).
- `run_id=text-qz4-20260717T0933-*` (3 files): the text-arm finetune lane (STRATIFY pool,
  quadmix published-core branch), seeds 0/1/2.

Each file was copied from the runtime tree with per-file SHA256 equality verified at copy
time. Directory layout preserves the original `outputs/` run-id structure. Large run logs
and the full experiment-artifact backup ship separately as a versioned release archive
(see `docs/ARTIFACTS_INDEX.md`).
