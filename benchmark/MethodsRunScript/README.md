# Standalone method runners

Each `run_<name>/` directory runs one method without the controller and writes the same run
record as a controller cell. The shell wrapper takes the track, dataset, seed and protocol. Any
further flags go to `tracks/common/experiment.py`.

```bash
bash benchmark/MethodsRunScript/run_herding/run.sh vision cifar100 0 v2 --smoke
bash benchmark/run_all.sh process tep21 0 v2 --smoke
```

The run record is described in `docs/RUN_RECORD.md`. Methods that read inputs a track does not
provide (for example EL2N on a forecasting track) are skipped with a reason in log.txt and in
leaderboard.json under `skipped_strategies`.
