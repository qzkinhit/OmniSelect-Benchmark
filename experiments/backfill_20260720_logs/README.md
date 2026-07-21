# 2026-07-20 补跑批次的真实运行日志

补齐此前遗漏的 5 个数据集(ETTh2、DaISy CSTR、DaISy steamgen、CIFAR-10 统一协议、CIFAR-100N)
在 2026-07-20 那批 backfill 的原始 stdout 日志。此前这批日志只留在服务器 `/root/` 下,
未同步进仓库,现已拉取归档。

- `ETTh2_dlinear_full_backfill-ETTh2-dlinear-full-20260720.log`、
  `daisy_cstr_dlinear_full_backfill-daisy_cstr-dlinear-full-20260720.log`、
  `daisy_steamgen_dlinear_full_backfill-daisy_steamgen-dlinear-full-20260720.log`:
  三个数据集均含 seed 0/1/2 完整过程,包含各自第一次 seed 0 因路径问题失败
  (`python_exit=127`)、重试后成功(`python_exit=0`)的完整记录,如实保留不作删减。
- `cifar10_clip_full_backfill-cifar10-clip-full-20260720.log`、
  `cifar100n_full_backfill-cifar100n-full-20260720.log`:
  CIFAR-10 统一协议与 CIFAR-100N 的 seed 0 运行日志。

每个日志末尾的 `saved -> .../results.json` 路径与 `results_canonical/` 下对应
数据集的 results.json 一一对应,可交叉核对逐方法指标数字。
