# QuaDMix style proxy 作废决定(PROTOCOL_INVALID_DUPLICATE_IDS)

日期:2026-07-17。审计项:二.6(重复 ID 作废)。证据文件:`experiments/selection_manifest_verdicts.json`(2026-07-17 自服务器 `/root/autodl-tmp/OmniSelect/experiments/selection_manifest_verdicts.json` 只读拉回本地,sha 同源;验证器 `scripts/validate_selection_manifests.py` v2,fail-closed)。

## 1. 事实(重放验证器输出,非推断)

- 验证器 scope:`"pubcore main-table 4 arms x 3 seeds ONLY"`(CIFAR-100 / ETTh1 / TEP21 / electricity)。summary:verified_cells=195,fail_count=12。因存在 FAIL,成功 marker `PUBCORE_4ARM_SELECTION_MANIFESTS_VERIFIED_OK` **未写出**(2026-07-17 服务器复核无此 marker)。
- 12 个 FAIL 全部为 "duplicates under no-replacement protocol":
  - 旧 quadmix style proxy 11 格:vision seed0/1/2,tep seed0/1/2,tabular seed0/1/2,timeseries seed1/2(timeseries seed0 的 quadmix 无重复,重放通过);
  - 控制器 1 格:timeseries seed2 mmds_adapt(该 seed 控制器 chosen=quadmix,选集继承了 quadmix 的重复)。
- 自重放 SELECT_ONLY manifests(`outputs/{arm}/*/run_id=selreplay-*/seed_*/results.json`)重算的名义预算 vs 唯一样本数(2026-07-17 服务器只读重算):

| arm seed | n_ids | unique | dup 率 |
|---|---|---|---|
| vision s0 | 2000 | 1207 | 39.7% |
| vision s1 | 2000 | 1206 | 39.7% |
| vision s2 | 2000 | 1171 | 41.4% |
| tep s0 | 1200 | 883 | 26.4% |
| tep s1 | 1200 | 869 | 27.6% |
| tep s2 | 1200 | 859 | 28.4% |
| tabular s0 | 1500 | 1008 | 32.8% |
| tabular s1 | 1500 | 991 | 33.9% |
| tabular s2 | 1500 | 1019 | 32.1% |
| timeseries s1 | 900 | 898 | 0.2% |
| timeseries s2 (quadmix) | 900 | 899 | 0.1% |
| timeseries s2 (mmds_adapt, chosen=quadmix) | 900 | 899 | 0.1% |

## 2. 判定

**旧 QuaDMix style proxy 的全部历史数值一律 PROTOCOL_INVALID_DUPLICATE_IDS**:selected_ids 含重复意味着实际唯一样本数低于名义预算,该行不是公平的等预算对照行,其与 random/其他基线的任何比较均无效。

作废范围与依据的区分:

1. **replay-verified**(直接证据):pubcore-paired 批 4 臂的上述 11 个 quadmix 格 + timeseries seed2 控制器格。
2. **same-code-path**(机制推断,逐格标注):重复是确定性代码缺陷——`src/mmdataselect/selectors/external_baselines.py::quadmix` 中 (a) 分位桶用双侧闭区间判成员,边界样本同时落入相邻两桶;(b) 桶内 farthest-first 的 argmax 不掩蔽已选局部索引,特征距离全零时反复选同一样本。四个模态 runner(`scripts/run_vision_experiment.py`、`run_timeseries_experiment.py`、`run_tep_experiment.py`、`run_tabular_experiment.py`)与文本 runner(`scripts/run_experiment.py` method "quadmix")调用同一函数,故 legacy 07-04 批、tep_calibrated2、CIFAR-100N、ETTh2/ETTm1/DaISy/Chronos 各家族及文本 lane(legacy-text-quadmix-20260716T1550,selected_ids 未存档,NOT-CAPTURED)的旧 quadmix 数值同判作废,即使未逐一重放。
3. **不受影响**:`quadmix_pub`(published-core transfer,Gumbel top-k 无放回)重放通过;DMF 两行、其余全部基线重放通过(verified_cells=195)。timeseries seed0 的 quadmix 虽个体无重复,但同属作废行(方法级作废,不做逐格豁免)。

## 3. 决定(audit 偏好)

1. **style proxy 撤出主表与外部表**,并**撤出控制器 portfolio**(向前适用):`quadmix_pub` 为 QuaDMix 唯一代表行。
2. 稿件动作**已完成(2026-07-17,commit da31d36)**:AAAI 稿 tab:external 的 "QuaDMix-style proxy (2025)" 行已撤下,叙述改为撤回披露(重放审计发现重复 ID,违反无放回等预算协议),Implementation disclosure 同步;表源生成器 EXT_ROWS 亦已换为 quadmix_pub(commit 061ad54)。ICLR 稿外部表本就只有 published-core 行、无 style proxy 行(2026-07-17 复核)。
3. **受影响的 canonical 控制器格 = pubcore timeseries seed2(chosen=quadmix)**:已在 `experiments/master_coverage.json` published_core_paired_batch 格打 flag;**定向重跑排期在 QZ3 收线之后**(2026-07-17 复核 QZ3 PIDs 372096/372241 存活约 7 小时,铁律不得触碰),且以修复提交 9116339 评审+部署为前提。canonical_tables.json 的 timeseries 控制器数字在重跑落地前继续沿用但携带本 flag。
4. **修复状态**:本地提交 **9116339**("Fix QuaDMix proxy duplicate selections")已存在——半开分位桶唯一归属(searchsorted)、farthest-first 已选索引掩蔽、稳定去重补齐、exact-k unique 断言 fail-fast;`tests/test_baseline_fidelity.py` 新增用例,13 tests pass。**未部署**:QZ3 收线前服务器代码冻结,部署前须评审。
5. 台账联动:`docs/baseline_fidelity_ledger.md` §7 + 硬性锁档 4 附注;`experiments/master_coverage.json` 全部旧 quadmix 格 grade=PROTOCOL_INVALID_DUPLICATE_IDS(reasons 逐格区分 replay-verified / same-code-path)。reproduction_verified 仍 = NONE,一切档位锁不变。
