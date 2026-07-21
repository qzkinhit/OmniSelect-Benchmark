# CCS 官方原协议锚点(audit 四.1)执行协议

日期:2026-07-17(UTC)。执行位置:服务器 `omni`,与 QZ3(PIDs 372096/372241)并行的唯一新增 GPU lane。
本文件只记录协议与证据路径;**结果数字在 lane 收线前一律不写**。

## 1. 代码来源与 SHA

- 官方 repo:https://github.com/haizhongzheng/Coverage-centric-coreset-selection(ICLR 2023, Zheng et al.)
- 服务器克隆位置:`/root/autodl-tmp/ccs_official`
- 论文时点 commit `c8812e5f33afd433b39285f5a9172d50c65d9eec`(2023-02-15)已实际 checkout 验证:**该 commit 只有 README(指向 OpenReview 附件),无任何代码**。
- 因此实际使用 main HEAD:**`b37f166ace760e17f4d5baae17a828bdb88c6667`**(2023-09-21,官方后续发布的清理版代码;与 `docs/baseline_fidelity_ledger.md` §6 "官方 repo b37f166 直接可用" 的预案一致)。这是一处如实记载的偏差:paper-era commit 无代码可跑。
- 代码 **未做任何修改**,逐条使用 README 给出的命令。

## 2. 数据

- CIFAR-10 经官方 loader(`core/data/MiscDataset.py`,torchvision `datasets.CIFAR10`,download=True)。
- 归一化 mean [0.4914,0.4822,0.4465] / std [0.2470,0.2435,0.2616];训练增广 `RandomCrop(32, padding=4, padding_mode="reflect")` + `RandomHorizontalFlip`(与原文 4-pixel crop + 水平翻转一致)。
- 多伦多源直连仅 ~60kB/s,故从 AutoDL 公共数据集拷贝:`/root/autodl-pub/cifar-10/cifar-10-python.tar.gz` → `/root/autodl-tmp/ccs_official/data/cifar10/`,md5 `c58f30108f718f92721af3b95e74349a` 与 torchvision 官方校验值一致。

## 3. 精确命令(lane 脚本 `/root/lane_ccs_anchor.sh` 逐字包含)

python 为 `/root/miniconda3/bin/python`(torch 2.8.0+cu128, torchvision 0.23.0, CUDA 可用),cwd `/root/autodl-tmp/ccs_official`。

1. **全量 200-epoch 训练(采训练动态,供 AUM)**
   `python train.py --dataset cifar10 --gpuid 0 --epochs 200 --lr 0.1 --network resnet18 --batch-size 256 --task-name all-data --base-dir ./data-model/cifar10 --data-dir ./data`
2. **重要性分数计算(含 accumulated_margin = AUM)**
   `python generate_importance_score.py --gpuid 0 --base-dir ./data-model/cifar10 --task-name all-data --data-dir ./data`
3. **CCS(AUM) 90% 剪枝重训**(ratio 0.1,`--mis-ratio 0.3`;README 示例即此,β=30% 与原文 App.B CIFAR-10@90% 一致)
   `python train.py --dataset cifar10 --gpuid 0 --iterations 40000 --task-name ccs-0.1 --base-dir ./data-model/cifar10/ccs --coreset --coreset-mode stratified --data-score-path ./data-model/cifar10/all-data/data-score-all-data.pickle --coreset-key accumulated_margin --coreset-ratio 0.1 --mis-ratio 0.3 --data-dir ./data`
4. **random 90% 剪枝配对参照**
   `python train.py --dataset cifar10 --gpuid 0 --iterations 40000 --task-name random-0.1 --base-dir ./data-model/cifar10/random --coreset --coreset-mode random --coreset-ratio 0.1 --data-dir ./data`
5. **CCS(AUM) 70% 剪枝重训**(ratio 0.3,`--mis-ratio 0.1`;β=10% 取自原文 ar5iv 2210.15809 Appendix B 的 CIFAR-10 β 表:30%/50%→0,70%/80%→10%,90%→30%)
   `python train.py ... --task-name ccs-0.3 ... --coreset-ratio 0.3 --mis-ratio 0.1 ...`(其余参数同第 3 条)
6. **random 70% 剪枝配对参照**
   `python train.py ... --task-name random-0.3 --coreset-mode random --coreset-ratio 0.3 ...`(其余参数同第 4 条)

## 4. 实际生效的训练配置(读代码核对,train.py b37f166)

- ResNet-18,batch 256,SGD lr 0.1 momentum 0.9,`CosineAnnealingLR(T_max=总迭代数, eta_min=1e-4)`,重训 40k iterations。
- **偏差(如实记录)**:代码硬编码 `weight_decay=5e-4, nesterov=True`(train.py L189 附近),而原文文本写 0.0002(2e-4) weight decay。本锚点用官方代码不改动,故实际 wd=5e-4+Nesterov。对表时须带上这一条。
- **种子(如实记录)**:官方代码 **没有任何 seed 参数/`manual_seed` 调用**(全 repo grep "seed" 无命中)。"seed 0" 无法在不改代码的前提下设置;本次为 1 次无种子运行,记作 **run-1** 而非 seed-0。
- 取 `Best acc`(代码在每次测试点保存 best,结束再测 last 并取 max)作为该 run 的结果读数。

## 5. Smoke(不计入结果)

- 2026-07-17,1-epoch smoke:exit 0,196 iters/epoch,约 5.8s/epoch(≈34 it/s),test acc 37.23%(1 epoch,仅证明 import/数据/GPU 前后向可用)。配置头(Namespace 全参数)已打印在 smoke 日志 `./data-model/cifar10-smoke/smoke/log-train-smoke.log`。
- smoke 时 nvidia-smi:GPU RTX 6000D 85651 MiB 总量,启动前占用 12409 MiB(QZ3 环境),lane 启动后约 14884 MiB。

## 6. Lane 运行体制

- 脚本:`/root/lane_ccs_anchor.sh`(nohup + setsid 启动);日志:`/root/lane_ccs_anchor.log`;每阶段退出码追加至 `/root/ccs_anchor_stage_exits.txt`;**全部阶段 exit 0 才 touch `/root/CCS_ANCHOR_DONE`**。
- 启动确认:2026-07-17,lane PIDs 410518/410519/410522,stage 1 epoch 速度 ≈5.1s/epoch。
- **ETA 估算(由观测 it/s 外推)**:stage1 200ep×~5.5s ≈ 20 min;stage2 分数计算约 5–10 min;stage3–6 各 40k iters @ ~34 it/s ≈ 20 min ×4;合计 **≈ 1.7–2 小时**。
- GPU guard:若 QZ3 离开 CPU 阶段、总显存逼近 80GB,可对本 lane `kill -STOP`(恢复 `kill -CONT`),脚本注释已写明;不实现自动守护。

## 7. 原文对表计划与诚实验收规则

原文数字(ar5iv 2210.15809 Table 3,CIFAR-10,5 seeds 均值±std;全量参照 95.23):

| 设置 | CCS(AUM) | random |
|---|---|---|
| 70% 剪枝 | 93.00±0.16 | 90.94±0.38 |
| 90% 剪枝 | 86.08±0.61 | 79.04±1.53 |

验收规则(1 个无种子 run):

- **主判据(90% 剪枝,优先)**:我们的 CCS(AUM) best acc 落在 **86.08 ± max(3σ, 1.0) = [84.25, 87.91]** 内记 PASS(σ=0.61;取 3σ 与绝对 1 个百分点的较大者作为单 run 复现容差,吸收 wd 5e-4 vs 2e-4、无种子、torch 2.8 vs 论文时代框架的合理漂移)。
- **次判据(70% 剪枝)**:93.00 ± max(3σ, 1.0) = [92.00, 94.00]。
- **配对相对判据**:CCS@90% − random@90% 应显著为正(原文 +7.04);若绝对值 PASS 但配对差消失,锚点仍判 FAIL。
- **明确声明**:1 个 run **不能验证 ±0.61 的 std**,只能验证均值落带;std 复验需 ≥5 seeds,且官方代码无 seed 口径,严格意义的 seed 复现本就不可得(只能做多次独立无种子重复)。
- 若 PASS:`docs/baseline_fidelity_ledger.md` 的 STATUS 才允许讨论从 INCOMPLETE 升档(须另行独立对齐审查,本文件不自动升档);本文外部表中 CCS 行仍是 **EL2N-binned 本地实现**,锁档不变——本锚点验证的是"官方 AUM 版在原协议下可对表",不是给本地实现升档。
- 若 FAIL:如实记录读数与差距,排查顺序:wd 偏差 → 无种子波动(补跑第 2、3 个 run)→ β/mis-ratio 口径 → 框架版本;不得静默调参凑数。

## 8. 结果登记(收线后填,当前 NOT-CAPTURED)

- run-1 CCS@90% best acc:NOT-CAPTURED(lane 运行中)
- run-1 random@90%:NOT-CAPTURED
- run-1 CCS@70%:NOT-CAPTURED
- run-1 random@70%:NOT-CAPTURED
- 各阶段退出码 / marker:NOT-CAPTURED(见 `/root/ccs_anchor_stage_exits.txt`、`/root/CCS_ANCHOR_DONE`)

## §8 结果回填(2026-07-17,lane 收线 ALL STAGES EXIT 0)

| 阶段 | 结果 | 原文 Table 3 | 判定 |
|---|---|---|---|
| s1 全量 200ep | Best acc **95.38**(epoch 507 附近) | 95.23 | +0.15,吻合 |
| s3 CCS 90%(mis 0.3) | **86.50** | 86.08±0.61 | **1σ 内;预注册验收带 [84.25, 87.91] PASS** |
| s4 random 90% | 83.23 | 79.04±1.53 | 偏高 +4.19(高方差档;偏差方向利随机,使 CCS 相对优势结论保守成立) |
| s5 CCS 70%(mis 0.1) | **93.13** | 93.00±0.16 | 1σ 内 |
| s6 random 70% | 91.15 | 90.94±0.38 | 1σ 内 |

阶段退出码全 0(`/root/ccs_anchor_stage_exits.txt`),marker `/root/CCS_ANCHOR_DONE`。
诚实边界(§6 已预先披露):单次 run(官方代码无 seed 参数)不能验证原文 ±std;
代码 wd=5e-4 与论文正文 2e-4 不符(按"代码不改"原则实跑);random-90% 偏差如实记录。

**判定(定稿措辞,2026-07-17):official released-implementation anchor PASS。**
本锚点验证的是**官方已发布实现**(main HEAD b37f166)在原数据、原命令下可对表原文 Table 3,
**不是** "strict paper-protocol reproduction verified":paper-era commit(c8812e5)只有 README 无代码;
已发布代码硬编码 wd=5e-4 而论文正文写 2e-4;代码无任何 seed 口径,本次为无种子 run-1。
因此该锚点以**独立档位「official released-implementation anchor」**入账;
`reproduction_verified` 作为 strict-paper-protocol 标志**保持 NONE**;
本地 EL2N-binned CCS 实现的档位(local-implementation)不因本锚点升档。
canonical 证据:`experiments/ccs_anchor_canonical.json`(服务器原件 `/root/ccs_anchor_canonical.json`)。
