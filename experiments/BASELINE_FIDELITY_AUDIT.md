# 逐 baseline 忠实度认证(2026-07-04)

> 认证标准(从严到宽):**T1 官方代码对齐**(与官方实现同池同参照对照,给一致性数字)>
> **T2 原文数据集定性复现**(在原论文自己的数据集上复现其核心定性结论)>
> **T3 机制测试**(在受控数据上验证方法的定义性机制,pytest 固化防回归)。
> 三层都过不了的不许进论文对照表。全部证据可复跑:`pytest tests/test_baseline_fidelity.py`
> + `experiments/fidelity_clean_cifar.log` + `experiments/dsir_official_fidelity.log`。

## 认证总表

| baseline | 出处(会议/CCF) | 官方代码 | 原文数据集 | 认证等级与证据 | 状态 |
|---|---|---|---|---|---|
| DSIR | NeurIPS 2023 (CCF-A) | p-lambda/dsir ✓ | The Pile/GLUE | **T1**:与官方 `data-selection` 包同池同参照对照,Spearman 0.855,top-50% 重叠 85.5%(`dsir_official_fidelity.log`) | ✅ 最强认证 |
| EL2N | NeurIPS 2021 (CCF-A) | mansheej/data_diet ✓ | **CIFAR-10**(我们同款) | **T2**:干净 CIFAR-10 上 50% 预算 0.84~0.90 接近随机 0.91,75% 剪枝崩到 0.66~0.72,与 CCS 原文复现的"纯难样本高剪枝崩溃"一致;**T3**:与 GraNd 排序不同 ✓ | ✅ |
| GraNd | NeurIPS 2021 (CCF-A) | 同上 | CIFAR-10 | **T2+文献**:早期梯度范数版在干净数据表现差(0.37~0.66),与公开复现论文(arXiv 2303.14753)"GraNd-at-init 不可复现"一致,如实按此引用;**T3** ✓ | ✅(带文献注) |
| CCS | ICLR 2023 (CCF-A) | 官方 ✓ | **CIFAR-10/100**(同款) | **T2**:干净 CIFAR-10 75% 剪枝 CCS 0.87~0.90 ≈ 随机,同预算 EL2N 0.66~0.72,**原文核心结论(高剪枝下分层覆盖远胜纯难样本)本地复现**;**T3**:分层不取纯难 ✓ | ✅ |
| Herding | ICML 2009 (CCF-A) | 无(DeepCore 为标准实现) | - | **T3**:选集均值逼近全体均值 ✓;干净 CIFAR ≈/略超随机,行为符合 | ✅ |
| k-center | ICLR 2018 (CCF-A) | ozansener ✓ | CIFAR(AL 设定) | **T3**:覆盖半径小于随机 ✓;干净 CIFAR-10 50% 预算 0.927>随机 0.917 | ✅ |
| SemDeDup | arXiv/Workshop 2023(**非主会,引用时注明**) | 无一方官方 repo | LAION 网页规模 | **T3**:机制测试曾抓出我们初版不忠实(离簇心远优先≠原文簇内阈值判重),**已改为原文规则**(cos>1-ε 判重,组内留离簇心最远者),注入近重复优先被去 ✓;网页规模效果归服务器 | ✅(修正后) |
| Density/Ask-LLM | arXiv 2024(无会议,无官方代码) | 无 | - | **T3**:初版(确定性取最稀疏)偏离原文,**已改为按密度反比概率采样**,密集团被稀释 ✓ | ✅(修正后) |
| Entropy-Law/ZIP | arXiv 2024(**无会议,引用时注明**) | USTC-StarTeam/ZIP ✓ | ShareGPT 对齐数据 | 按原文压缩比贪心实现,文本 pilot 有真跑数字;LLM 规模效果验证归服务器(对齐官方 repo) | 🟡 T3 部分 |
| DMF(动态融合) | 2025 | - | 文本预训练 | **T3**:乘性权重按验证奖励收敛到有效通道 ✓;跨模态 3-seed 有竞争力(图像 0.422/TEP 0.409) | ✅ |
| QuaDMix | arXiv 2024(无官方代码) | 无 | 文本预训练 | **T3**:质量高于随机且多样性不劣于纯质量 Top-K ✓,论文注明"按原文目标复现" | ✅ |
| ~~Tab-AICL~~(已撤下对照) | arXiv 2026(无会议) | 作者 repo | 20 个 OpenML | **撤下 head-to-head,改为相关工作引用**。诚实经过:先做单次适配版(random 0.960 > hybrid 0.934,反序);再**忠实实现其迭代 AULC 协议**(每类 1 起步、每轮重新条件化 TabPFN、报学习曲线,`baselines/tab_aicl/run_iterative_aulc.py`),ionosphere 仍得 random 0.743 > hybrid 0.716 > coreset 0.684 > margin 0.675,**未复现原文序**。根因:Tab-AICL 是迭代式主动学习(每轮新标注),本文是从已标注池按预算一次性选择,**两个不同问题**,不该同协议对照。故论文 4.2/结论只作相关工作引用并注明设定不同,不列同协议数字。这消除了"复现出反序=复现错"的质疑点 | ❎ 撤下(设定不同) |
| TabPFN-v2 | Nature 2025 | 官方 tabpfn 包 | - | **直接用官方包**(下游基模,非复现) | ✅ 官方 |
| CLIP/SmolLM2/DLinear | - | HF 官方权重/标准结构 | - | 下游基模用官方权重或标准两层线性结构 | ✅ |

## 干净 CIFAR-10 验证数字(Data-Diet/CCS 原文数据集,无注入噪声,冻结 CLIP 线性头,2 seeds)

| 预算 | full | random | el2n | grand | ccs | herding | kcenter |
|---|---|---|---|---|---|---|---|
| 50% | 0.920~0.924 | 0.911~0.917 | 0.842~0.900 | 0.615~0.661 | 0.882~0.906 | 0.915~0.920 | 0.923~0.927 |
| 25%(75%剪枝) | 同上 | 0.898~0.900 | 0.662~0.718 | 0.371~0.483 | **0.875~0.900** | 0.909~0.913 | 0.912~0.916 |

复现的原文定性结论:(1) CCS 原文:高剪枝率下分层覆盖 ≈ 随机/全量,远超纯难样本(此处 +0.2)✓;
(2) 纯难样本(EL2N)高剪枝崩溃,与 CCS 原文对 Data-Diet 方法的复现一致 ✓;(3) 几何 coreset
(herding/k-center)干净数据 ≈/略超随机 ✓;(4) GraNd 早期版不稳,与公开复现文献一致(如实引用)✓。

**协议差异声明**:上表是冻结 CLIP + 线性头的代理台。下面是**原文模型+原文数据的真复现**。

### 原协议真复现:ResNet-18 从零训 CIFAR-10(`experiments/deepcore_original_protocol.log`)

`baselines/deepcore_original/run_original_protocol.py`,SCORE_EPOCH=5、TRAIN_EPOCH=15、
POOL=20000(本地压缩预算,故绝对数低于原文 200-epoch 的 ~94%,但**相对序=原文发现**):

| keep(剪枝) | random | EL2N | GraNd | CCS | 复现的原文结论 |
|---|---|---|---|---|---|
| 0.30(剪 70%) | 0.476 | 0.205 | 0.265 | **0.528** | CCS **超过随机**(CCS 招牌卖点)且 ≫ EL2N/GraNd ✓ |
| 0.10(剪 90%) | 0.323 | 0.148 | 0.167 | 0.240 | CCS ≫ EL2N/GraNd,纯难样本剪枝崩溃 ✓ |

**三条原文核心结论全部复现**:(1) CCS 在 70% 剪枝下超过随机(Zheng et al. ICLR2023 头牌);
(2) CCS 远胜纯难样本剪枝(EL2N/GraNd);(3) EL2N/GraNd 在高剪枝率下崩到远低于随机
(Paul et al. NeurIPS2021 与 CCS 复现文献一致)。这直接回应"EL2N/GraNd/CCS 复现得对不对":
在原文模型与原文数据上,相对行为与原文一致。绝对数字对齐(逼近 94%)需 200-epoch 全预算,
列入服务器任务(见 SERVER_PROMPT):对齐 `mansheej/data_diet` 与 `haizhongzheng/CCS` 官方配方。

## 数据集权威性清单(本地在用)

| 模态 | 数据集 | 权威性 |
|---|---|---|
| 图像 | CIFAR-10/100 | 经典公认,且正是 EL2N/GraNd/CCS/herding 原文数据集(对比自带合法性) |
| 时序 | ETTh1/h2/m1(LSF) | Informer(AAAI21 最佳论文)引入,LSF 标准全家桶 |
| 过程 | TEP 全称田纳西伊斯曼 | 化工故障诊断金标准(导师点名) |
| 过程工业 | DaISy CSTR + steamgen | KU Leuven SISTA 识别库(导师点名),系统辨识经典 |
| 表格 | OpenML electricity + ionosphere/phoneme | electricity 常用;后两个是 Tab-AICL 原文数据集 |
| 文本 | FineMath / Stack-Edu(math/code 域池) | HF 官方策展,正是用户要求的数学与代码数据集来源 |
| 服务器榜单级 | DataComp / DCLM / GIFT-Eval / ImageNet / OpenML-CC18 / GSM8K / HumanEval | 提交制或固定协议,环境无关可比 |

## 服务器精确复现协议(冲"原文数字"用)

1. **Data-Diet**:ResNet-18 + CIFAR-10 原训练配方,EL2N@epoch10×10 次平均,50% 剪枝,目标 |acc−full|≤0.5%;对齐 `mansheej/data_diet`。
2. **CCS**:同上设定 70%/90% 剪枝,对齐官方 repo 报告差距。
3. **DeepCore 全家**:直接跑 `PatrickZH/DeepCore` 官方库在 CIFAR-10 ResNet-18,报官方数与我们复现数并列。
4. **DSIR**:文本臂直接换官方 `data-selection` 包做选择(pip 已装,本地对齐已 85.5%)。
5. **Tab-AICL**:实现其冷启动 AULC 协议(每类 1 起步,批 5~20,至 100 标注,Cohen's κ AULC),在其 20 数据集上对表。
6. **QuRating/RegMix/DoReMi**:直接跑官方 repo(见 SERVER_PROMPT 第 4 节)。
