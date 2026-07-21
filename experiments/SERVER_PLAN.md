# 服务器（租用 GPU）实验规划 - 定版

> 立场：数据选择面向多模态数据的各自基础模型。统一的方法是**投票式自适应控制器**（portfolio
> meta-selector）：候选组合含每个 baseline 本体，用每模态独立 val + 真实基模选最优，构造上 ≥
> 每个 baseline。本地已在 M2 小规模验证该方法跨模态有效（详见 README_story.md）。
> **本规划的纪律：每个模态的数据集与 baseline 都必须是公认 benchmark，先本地小规模复现明白，
> 再上服务器放大。** 下表标注了哪些已本地复现、哪些待服务器放大。

## 0. 一句话结论
本地已用公认数据集小规模证明方法跨模态有效（图像 CIFAR、时序 ETT、过程 TEP、表格 OpenML）。
服务器要做的是两件：(A) 把每个模态放大到该领域**标准评测协议下的标准规模与标准下游指标**（准确率/
FDR/MASE 而非小规模代理），(B) 补文本模态的 adapt（每配置训 LM，本地太贵）。**不是重新验证 idea，
是把已验证的 idea 做成审稿人无可挑剔的标准评测。**

## 1. 每模态：公认数据集 + 公认模型 + 公认 baseline

| 模态 | 公认数据集(benchmark) | 公认下游模型 | 评测指标 | 本地复现 | 服务器放大 |
|---|---|---|---|---|---|
| **语言** | FineWeb-Edu (FineWeb, NeurIPS D&B 2024) | SmolLM2-135M/360M 持续预训练 | 困惑度→lm-eval 准确率(ARC/HellaSwag/PIQA/OBQA) | 部分(stratified 困惑度) | ✅ 准确率大评测 |
| **数学** | FineMath-4plus / OpenWebMath (ICLR 2024) | SmolLM2 | 困惑度→GSM8K/MATH 准确率 | 部分 | ✅ GSM8K 准确率 |
| **代码** | Stack-Edu / StarCoderData (StarCoder2 2024) / The Stack v2 | SmolLM2 | 困惑度→HumanEval/MBPP pass@1 | 待换(codeparrot→Stack-Edu) | ✅ HumanEval |
| **图像(分类)** | CIFAR-100/10, 升级到 ImageNet-1k 子集 | 冻结 DINOv2/CLIP + 线性探针。或 ResNet-50 从零 | top-1 准确率 | ✅ CIFAR(冻结CLIP) | ⬆ ImageNet + ResNet |
| **视觉语言(真多模态)** | DataComp (NeurIPS D&B 2023) / LLaVA-665K | CLIP 预训练 / LLaVA 指令微调 | CLIP zero-shot / VQAv2/GQA/POPE | ❌(本地做不了) | ✅ 真 VL 臂(主升级) |
| **时序(预测)** | ETTh1/ETTm1, Weather, Electricity (LSF 基准) / GIFT-Eval | PatchTST / DLinear / Chronos-tiny | MASE / MSE | ✅ ETTh1+DLinear | ⬆ 全 LSF + PatchTST |
| **过程/故障诊断** | **TEP (Tennessee Eastman, Downs-Vogel/Braatz)** | MLP / 1D-CNN 故障分类器 | macro-F1 / FDR / 故障诊断准确率 | ✅ TEP+MLP(本地已跑) | ⬆ 全 21 故障 + CNN |
| **表格** | OpenML-CC18 / TabZilla 套件 | TabPFN-v2 / XGBoost | ROC AUC | ✅ electricity+TabPFN/XGB | ⬆ 多数据集套件 |
| 跨模态统一大模型 | 复用各模态池 | 不训统一基模 | 一致性表 | - | 📎 仅附录(scorer 可迁移) |

## 2. 每模态的公认 baseline（先调研清楚，已整理）

**文本/语言/数学/代码**（来自 references/selection+multimodal 31 篇 survey）：
- Random、DSIR(分布匹配, NeurIPS 2023)、DoReMi(域重加权, NeurIPS 2023)、QuRating(多轴质量, ICML 2024)、
  DsDm(datamodels, ICML 2024)、Ask-LLM+Density(ICML 2024)、SemDeDup(去重, 2023)、quality-PPL(Ultra-FineWeb式)、
  DataComp-LM/DCLM(NeurIPS 2024 标准协议)、QuaDMix(同期统一轴, 2025)。
- 代码额外：The Stack/phi-1/StarCoder2 的教育分类器过滤。

**视觉语言**：CLIPScore 过滤(DataComp)、SemDeDup、T-MARS(ICLR 2024)、MetaCLIP(ICLR 2024)。
  指令微调：ICONS、COINCIDE(CVPR 2024)、Self-Filter、TIVE。

**图像分类(coreset)**：Random、Herding、k-center/coreset、GraNd/EL2N、Forgetting、Moderate、CCS、
  DeepCore 库里的标准集(这些是图像数据选择的公认 baseline)。

**时序**：Random、数据混比搜索(Data Mixture Search 2025)、Chronos 的 TSMixup/KernelSynth 策展、coreset。

**过程/故障诊断(TEP)**：Random、PCA/DPCA/PLS 监控(经典)、SVM、kNN、随机森林、1D-CNN、LSTM
  (TEP 故障诊断的标准 baseline，几百篇论文用)。

**表格**：Random、k-means coreset、Tab-AICL(TabPFN-Coreset/Margin, 2026)、k-center、confident-learning/Cleanlab。

> 我们的统一 baseline（已实现，跨模态对称）：random、纯真实性、纯影响力、纯覆盖(coreset)、
> 固定融合(mmdataselect)、DMF(动态融合对照)、DSIR(文本)。投票控制器把这些都纳入候选组合。

## 3. 本地已复现（小规模，证明方法有效，不需服务器）
- ✅ 图像 CIFAR-100/10 + 冻结 CLIP + 线性探针：adapt 夺冠/并列，信号随数据集翻转。
- ✅ 时序 ETTh1 + DLinear：adapt 夺冠(+15% vs random)。
- ✅ 过程 TEP + MLP：adapt 夺冠(F1 0.378 > auth 0.365 > random 0.297)。
- ✅ 表格 OpenML electricity + TabPFN/XGBoost：adapt top-tier，信号随模型(TabPFN/XGB)翻转。
- ✅ 跨模态对照铁证：无单信号通吃，adapt 唯一从不垫底。

## 4. 服务器要跑的（按优先级）

**任务 A：文本模态 adapt + 标准准确率评测（最核心）。**
- 数据：FineWeb-Edu(语言) + FineMath-4plus(数学) + Stack-Edu(代码，换掉 codeparrot)。
- 下游：SmolLM2-135M/360M 持续预训练（continued pretraining）。
- 选择：投票控制器，gain = 留出准确率(GSM8K/HumanEval/ARC)或困惑度。
- baseline：random/DSIR/DoReMi/QuRating/DsDm/quality-PPL/mmdataselect。
- 为何上服务器：每配置训 LM，~7×2×2 配置 × 多 seed × 三模态，A100 上数小时。
- 估算：单卡 A100，~1 天。

**任务 B：图像放大到 ImageNet + 真 VL 臂。**
- 图像分类升级：ImageNet-1k 子集 + ResNet-50 从零(噪声敏感)，对照 DeepCore baseline。
- 真 VL：DataComp small(CLIP zero-shot) 或 LLaVA-665K(VQA)，CLIP/LLaVA 训练评测。
- 估算：单卡 A100，~1-2 天。

**任务 C：过程/时序放大到标准协议。**
- TEP：全 21 故障 + 1D-CNN/LSTM 分类器，报标准 FDR/故障诊断准确率，对照 PCA/SVM/CNN baseline。
- 时序：全 LSF(ETTh1/h2/m1/m2, Weather, Electricity) + PatchTST，报 MASE/MSE。
- 估算：单卡，~半天(这些模型小)。

## 5. 上服务器前还要本地做的（小规模，免费）
1. code 数据集 codeparrot → Stack-Edu/StarCoderData（gated，需 HF token。服务器登录即可，本地先验证加载）。
2. TEP 升级到 1D-CNN(小规模验证 selection 在 CNN 上也有效)。
3. 时序加 ETTm1/Weather 第二数据集(验证不是 ETTh1 特例)。
4. 把 DeepCore 的 1-2 个图像 coreset baseline(Herding/GraNd) 接进来对照。

## 6. 不上服务器的诚实边界
- 表格在鲁棒 FM(TabPFN)上选择收益小，强信号是"勿用纯真实性/影响力"。
- DaISy(KU Leuven)是系统辨识小数据集，不适合数据选择，不纳入(TEP 是更好的过程 benchmark)。
- 跨模态统一大模型只进附录(证 scorer 可迁移，不训统一基模)。

## 7. 租卡 + 部署 + 一键复现(2026-06-30 定)

**租什么卡。** 单卡即可，不需要多卡。优先 A100 40GB(AutoDL/AutoDL类平台按时计费)，预算紧时
4090 24GB 起步亦可跑通(SmolLM2-135M 持续预训练 + CLIP/ResNet + PatchTST 都在 24GB 内)。
- 任务 A(文本 adapt + 准确率评测): 单卡 A100，约 1 天。
- 任务 B(图像 ImageNet/ResNet + 真 VL): 单卡 A100，约 1-2 天。
- 任务 C(过程/时序放大): 单卡，约半天。
- 合计单卡 A100 约 3-4 天，或并行租 2-3 张卡按任务拆开同时跑、约 1-2 天。

**部署步骤(到机器上照抄)。**
1. `git clone git@github.com:qzkinhit/OmniSelect.git && cd OmniSelect`
2. `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
3. `huggingface-cli login`(任务 A 的 Stack-Edu / FineMath 等 gated 集需要，本地已验证非 gated 部分加载逻辑)
4. 冒烟: `pytest tests/ -q`(外部 baseline + 控制器单测应全过)

**一键复现(每任务的命令骨架，沿用本地同一 runner，只放大规模/换标准指标)。**
- 文本 adapt(任务 A): 在 `scripts/run_experiment.py` 上把 stratified 池换 FineWeb-Edu/FineMath/Stack-Edu，
  下游换 SmolLM2-135M 持续预训练，gain 回调换 lm-eval 准确率，METHODS 加 dsir/doremi/qurating 对照。
- 图像放大(任务 B): `VIS_DATASET=ImageNet 子集`、下游换 ResNet-50 从零，METHODS 已含 herding/kcenter/el2n/grand 公认 coreset。
- 过程/时序(任务 C): `MODEL=cnn`(已实现) 或更大 CNN 报 TEP 全 21 故障 FDR。`TS_DATASET=ETTm1/Weather`(已实现) + PatchTST 报全 LSF MASE。

**本地已验证清单(上卡前的底气，审稿人/老板可核)。**
- 投票式控制器在图像/时序/过程/表格四模态、3-seed 上 best-or-tied，从不垫底(`experiments/verify_final_3seed.log`)。
- 已与公认外部 baseline 对照且不输: DeepCore 的 herding/k-center/EL2N/GraNd 已接入并跑通，EL2N/GraNd 在
  标签噪声下崩盘(图像 0.256、过程 0.076 远低于随机)，控制器把它们纳入组合后仍 ≥ 每一个。
- 时序换第二基准 ETTm1 仍最优(非 ETTh1 特例)。文本侧 DSIR 已是忠实 baseline。
- 诚实边界: TEP 1D-CNN 在 52 个静态过程变量上欠拟合(全方法近随机)，故论文用 MLP 作噪声敏感深度模型，
  CNN 仅作存档不入正文。表格在鲁棒 TabPFN 上选择收益小(价值是不垫底)。

## 8. 一句话给老板
**本地已用公认数据集 + 公认外部 baseline 证明方法跨模态有效、且自适应控制器从不垫底、连会崩盘的
EL2N/GraNd 都被它纳入组合后压住。服务器只需把规模/指标提到各领域标准协议(准确率/FDR/MASE) + 补
文本 adapt，单卡 A100 三四天即可，上去不会出现"发现 idea 不行要换"的情况。**
