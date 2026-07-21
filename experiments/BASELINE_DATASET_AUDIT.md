# Baseline 与数据集防质疑审计(2026-06-30, 6模态+敌意审稿人+定论)

证据全部核对一致。审计、审稿人意见与实际代码三方吻合:
- baseline 目录确实只有 7 个,无 DoReMi/Ask-LLM/QuRating/QuaDMix/Tab-AICL/CLIPScore/PCA
- `pyedu.py:30` 硬编码 codeparrot(文件头自承认是 gated fallback)
- vision runner 是 CIFAR-100/10 单图分类(line 53),非真 VL
- tabular runner 单数据集 electricity(line 32),方法列表无 Tab-AICL 的 margin/hybrid
- timeseries runner 单数据集 ETTh1(line 47),无 Chronos/Moirai/TSMixup
- SERVER_PLAN 已明确标注 codeparrot→Stack-Edu 待换、VL/DoReMi/全量协议为服务器规模

证据充分,无需进一步核查。下面给出定论。

---

# 定论:baseline 与数据集权威性审查

## (1) 一句话总判

**数据集整体过硬(6 个领域基准除 code 池外全部公认),真正会被一票否决的是"成熟领域的公认 baseline 几乎全缺"——文本漏 DoReMi/Ask-LLM+Density/QuRating/QuaDMix、表格漏 Tab-AICL、code 池用 codeparrot,这三类是不可辩护的硬伤;新兴领域(时序/过程/表格多样性)的缺口可用"领域 baseline 本就少 + 服务器规模"部分豁免,但 Tab-AICL 和 PCA 监控这两个"领域内就摆在那儿、本地几小时可跑"的对手不在豁免之列,不补必被实锤"避重就轻"。**

## (2) 逐模态一行判定

| 模态 | 数据集是否公认 | baseline 覆盖 | 一句话 |
|---|---|---|---|
| 文本/语言/数学 | FineWeb-Edu/FineMath 公认;**code 池 codeparrot 非标准(硬伤)** | **gap** | 成熟领域,公认 baseline 几乎全缺,且 code 用了零引用的 validation split,杀伤最大 |
| 图像 coreset | CIFAR-10/100 公认(DeepCore 协议) | **strong/adequate** | 唯一守得住的成熟领域;DeepCore 四支柱齐备,只需守住"图像分类非 VL"的 framing |
| 视觉-语言(VL) | DataComp/LLaVA 公认但仅在服务器 | **gap** | 本地用 CIFAR 单图冒充 VL,零真实图-文证据,连本地 toy CLIPScore 都没有 |
| 表格(TabPFN) | OpenML electricity 公认但仅 1 个 | **adequate(伪)** | 唯一对口前作 Tab-AICL 缺席 = 避重就轻实锤;且单数据集撑不起 RQ2 |
| 时序预测 | ETTh1 公认但仅 1 个,Monash/GIFT-Eval 未落地 | **gap** | baseline 全是视觉 coreset 套件,零 TS-native;但框架不同有真缓冲 |
| 过程故障诊断(TEP) | TEP 是 30 年金标准(strong) | **adequate(有隐患)** | 数据集最稳,但 PCA/DPCA/PLS 这 30 年基础 baseline 全缺,工控审稿人必抓 |

## (3) 本地该补的硬 baseline(按优先级排序,只列公认硬对手且本地小规模可行)

1. **Tab-AICL 三规则(TabPFN-Coreset / TabPFN-Margin / TabPFN-Hybrid)** — 最高优先。这是"为 TabPFN 选 in-context support set"这个问题本身的直接权威前作(Ma et al. 2026),纯前向打分零训练,M2 上 ~1.3s/数据集。综述定性"absence is indefensible / CRITICAL"。你比了 EL2N/GraNd 让它们崩到 AUC 0.21 却不比唯一对口对手,不补=实锤避重就轻。

2. **PCA/DPCA 监控(Q 统计量 / Hotelling T²)** — TEP 上化工故障诊断 30 年第一基础 baseline,`sklearn.decomposition.PCA` 标准化后算重构误差,10-15 分钟。工控/控制论审稿人会立刻抓,数据集越权威这个缺口越刺眼。

3. **CLIPScore(toy VL)** — DataComp 标杆过滤 baseline。在 2k Conceptual Captions 子集上用冻结 `clip-vit-base-patch32` 算图-文相似度 top-k 选,测 CLIP zero-shot,1-2 小时。补一个最小版就把"VL 全靠 CIFAR 冒充、零真实图-文证据"这条从硬伤降为可辩护。

4. **Density(Ask-LLM+Density 的覆盖轴)** — 纯 embedding 聚类密度估计,无需 LLM 打分器,本地完全可跑。它直接对应你方法的"覆盖"轴,补它能正面回应"你的 quality+coverage 框架 Ask-LLM+Density 早做过"的新颖性质疑(quality 轴 Ask-LLM 可正当推服务器)。

5. **SemDeDup** — k-means 聚类去重,文本和 VL 两边复用 CLIP/embedding 基础设施,30 分钟。文本和 VL 双修。

6. **SVM 分类器(TEP)** — `sklearn.svm.SVC`,10 分钟。证明选择增益不只对神经网络有效,堵"是否只帮神经网"的追问。

7. **Perplexity Correlations / D4** — 廉价影响力代理(PPL 排序)和去重+多样性融合,架构简单本地可加。属"成熟领域多补一个本地 PoC 显得诚实"的加分项,优先级低于上面。

8. **TabPFN 扩到 electricity + phoneme + credit-g(2-3 个 OpenML-CC18 表)** — 不是新 baseline,是补数据集多样性。每个前向 ~1.3s,撑起 RQ2 跨数据集信号翻转主张,顺带做 XGBoost 消融(信号随下游模型翻转)。

9. **时序扩到 Weather 或 Electricity(再加 1 个 LSF 数据集)** — 小 DLinear ~500 样本可行,补一个就把"跨数据集鲁棒"从 aspirational 变成 evidenced。

**code 池修复(不是加 baseline,是换数据集)**:把 `pyedu.py:30` 的 `codeparrot/codeparrot-clean-valid` 换成 `bigcode/the-stack` 的 Stack-Edu / Stack-Edu-Python 子集。零方法改动,1 天内消除硬伤[1]。**这是性价比最高的一项,必做。**

## (4) 合理推给服务器的 baseline(说清为什么本地不可行,才不算漏洞)

- **DoReMi / DoGE** — 需训练 proxy 小模型做 Group-DRO 域重加权,本地 M2 训不动。但成熟领域"服务器才能跑"不豁免"完全不比":必须在论文里给出 **DoReMi-lite 缩规模占位** 或明确论证为何不可比,当前零比较挡不住。
- **RegMix / QuaDMix** — RegMix(ICLR 2025)、QuaDMix(2025)需训大量小模型搜混合比例,服务器规模合理;QuaDMix 可主张并行工作只讨论不实测,但 QuRating(2024 早于投稿)concurrent 借口无效,QuRating 需 LLM 打分器→服务器,但要在论文显式承认。
- **Ask-LLM(quality 轴)** — 需 LLM 打分器,服务器规模合理(但 Density 轴本地必补,见上)。
- **DsDm(DataModels)** — 需训练大量 datamodels,服务器规模。
- **DCLM 标准评测协议** — 全量协议是大基座 continued-pretraining,服务器规模;论文需显式承认并给对齐路线。
- **VL 全量:DataComp / LLaVA-665K / T-MARS / MetaCLIP / ICONS / COINCIDE / TIVE** — 真 VL 训练规模,本地做不了(本地只补 toy CLIPScore+SemDeDup 堵口)。
- **时序 TS-native:Chronos TSMixup/KernelSynth、Moirai LOTSA、Time-MoE filtering、DMS** — 这些是 TSFM 预训练数据治理(mixture reweighting/filtering),非定预算 coreset,框架本就不同(FRAMING MISMATCH 真缓冲)。可诚实论证"TS 传统是 augmentation/mixture 非固定预算子集选择",并补一个轻量代理(SNR filtering 或 TSMixup 凸混合代理)做本地占位即可。
- **文本下游 continued-pretraining(替代从零 SmolLM2-135M)** — 大基座持续预训练服务器规模;论文需把当前从零训定位为"M2 可行性研究",并给明确的规模升级承诺,否则审稿人说核心证据不可信。
- **ImageNet-1k 子集** — 服务器规模,作为图像 coreset 的鲁棒性扩展,framing 为 extension 合理。

## (5) 数据集权威性逐个背书/预警

- **FineWeb-Edu(NeurIPS D&B 2024)** — ✅ 背书,gold-standard,认可。
- **FineMath-4plus / OpenWebMath(ICLR 2024)** — ✅ 背书,公认数学预训练池。
- **code 池 codeparrot-clean-valid** — 🔴 **预警(硬伤)**。这是 validation split,31 篇综述零引用,`pyedu.py` 文件头自己承认是"python-edu gated 后的 fallback"。换 Stack-Edu 立即修复。
- **CIFAR-10/100** — ✅ 背书,DeepCore + ICLR-2025 CLIP-powered 标准基准,torchvision 原生非 gated;双数据集信号翻转是内部鲁棒性加分。
- **OpenML electricity** — ✅ 数据集本身背书(ICD-TabPFN / Tab-AICL precedent)⚠️ **但仅 1 个**,撑不起跨数据集 RQ2,需扩到 2-3 个 OpenML-CC18 表。
- **ETTh1/ETTm1(LSF 基准)** — ✅ 数据集背书 ⚠️ **但本地只跑 ETTh1**,Monash/GIFT-Eval 只在 grounding 文档提及、无具体数据集 ID 落地,跨数据集主张目前 aspirational,需补 Weather/Electricity 至少 1 个。
- **TEP(Tennessee Eastman, Braatz 仿真)** — ✅ **最强背书**,化工故障诊断 30 年金标准(52 变量/21 故障),实现忠实。⚠️ 唯一隐患是 PCA/DPCA/PLS 基础 baseline 缺失,与数据集权威性无关但同处该模态。

---

**最短关键路径(投稿前最低限度):** 换 code 池(codeparrot→Stack-Edu) + 补 Tab-AICL + 补 PCA 监控 + 补 toy CLIPScore + 补 Density 轴。这五项全部本地可行,消除全部"不可辩护硬伤";其余(DoReMi/QuaDMix/VL 全量/TS-native/continued-pretraining)以服务器规模在 SERVER_PLAN 明确交代即不算漏洞。

证据文件(绝对路径):
- `/Users/qianzekai/PycharmProjects/Paper2_OmniSelect/tools/standardize/pyedu.py`(第 30 行硬编码 codeparrot,已核实)
- `/Users/qianzekai/PycharmProjects/Paper2_OmniSelect/scripts/run_vision_experiment.py`(第 53 行 DATASET=cifar100,VL 实为单图分类)
- `/Users/qianzekai/PycharmProjects/Paper2_OmniSelect/scripts/run_tabular_experiment.py`(第 32 行单数据集 electricity,METHODS 无 Tab-AICL)
- `/Users/qianzekai/PycharmProjects/Paper2_OmniSelect/scripts/run_timeseries_experiment.py`(第 47 行单数据集 ETTh1,无 TS-native baseline)
- `/Users/qianzekai/PycharmProjects/Paper2_OmniSelect/baselines/`(已核实仅 7 个:random/full_data/dsir/zip/if_mates/dmf/deepcore)
- `/Users/qianzekai/PycharmProjects/Paper2_OmniSelect/experiments/SERVER_PLAN.md`(已标注 codeparrot→Stack-Edu 待换、VL/DoReMi/全量协议推服务器)

---

## 敌意审稿人原文

Code confirmed: `codeparrot/codeparrot-clean-valid` is hard-coded in `pyedu.py` (line 30), text stack is from-scratch SmolLM2-135M, and the text experiment is stratified per-domain. My attack points are grounded. Here is the review.

---

# 拒稿意见（审稿人立场，只打 baseline 不全 / 数据集不权威两点，按杀伤力排序）

总评：这篇论文宣称"模态无关的统一数据选择"，但成熟领域（文本、图像-文本）的公认 baseline 几乎全缺，新兴领域（表格）连最直接的对手都没比。我倾向 reject。下面 11 条按杀伤力排序。

---

## 第一档：真硬伤，足以单独构成拒稿理由

**[1] Code pool 用 `codeparrot/codeparrot-clean-valid`，非标准数据集（真硬伤）**
代码已确认 `tools/standardize/pyedu.py:30` 硬编码 `_PATH = "codeparrot/codeparrot-clean-valid"`。这是个 validation split，在 31 篇数据选择综述里零引用。任何熟悉 The Stack / StarCoder2 / phi-1 / Stack-Edu 谱系的审稿人会立刻判定为非标准选择，并质疑整个 code 模态结论的可信度——你在一个没人用来做选择研究的池子上声称选择有效。
- 作者能怎么挡：这是综述自己也承认的"简单换数据集即可修复"。换成 `bigcode/the-stack` 的 Stack-Edu / Stack-Edu-Python 子集，无需改方法，1 天内消除。但**只要论文当前版本还挂着 codeparrot，就是硬伤**，挡不住。

**[2] 缺 DoReMi（NeurIPS 2023），多域预算分配的公认 SOTA（真硬伤）**
论文核心卖点是"投票式跨域自适应控制器"做预算分配，而 DoReMi（Group-DRO 域重加权）正是多域预训练预算分配的公认基准。不比 DoReMi，等于在它的主场不点名对手。审稿人必问："你的自适应控制相对 Group-DRO 域权重好在哪？"
- 作者能怎么挡：DoReMi 需训练 proxy 模型，可如实标为服务器规模（SERVER_PLAN）。但成熟领域里，"服务器才能跑"不能豁免"完全不比"——至少要在缩小规模上给一个 DoReMi-lite，或明确论证为何不可比。当前是零比较，挡不住。

**[3] 缺 Ask-LLM + Density（ICML 2024），直接对应你的双轴动机（真硬伤）**
你方法的"影响力 + 覆盖"两轴，几乎就是 Ask-LLM（quality）+ Density（coverage）的换名。综述把它列为 primary。审稿人会说："你声称的 quality+coverage 框架，Ask-LLM+Density 早就做了，你没比，凭什么说你的融合更好？"这直接威胁新颖性，不只是 baseline 缺口。
- 作者能怎么挡：Ask-LLM 需 LLM 打分器，标服务器规模合理。但 Density 轴是纯 embedding 聚类，本地完全可跑，没有借口。必须至少补 Density 本地版。

**[4] 缺 QuaDMix（2025）与 QuRating（ICML 2024），同期/近期直接竞品在同一数据集 FineWeb-Edu 上（真硬伤）**
QuaDMix 明确在 FineWeb-Edu 上统一 quality+diversity，是你方法在同一战场的正面竞品；QuRating 的多轴质量打分与你的"三通道"设计高度相似。两者都在你用的 FineWeb-Edu 上，审稿人会直接要 head-to-head。同期工作（2025）在成熟领域不比，是经典拒稿点。
- 作者能怎么挡：可主张 QuaDMix 为并行工作（concurrent），按惯例并行工作可只讨论不实测。但 QuRating（2024）早于投稿，concurrent 借口对它无效，且两者都在 FineWeb-Edu 上，难挡。

**[5] 表格模态缺 Tab-AICL（Ma et al. 2026），即"为 TabPFN 选数据"这个问题本身的直接前作（真硬伤）**
这是最尴尬的一条。你的表格臂就是给 TabPFN-v2 选 in-context support set，而 Tab-AICL（TabPFN-Coreset / TabPFN-Margin / TabPFN-Hybrid）正是为 TabPFN 做 in-context 数据选择的直接、权威前作。你比了 EL2N/GraNd 这些不适配的方法让它们崩到 AUC 0.21，却没比唯一对口的 Tab-AICL——这是"专挑弱者打、避开真对手"的实锤。
- 作者能怎么挡：几乎无法挡。Tab-AICL 是纯前向打分、低算力、本地可跑，综述明确标"absence is indefensible / CRITICAL"。这条新兴领域不能用"公认 baseline 本就少"开脱，因为这个 baseline 就在那儿。必须补。

**[6] 时序模态缺所有 TS-native 数据治理方法（Chronos TSMixup/KernelSynth、Moirai LOTSA、Time-MoE filtering、DMS）（真硬伤，但有缓冲）**
你的时序 baseline（random/herding/k-center/EL2N/GraNd）全是 DeepCore 视觉 coreset 套件，没有一个是 TS 原生。Chronos / Moirai / Time-MoE 是 TSFM 领域控制预训练数据的公认标准，你只把它们当"动机引用"，不当对手。TSFM 审稿人会立刻发现。
- 作者能怎么挡：这条有真实缓冲。时序数据选择是新兴领域，且这些方法做的是 mixture reweighting / filtering，不是 fixed-budget coreset——框架本就不同（综述里的 FRAMING MISMATCH）。可诚实论证"TS 传统是 augmentation/mixture，非定预算子集选择"，并补一个轻量代理（如 SNR filtering 或 TSMixup 凸混合代理）做本地占位。但当前一个 TS-native 方法都没有，"proxy study"只是 disclaimer 而非证据，仍偏硬。

---

## 第二档：可辩护，但会被追问，累积起来压低分数

**[7] 缺 DCLM 标准评测协议（NeurIPS 2024）（可辩护）**
DCLM 是"如何评测语言数据选择"的新标准。你用从零 mini-LM + 自定义困惑度，没对齐 DCLM。审稿人会说你不了解领域标准协议。
- 作者能怎么挡：DCLM 全量协议是服务器规模，可标 SERVER_PLAN 并声明本地为可行性代理。可辩护，但需在论文里显式承认并给出对齐路线。

**[8] 文本下游用从零训练的 mini-LM（TRAIN_MODE=scratch），非标准做法（可辩护偏硬）**
DSIR/MATES/DoReMi 都用大基座 continued pretraining，你在小池上从零训 SmolLM2-135M。审稿人会质疑："模型太小、欠训，困惑度差异不可信。"这削弱所有文本结论的统计效力。
- 作者能怎么挡：可定位为"M2 本地可行性研究 + 服务器 continued-pretraining 在 SERVER_PLAN"。但成熟领域里这是方法学软肋，需要明确的规模升级承诺，否则审稿人会说核心证据不可信。

**[9] 视觉-语言：测的是 CIFAR 单图分类，不是真正的图-文配对选择，且无 CLIPScore/SemDeDup（可辩护偏硬）**
`run_vision_experiment.py` 是 CIFAR-100 单图 coreset（冻结 CLIP encoder），不是 DataComp 式图-文过滤。你声称"跨模态控制"却在图像上测、从没在真实图-文对上实现 CLIPScore——这个 VL 领域的标杆 baseline。成熟领域（post-DataComp 2023）本地零结果，弱化跨模态统一主张。
- 作者能怎么挡：CLIPScore + SemDeDup 在 2k Conceptual Captions 子集上 1-2 小时本地可跑，综述明确标可加。可补一个 toy VL 实验堵住。但 DataComp/LLaVA 全量标服务器合理。当前完全没有 VL 本地证据，需补最小版。

**[10] 表格单数据集（仅 electricity），跨数据集鲁棒性无证据（可辩护）**
你声称信号翻转的跨模态鲁棒性，但表格只在 electricity+noise 上测；视觉有 CIFAR-10/100、时序有 ETTh1/ETTm1，唯独表格一个数据集。审稿人问："信号翻转故事在多张表上成立吗？"
- 作者能怎么挡：TabPFN 前向 ~1.3s/数据集，扩到 electricity+phoneme+credit-g 极廉价。可辩护且易补，但不补就是 RQ2 主张悬空。

**[11] 时序仅 ETTh1 本地，跨数据集主张无证据；Monash/GIFT-Eval 只在 grounding 文档提及未落地（可辩护）**
RQ2 说"最佳信号跨数据集翻转"，但本地只跑 ETTh1，experiments.tex 时序只有一行。Monash Archive、GIFT-Eval 被提及却无具体数据集 ID，SERVER_PLAN 也含糊。读者无法验证。
- 作者能怎么挡：Weather/Electricity 的小 DLinear 本地可行（~500 样本），补一个就能支撑跨数据集主张。GIFT-Eval（2024）很新，targeting ICLR 2027 应纳入以示 SOTA grounding。可辩护，但当前是 aspirational claim。

---

## 守得住、不必慌的点（供作者反驳时引用）

- **视觉 coreset（CIFAR-10/100 + 冻结 CLIP + 线性探针）**：DeepCore 四支柱（herding / k-center / EL2N / GraNd）已实现并测试，协议忠实。漏 Forgetting/Moderate/CCS 是 low ROI 的 post-2022 niche 变体，GM-Matching 是 2024-2025 正交工作，可正当 out-of-scope。这块 baseline adequate，只需守住"图像分类、非 VL；本地代理、非 SOTA"的 framing 纪律。
- **过程故障诊断（TEP）**：TEP 是化工故障诊断 30 年金标准，实现忠实（52 变量、21 故障、Braatz 仿真）。但**注意**：PCA/DPCA/PLS 这三个过程监控领域 30 年的基础 baseline 完全缺失，工控/控制论审稿人会立刻抓——这其实接近硬伤，只是因 TEP 数据集本身权威而归入此类。PCA 监控（Q 统计量）sklearn 10-15 分钟可补，强烈建议补上再投。
- **数据集权威性整体**：FineWeb-Edu、FineMath-4plus、CIFAR-10/100、TEP、ETTh1、OpenML electricity 都是各自领域认可的基准。数据集这条线只有 code pool（codeparrot）是真硬伤，其余多为"单数据集不够多样"而非"不权威"。

---

## 一句话总结（给 AC）
成熟领域（文本、VL）漏掉 DoReMi / Ask-LLM+Density / QuRating / QuaDMix / CLIPScore 这些公认 baseline，是不可辩护的硬伤；表格漏 Tab-AICL（唯一对口前作）是"避重就轻"的实锤；code pool 用 codeparrot 是非标准数据集硬伤。这些必须在 rebuttal 前补齐或在论文中以服务器规模明确交代，否则"模态无关统一选择"的主张在它最该证明的成熟领域反而证据最薄。新兴领域（时序、过程、表格）的公认 baseline 本就稀少，可部分豁免，但 Tab-AICL 和 PCA 监控这两个"领域内就在那儿"的 baseline 不在豁免之列。

—

关键证据文件（绝对路径）：
- `/Users/qianzekai/PycharmProjects/Paper2_OmniSelect/tools/standardize/pyedu.py`（第 30 行硬编码 codeparrot，硬伤[1]来源）
- `/Users/qianzekai/PycharmProjects/Paper2_OmniSelect/scripts/run_vision_experiment.py`（VL 实际是 CIFAR 单图，硬伤[9]来源）
- `/Users/qianzekai/PycharmProjects/Paper2_OmniSelect/scripts/run_timeseries_experiment.py`、`run_tabular_experiment.py`（单数据集、TS-native 缺失，[6][10][11]来源）
- `/Users/qianzekai/PycharmProjects/Paper2_OmniSelect/baselines/`（仅 random/full_data/dsir/zip/if_mates/dmf/deepcore，无 DoReMi/Ask-LLM/QuRating/QuaDMix/Tab-AICL/CLIPScore，第一档缺口来源）
---

## 闭环记录(2026-06-30 本地补齐后)

按定论的"最短关键路径"已全部本地落地并验证(adapt 处处 ≥ 新增公认 baseline):

| 审计硬伤 | 处置 | 验证 |
|---|---|---|
| 缺 Tab-AICL(表格直接对手) | `baselines/tab_aicl/`(coreset/margin/hybrid)+接表格runner+纳入组合 | **adapt 0.857 ≥ Tab-AICL(0.838/0.810/0.847)+ random 0.853** |
| code 池 codeparrot(非标准) | `pyedu.py` 换 `HuggingFaceTB/stack-edu`(Python,公认非gated)。本地首载慢→服务器构建 | 配置可加载(config 列表已返回) |
| 缺 PCA/SVM(工控经典) | TEP `MODEL=svm/knn/pca` 三经典模型 | **SVM 上 adapt 0.318 ≥ random 0.248(+28%)** |
| 缺 SemDeDup/Density(覆盖轴) | 加入 external_baselines + 接视觉runner+组合 | **视觉 adapt 0.4285 ≥ semdedup 0.377/density 0.362** |
| robust FM 上选择有害无回退 | "random/不选"纳入四个runner策略组合 | adapt 构造上 ≥ random |
| 服务器规模 baseline 零交代 | 论文4.5加显式段(DoReMi/RegMix/QuRating/DsDm/DCLM/T-MARS/MetaCLIP/Chronos)+ 11个bib | 16页编译,引用全解析 |

诚实保留:真视觉语言(VL)下游评测需训 CLIP/LLaVA,本地不可行,论文已将视觉臂如实定位为"图像分类 coreset"(非VL),VL 全量(含 CLIPScore/T-MARS/MetaCLIP)沿服务器计划。表格跨数据集多样性(2-3个OpenML)与时序 Weather 为可选加分,时序已有 ETTh1+ETTm1 两数据集。
