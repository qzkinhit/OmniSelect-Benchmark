# Paper2 OmniSelect — 跨模态数据选择实验总规划

> 立场（一句话）：**数据选择面向多模态数据的各自基础模型**。一套统一的选择框架
> （真实性 × 影响力 × 覆盖 三正交通道 + 冲突感知**自适应控制器** + 预算约束选择），
> 每个模态服务它自己的基模（文本→小型 LM，图像→视觉模型，表格→TabPFN，时序→预测模型），
> 用各自真实模型 + 真实评测 + 各自文献立场，而非把一切序列化成文本。

## 0. 目录结构

```
experiments/
  README.md                 ← 本文件，总规划 + 状态 + how-to-run
  README_story.md           ← 跨模态故事与观点性结论（docs/cross_modal_story.md 副本）
  README_defensibility.md   ← 31 篇论文数据集防质疑调研
  README_grounding.md       ← 每模态 grounding（公认数据集/M2模型/评测）
  text/                     ← 文本臂（语言/数学/代码）
  vision/                   ← 图像臂（CIFAR-100 + CLIP）
  tabular/                  ← 表格臂（OpenML + TabPFN，待加 XGBoost 对照）
  timeseries/               ← 时序臂（待建，Chronos/PatchTST）
runner 脚本（带 import 路径，留在 scripts/）：
  scripts/run_experiment.py            ← 文本（stratified 分层预算）
  scripts/run_vision_experiment.py     ← 图像
  scripts/run_tabular_experiment.py    ← 表格
  scripts/run_timeseries_experiment.py ← 时序（待建）
框架核心：
  src/mmdataselect/fusion/adaptive.py  ← AdaptiveController（自适应控制器）
  src/mmdataselect/signals/            ← 真实性/影响力/冗余 通道
  src/mmdataselect/selectors/budget_select.py ← 预算约束 importance×diversity
```

## 1. 核心方法创新：AdaptiveController

既有多信号融合（DMF/Meta-rater 式）用**固定权重**，把"哪个信号重要"当超参数预设。
我们把它变成**由每模态自身 held-out 增益在线决定**的自适应量：给定三通道分数 + 一个
modality 自带的 `held_out_gain(selection)->float` 回调（图像=探针 top-1，表格=TabPFN AUC，
文本=-PPL，时序=-MASE），控制器搜小网格的通道权重 + 多样性强度，按 held-out 增益选最佳，
再在全池全预算重选。**同一个控制器跨模态复用，只换 gain 回调。**

关键纪律（踩过的坑，务必守）：
- **held-out val 必须独立、与 test 同分布**（不能从带噪池子抠干净样本，会泄漏偏向真实性）。
- 搜索用的 gain 必须用**该模态真实下游模型**（表格用 TabPFN，不能用 logreg 代理，否则选错信号）。
- adapt 最终选择在**全池全预算**重做，跟其他方法等预算公平。

## 2. 四模态状态表

| 模态 | 基模 | 数据/评测 | 状态 | 关键结果 | 防质疑 |
|---|---|---|---|---|---|
| **图像 vision** | 冻结 CLIP + 线性探针 | CIFAR-100 + 标签噪声 / top-1 acc | ✅ **完成，最强** | adapt **0.4252 最优**，auth +16%，固定融合崩(0.340) | 强（DataComp/SemDeDup 立场，且 M2 几秒一个，不贵） |
| **表格 tabular** | TabPFN-v2（in-context） | OpenML electricity + 噪声 / AUC | ✅ 完成，待强化 | 紧预算 mmdataselect 0.857 最优；**auth/influence 掉点**（与图像相反） | 中-强（ICD-TabPFN/Tab-AICL 立场）；**待加 XGBoost 非FM对照** |
| **文本 text** | 从零小型 LM（stratified） | FineWeb-Edu/FineMath/code + 噪声 / 困惑度 | ✅ 固定法完成，adapt 待补 | image/math 等各模态最佳配置随域变（佐证）；code 待换 Stack-Edu | 强（看家三模态），**code 数据集 codeparrot→Stack-Edu 待修** |
| **时序 timeseries** | DLinear 从零 | ETTh1 OT 窗口 + 噪声 / MASE | ✅ **完成，强！** | **adapt 0.996 全场最优(3/3 seed)**，比 random 好 ~15%，固定融合第二 | 中-强（Chronos/Moirai 数据策展立场；ETTh1 是标准 LSF 基准） |
| **过程 TEP（故障诊断）** | MLP 故障分类器 | Tennessee Eastman 52变量22类+噪声 / macro-F1 | ✅ **完成，强！公认benchmark** | **adapt 0.391 全场最优(3-seed)**，+22% vs random；RF 上信号翻转 | 强（TEP 是化工故障诊断金标准，数百篇论文） |
| 跨模态统一大模型 | 一个模型吃所有 | — | 📎 附录 | 只证 scorer 可迁移，不训统一基模 | — |

## 3. 完美故事（真 3-seed 铁证，详见 README_story.md）

| 方法 | 图像 acc | 表格 AUC |
|---|---|---|
| 纯真实性 auth_only | **王 0.4252** | **冷宫 0.848** |
| 固定融合 mmdataselect | **崩 0.340** | **王 0.857** |
| **自适应 mmds_adapt** | **王 0.4252** | **最优梯队 0.855** |

**观点性结论：数据质量是多面的，"哪一面重要"随模态与基模翻转；没有单信号或固定配置能跨模态
通吃，唯有 held-out 增益驱动的自适应控制器两边都稳。** 选择收益随基模"质量敏感度"缩放（噪声
敏感模型收益大，鲁棒 FM 收益小），但自适应方法从不垫底。

## 3b. 与公认外部 baseline 的对照（3-seed 均值，回应审稿人）

把 DeepCore 系列接进 4 个模态并纳入策略组合（视觉/TEP/表格 herding+k-center+EL2N+GraNd，时序 herding+k-center，文本 DSIR 现成忠实实现）。

| 方法 | 图像 acc↑ | 时序 MASE↓ | 过程 F1↑ | 表格 AUC↑ |
|---|---|---|---|---|
| herding (DeepCore) | 0.387 | 1.084 | 0.351 | 0.840 |
| k-center (DeepCore) | 0.374 | 1.138 | 0.325 | 0.842 |
| EL2N (DeepCore) | 0.256 | 不适用 | 0.076 | 0.214 |
| GraNd (DeepCore) | 0.256 | 不适用 | 0.076 | 0.214 |
| **mmds_adapt（本文）** | **0.424** | **0.967** | **0.391** | **0.859** |

**铁证：EL2N/GraNd 按样本误差取最难样本，在标签噪声池里恰好选中错标样本，在全部三个分类模态（视觉0.256、过程0.076、表格0.214）一致崩盘到远低于随机。几何核心集平庸。投票式控制器把它们全纳入组合后在每个模态都不弱于其中每一个。** 数据见 `external_baselines_3seed.log` 与 `tabular_external_3seed.log`。

## 4. how-to-run（都在项目 .venv，M2/MPS）

```bash
# 图像（编码缓存后秒级）
POOL_N=4000 TEST_N=2000 SEED=0 METHODS=random,coreset,auth_only,mmdataselect,mmds_adapt \
  .venv/bin/python scripts/run_vision_experiment.py
# 表格（TabPFN v2，CPU；紧预算 0.3 才显选择收益）
TAB_DATASET=electricity BUDGET_FRAC=0.3 SEED=0 .venv/bin/python scripts/run_tabular_experiment.py
# 文本（stratified 分层预算，MPS）
STRATIFY=1 INFL_KIND=pplq SEED=0 .venv/bin/python scripts/run_experiment.py
```

## 5. 详细下一步（按优先级，本轮执行）

1. **强化表格** — 加 `MODEL=xgboost` 非FM对照。预期：XGBoost 噪声敏感 → auth（清洗）翻回来有用
   → 证明"哪个信号重要也随**模型**变，不只随模态"，故事更完整。同时把 code 数据集修成 Stack-Edu。
2. **时序臂** — 新建 `scripts/run_timeseries_experiment.py`：ETT/Monash 真实序列 + 质量波动噪声
   （平移/缩放/翻转/重复）→ DLinear/PatchTST-small 从零或 Chronos-tiny → MASE。同一套三通道 +
   AdaptiveController。力争 selection 明显有用（紧预算 + 噪声敏感小模型）。
3. **文本 adapt**（核心模态）— 在 stratified 上跑 AdaptiveController（gain=held-out PPL），
   M2 上贵 → 可小网格或放 AutoDL。
4. **回填论文** — 把 per-modality + 自适应控制器 + 跨模态对照写进实验章。

## 6. 已知诚实边界（写进 limitations）
- 表格在鲁棒 FM 上选择收益小（紧预算才显），强信号是"别用纯 auth/influence"。
- 时序是最高风险模态（文献框成数据增强/混比而非定额选择，小模型是代理）→ go/no-go，不行进附录。
- 文本 adapt 的 held-out 增益是 LM 训练，成本高。
