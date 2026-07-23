<!-- LANG -->
[English](README.md) | **简体中文**

# OmniSelect

**面向多模态各自基础模型的跨模态鲁棒数据选择,以配对认证裁决实现。**

固定训练预算下盲目堆叠样本会稀释有效信号,所以选择保留哪些数据很关键。OmniSelect 针对的核心难点是,**没有任何单一质量信号或固定融合能够跨模态通用**,最佳信号随模态、随下游模型、甚至随数据集翻转,一个方法在某个模态很强,换到另一个模态却可能跌到随机以下。

OmniSelect 把选哪个选择策略这件事,变成每个模态用它自己的干净验证集与自己的下游模型来回答的问题。它持有一个**冻结且已核查的候选策略组合**,含真实性、影响力、覆盖三个互补质量信号,它们的融合,以及每一个被对照的基线,基线是**作为候选被执行**的,绝不读取任何基线的已发布结果,随后在组合上做**裁决**。本文部署采用的裁决方法先在验证集的构造半区上冻结一个有序的参照与挑战者配对,再在裁决半区上用指标专属的单边置信半径**认证**是否切换,从而使采纳有害挑战者的概率不超过 `δ = 0.05`(定理 4'')。它只有拿到证书才切换,否则保持参照,这正是它在每个模态都不落最优梯队之外、并规避灾难性失效的原因。

> Python 包沿用历史名 `mmdataselect`,论文中系统名为 **OmniSelect**。

本仓库同时是 **OmniSelect-Benchmark**:一套跨模态共享的数据选择方法评测台架,统一等预算
协议,baseline 作为候选被执行(绝不是只作对照的列),每个数字都有已提交的 `results.json`
为证。完整协议说明(任务定义、噪声分类、候选集/参考集/留出集隔离、运行产物格式、种子
约定)见 [BENCHMARK_zh.md](BENCHMARK_zh.md),新增数据集或 baseline 见
[docs/CONTRIBUTING_DATASETS.md](docs/CONTRIBUTING_DATASETS.md)(英文)。

---

## 快速开始(纯 CPU,不需 GPU,不需下载)

```bash
git clone https://github.com/qzkinhit/OmniSelect-Benchmark.git
cd OmniSelect-Benchmark
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"              # 核心依赖 + 纯 CPU 测试依赖
pytest -q                             # 纯 CPU 测试套件
run_scripts/reproduce_cached.sh       # 从已提交的小结果重建论文的 canonical 表
```

`reproduce_cached.sh` 从仓库内 `results_canonical/` 已提交结果重建 9 数据集主结果表(`experiments/canonical_tables_seed0.json`),每个数字直接读自某个 `results.json` 行,不需 GPU,几秒完成。公开资产共覆盖 12 个任务(另含 CIFAR-10、ImageNet-100 与五域文本臂),完整清单见 [`results_canonical/README.md`](results_canonical/README.md) 与下文[结果](#结果)一节。

## 自己跑 baseline 和 OmniSelect

一个模态 × 一个数据集 × 一次运行,方法可任选子集:

```bash
# 训练依赖只需装一次(视觉/时序/表格臂建议用 GPU)
pip install -e ".[train,eval,arms]"   # arms = tabpfn/xgboost/chronos-forecasting,表格/时序臂需要
python scripts/fetch_data.py                      # 拉取并按 SHA 校验数据集

# <模态> <数据集> <种子> [方法列表]
run_scripts/run_single_arm.sh vision      uoft-cs/cifar100    0
run_scripts/run_single_arm.sh timeseries  ETTh1       0   random,auth_only,mmds_adapt
run_scripts/run_single_arm.sh tabular     electricity 0
```

覆盖主结果表全部 9 个数据集的命令(对应[结果](#结果)一节的表):

```bash
run_scripts/run_single_arm.sh vision      uoft-cs/cifar100    0   # CIFAR-100
VIS_NOISE=real run_scripts/run_single_arm.sh vision uoft-cs/cifar100 0   # CIFAR-100N(同一批图像,真实人工标注)
run_scripts/run_single_arm.sh timeseries  ETTh1       0
run_scripts/run_single_arm.sh timeseries  ETTm1       0
run_scripts/run_single_arm.sh timeseries  ETTh2       0
run_scripts/run_single_arm.sh timeseries  daisy_cstr  0   # DaISy CSTR
run_scripts/run_single_arm.sh timeseries  daisy_steamgen 0   # DaISy steam generator
run_scripts/run_single_arm.sh tep         21          0   # TEP21
run_scripts/run_single_arm.sh tabular     electricity 0
```

CIFAR-10(`run_scripts/run_single_arm.sh vision uoft-cs/cifar10 0`)属于 12 任务清单中的
追加协议(原版 DeepCore 协议,兼作 baseline 忠实度核查),不计入上面 9 行主结果表;
见 `results_canonical/vision/cifar10_full/` 与 `docs/baseline_fidelity_ledger.md`。

在以上任一命令前加 `SPLIT_EXPORT_DIR=<目录>` 可额外落盘该次运行里全部方法共享的
池/验证/测试划分(`pool_ids`/`val_ids`/`test_ids` 与随机种子配方);不加则
`results.json` 只带 `sel_sha12` 这个指纹。

每次运行落盘 `outputs/<模态>/<数据集>/<标签>/seed_N/results.json`,每个方法一行(全部 baseline **与**控制器),各带指标、选后子集指纹 `sel_sha12`、训练顺序指纹 `train_order_sha12`。`<标签>`只有在设置了 `RUN_ID` 环境变量时才会带 `run_id=...` 这一段(`reproduce_full.sh` 内部会设置,上面这种直接调用 `run_single_arm.sh` 的写法不会)。同一份 `METHODS` 列表在**同一等预算协议**下同时跑 baseline 和 OmniSelect,对照按构造公平。

完整复现(四个主臂,种子 0 → 重建表,对应主结果表口径):

```bash
run_scripts/reproduce_full.sh                     # 或:SEEDS="0 1 2" run_scripts/reproduce_full.sh 做旧版3-seed复核
```

## 加入你自己的数据集 / 模态

OmniSelect 是一个 benchmark,接入新数据集是一处小而封闭的改动。完整模板见 [`docs/CONTRIBUTING_DATASETS.md`](docs/CONTRIBUTING_DATASETS.md),简言之,(1) 写一个加载器加受控噪声配方,(2) 把**每一个** baseline 作为候选传入控制器(唯一铁律,基线是候选,绝不是只作对照的列),(3) 按标准布局落盘 `results.json`,(4) 在 `docs/dataset_provenance.md` 记录来源与 SHA。提供了脚手架:

```bash
run_scripts/add_dataset.sh my_new_dataset          # 把一个臂 runner 复制成带标注的模板
```

## 仓库结构

```
src/mmdataselect/      系统核心:质量信号、裁决控制器、选择器
baselines/             忠实的 baseline 实现(各含 method/ + run_*.py + README)
run_scripts/           一键入口(单臂、reproduce_cached/full、add_dataset)
scripts/               各模态臂 runner、fetch_data.py、表格生成器、验证器
results_canonical/     论文每个数字背后的小结果(已提交)
experiments/           canonical JSON 账本 + 已验证运行日志 + 划分 ID 清单
data/                  git 内的小型原始集(TEP、DaISy),大集为指针加 SHA
docs/                  复现、数据溯源、baseline 忠实度、系统架构
environment/           固定环境锁(CPU 与 CUDA 12.8 / torch 2.8.0)
tests/                 纯 CPU 测试套件(系统 + baseline 忠实度门)
```

## 结果

OmniSelect 在图像分类、长序列预测、过程故障诊断、表格分类四类模态的 9 个数据集(CIFAR-100、
CIFAR-100N、ETTh1、ETTm1、ETTh2、TEP21、Electricity、DaISy CSTR、DaISy steam generator)上,
相对 11 个标准 baseline(random、coreset、herding、EL2N、GraNd、CCS、
Density、QuaDMix 发表式迁移、DMF 发表式迁移、纯影响力、固定权重融合)全部排名第一或并列第一:
5 个严格第一,4 个并列第一,对全部 11 个 baseline 在全部 9 个数据集上从不被超越。每个方法共享
同一数据池、预算、下游模型与种子,在同一等预算协议下逐一执行,对照按构造公平。完整运行日志
保留在 `results_canonical/`。k-center 与 herding 同属几何核心集、SemDeDup 的簇内去重规则同属
覆盖信号,三者均作为控制器自身候选参与构造,不计入这 11 个对照 baseline,详见
`docs/baseline_fidelity_ledger.md`。

以上 9 行是固定主运行的主结果表。公开资产共包含 **12 个任务**:在 9 个数据集之外,还有
CIFAR-10、ImageNet-100 与五域文本语言建模臂三个追加协议,其运行覆盖逐项列在
[`results_canonical/README.md`](results_canonical/README.md)中,不会悄悄并入上面 9 行的排名口径。

Full/NoSelect(不做选择、使用全部候选数据)参考覆盖 12 个基准任务中的 10 个,另有独立的冻结 CLIP CIFAR-10 实例一并登记。OmniSelect 在五个预测任务上
仅用 30% 数据预算即全部超过 Full,详见
[`results_canonical/FULL_REFERENCE_COMPARISON.md`](results_canonical/FULL_REFERENCE_COMPARISON.md)。

- **Baseline**:能拿到官方实现的用官方实现,否则忠实重实现其发表的选择规则,忠实度分档披露于
  [`docs/baseline_fidelity_ledger.md`](docs/baseline_fidelity_ledger.md)。
- **覆盖**:EL2N/GraNd/CCS 是基于分类误差的剪枝方法,在回归目标(时序 MASE)上没有定义,故在
  时序数据集上记为 `--` 而非强行跑,这是这些方法自身的结构性局限,不是 OmniSelect 覆盖面的
  缺口。OmniSelect 本身覆盖全部模态。
- 逐表逐图复现配方见 [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)。

## 环境

核心导入与缓存表重建只需 `pip install -e .`;运行纯 CPU 测试套件需 `pip install -e ".[dev]"`。核查的 GPU 栈为 Python 3.12、`torch 2.8.0+cu128`、570 系驱动(`environment/pip_freeze_server_vgpu.txt`、`environment/constraints-cu128.txt`)。大型数据与产物以版本化 release 包分发(URL 待发布,见 [`docs/ARTIFACTS_INDEX.md`](docs/ARTIFACTS_INDEX.md))。

## 许可

代码 MIT。数据集许可各异,见 `docs/ARTIFACTS_INDEX.md`,无明确再分发许可的集合仅以下载指针加 SHA 提供。
