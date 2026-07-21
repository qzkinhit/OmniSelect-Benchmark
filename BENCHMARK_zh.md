<!-- LANG -->
[English](BENCHMARK.md) | **简体中文**

# OmniSelect-Benchmark:协议说明

本文档给出本仓库实现的基准协议:每个模态臂评测什么任务、质量波动池如何构造、每个方法
(基线或控制器)必须遵守的等预算与候选集/参考集/留出集隔离规则、运行产物的格式,以及
种子约定。快速开始与当前结果表见 [README.md](README_zh.md),新增数据集/模态/baseline
的具体做法见 [docs/CONTRIBUTING_DATASETS.md](docs/CONTRIBUTING_DATASETS.md)(英文)。

## 这个基准测什么

固定训练预算下,应该从一个带噪声的数据池里保留哪个子集。本基准在图像、时序、表格、
过程故障诊断四类模态上用同一套评测台架回答这个问题,使一个选择方法的跨模态表现可以
直接比较,它是在每个模态都保持领先梯队,还是因为只依赖某个模态才有效的信号而在其他
模态崩溃。

九个主表数据集/噪声配置已给出可复现的证据(`results_canonical/`):CIFAR-100、
CIFAR-100N(真实人工标注噪声)、ETTh1、ETTm1、ETTh2、DaISy CSTR、DaISy 蒸汽发生器、
田纳西伊斯曼过程(TEP21)、OpenML Electricity。CIFAR-10(原版 DeepCore 协议)作为独立的
baseline 忠实度对照,不计入这九个。

## 质量波动池

纯高质量数据的池子测不出选择的价值,任何方法看起来都一样好。本基准的每个池子按约六成
高质量加四成受控、逐条打标的低质量注入构造,按模态至少覆盖四类退化(如图像的标签翻转
/近重复/跨域/输入损坏,时序与过程数据的截断损坏/重复/平坦信号/乱序)。标签从不提供给
任何选择方法,只在选择之后用于统计 `clean_pct`,即某方法选中子集里真正高质量样本的占比。

## 等预算协议

同一数据集与种子下,被比较的每个方法都从**同一个池**里选**同样的预算** `k`,用**同一个
下游模型**与**同一个训练测试划分**,这样指标的差异只能归因于选了哪些样本,而不是选了
多少样本或训练条件不同。预算与池/测试集规模按数据集与模态固定
(见 `scripts/run_*_experiment.py` 与 `docs/per_modality_experiment_plan.md`),不按方法
调整。

## 候选集/参考集/留出集隔离

每次运行都强制三路划分,防止选择信号泄漏进本应度量它的那个数字:

- **候选池**:每个方法从中选择的数据。
- **参考/验证集**:驱动信号计算,以及(对控制器而言)驱动候选策略之间的裁决,最终指标
  从不使用这部分。
- **留出测试集**:只在选择与训练全部完成之后打分一次。

如果某方法用自己的测试集表现来挑选选择规则,或者在后续将要选择的那批样本上拟合信号,
就违反了这一隔离,不算有效的参赛条目。

## baseline 唯一铁律

**每个被比较的 baseline 都作为控制器可以裁决的候选被执行,而不是脱离控制器单独跑出一份
`results.json` 再拿数字来对照。** 具体做法是,baseline 通过同一次调用里的
`extra_strategies=[(name, fn), ...]` 传入控制器(见
`docs/CONTRIBUTING_DATASETS.md`),这样控制器"验证集上不弱于组合内任一策略"的保证是
被实测出来的,不是假定的。若某 baseline 发表的协议与本基准的一次性等预算设定在结构上
不同(如迭代式主动学习),会如实披露而不是强行套进本设定,逐 baseline 的忠实度分档
(T1 官方代码对齐/T2 原文数据集复现/T3 机制测试)与诚实边界见
`docs/baseline_fidelity_ledger.md`。

## 运行产物格式

每次运行按 `(模态, 数据集, 运行标签, 种子)` 写一份 `results.json`
(`outputs/<模态>/<数据集>/<标签>/seed_N/results.json`,`<标签>`只有在设置了 `RUN_ID`
时才带 `run_id=...`)。字段格式见 [docs/results_schema.json](docs/results_schema.json)。
每个方法行的关键字段:`metric`(任务指标,方向按模态而定)、`n`(选中样本数)、
`clean_pct`、`sel_sha12`/`train_order_sha12`(选择集与训练顺序的指纹,用于精确复现核验)。
`results_canonical/` 保存了论文里每个数字直接读自的那一小部分已提交结果,不需要重新训练
(`run_scripts/reproduce_cached.sh`)。

## 种子

`results_canonical/` 里做过完整 3 seed 的部分覆盖种子 `{0, 1, 2}`,单种子的证据按惯例只给
种子 0。`run_scripts/run_single_arm.sh` 与 `run_scripts/reproduce_full.sh` 默认
`SEED=0`/`SEEDS="0"`,显式传 `SEEDS="0 1 2"` 才复现 3 seed 复核。新增的证据或展示性内容
(如 `experiments/selreplay_evidence/`)只展示种子 0 作为代表,以保持仓库精简,这不改变
实际测过什么,完整 3 seed 扫描的逐种子指纹保存在 `results_canonical/` 里。

## 数据集来源与许可

每个数据集的官方来源、锁定版本/revision、SHA256、许可,以及"原样使用,仅含已披露的噪声
注入"声明记录在 [docs/dataset_provenance.md](docs/dataset_provenance.md) 与
[docs/ARTIFACTS_INDEX.md](docs/ARTIFACTS_INDEX.md)。没有确认可再分发许可的数据集
(CIFAR-10/100、ImageNet-100)不以原始字节提交,运行 `python scripts/fetch_data.py` 从
官方来源拉取并做 SHA 校验。TEP 与 DaISy 属公开可用/研究可复用来源且体量小,以原始文件
直接提交在 `data/` 下,逐文件 SHA256 见 `docs/provenance_evidence/`。

## 新增 baseline 或数据集

完整模板见 [docs/CONTRIBUTING_DATASETS.md](docs/CONTRIBUTING_DATASETS.md)
(`run_scripts/add_dataset.sh` 会脚手架出一个新的模态臂)。简言之:写一个用同一套受控噪声
配方的加载器,把包括新加入的这个在内的每个 baseline 都作为候选传入控制器,按标准布局
落盘 `results.json`,并记录数据集来源。

## 引用

如果使用本基准,请引用 OmniSelect 论文(论文公开后补充引用信息,当前预印本状态见
[README.md](README_zh.md))。
