# OmniSelect

[English](README.md)

OmniSelect 在固定预算下为下游模型选择训练子集。它用三个质量信号（真实性、影响力、覆盖度）为数据池中的每条记录打分，并由单信号规则、信号融合、各信号分担不同角色的协同候选以及已发表的选择方法组成候选子集的组合。组合在干净验证数据的构造划分上冻结，排序划分用任务自己的下游学习器选出一个候选，确认划分核验当选的那一对策略。

协同候选先接纳真实性分数或学习型干净度分数最高的记录，再在覆盖度的几何空间中移除离群点和近重复记录，最后在接纳集合内用 herding 或 k-center 用完预算。学习型干净度分数是一个区分干净验证记录与数据池记录的分类器。每次拟合都经过按子集建键的一次训练缓存，因此每个不同的子集只训练一次，运行记录以全量训练当量报告各阶段的开销。

本仓库包含控制器（`omniselect/`）、带保真度标注的对比选择方法（`benchmark/Methods/`）、每个数据族一条赛道（图像、时间序列、工业过程、表格和文本，位于 `tracks/`）、写出完整运行记录的单格驱动程序，以及论文各批实验的队列模板。协议 `v2_2` 是论文采用的协议。

## 安装

使用 Python 3.10 或更高版本。

```bash
git clone https://github.com/qzkinhit/OmniSelect-Benchmark.git
cd OmniSelect-Benchmark
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision          # CPU 或 Apple MPS 版本
python -m pip install -r requirements.txt        # 本包、各赛道依赖库和测试工具
python -m pytest omniselect/tests -m "not slow"
```

在装有 CUDA 12.8 驱动的 Linux 机器上，按论文 GPU 运行所用的固定版本安装。

```bash
python -m pip install -e ".[train,arms,dev]" -c environment/constraints-cu128.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

## 五分钟冒烟运行

Tennessee Eastman 文件已随仓库提供，ETT 序列按固定提交下载。下面的命令在笔记本 CPU 上以协议 `v2_2` 为过程赛道和预测赛道各跑一个小格子，写出运行记录，从存储的逐单元文件重算 macro-F1，并核验每个存储的选择。

```bash
python data/fetch_data.py --only ett tep
TRACK_CELLS="process:tep21 timeseries:ETTh1" bash run.sh smoke
```

不设置 `TRACK_CELLS` 时，`bash run.sh smoke` 还会运行冻结 CLIP 上的 CIFAR-100、使用 TabPFN-v2 的 Electricity 和使用 SmolLM2-135M 的文本赛道。这些格子首次运行时会下载 CIFAR-100、CLIP ViT-B/32、TabPFN-v2 权重和 SmolLM2-135M，文本格子还需要文本数据池（`python data/build_text_pool.py`）。模型或数据集无法加载的格子会给出提示后跳过。下面的命令以完整规模运行一个格子。

```bash
bash run_omniselect/run_cell.sh process tep21 0 v2_2
```

## 复现论文

论文在每个任务上以多个种子评估，同一次运行中的全部策略共享数据池、污染抽样、验证划分和下游模型初始化。`results/paper/seeds_reported.json` 列出每个任务报告的运行，`run_omniselect/queues/` 提供对应命令。

| 队列 | 协议 | 论文中的内容 | `results/paper/records/` 中的记录包 |
|---|---|---|---|
| `main_v2_3.txt` | `v2_2` | 表 3 主实验 | `main_v2_3.tar.gz` |
| `ablation_v2_3.txt` | `v2` | OmniSelect-NC 各行 | `main_v2_3.tar.gz` |
| `main_adapt_v2_2.txt` | `v2_2`，独立方法 | 预测与文本上的 EL2N、GraNd 和 CCS | `main_adapt_v2_3.tar.gz` |
| `scale_up.txt` | `v2_2`、`v2`，独立方法 | 表 3 的 IN-100 与 Text-100k 两列 | `scale_up.tar.gz` |
| `robustness.txt` | `v2_2` | 图 4(e)(f) | `robustness_v2_3f.tar.gz` |
| `signal_drop.txt` | `v2_2` | 图 4(d) 信号移除 | `ablation_v2_3.tar.gz` |
| `validation_size.txt` | `v2_2` | 图 4(g) 验证规模 | `validation_size_v2_3.tar.gz` |

每个稳健性条件及信号移除比较都使用对应任务报告的运行。40% 注入点与无验证噪声点复用主表记录。`paired_units.tar.gz` 保存图 3 使用的 69 个排序划分记录。

```bash
python data/fetch_data.py
python -m run_omniselect.run_queue run_omniselect/queues/main_v2_3.txt --gpus 0,1 --jobs-per-gpu 2 --threads 4
python -m run_omniselect.run_queue run_omniselect/queues/scale_up.txt --gpus 0,1 --jobs-per-gpu 1 --threads 8
```

IN-100 使用 256 px 图像数组和 224 px 裁剪。Text-100k 使用每域 20,000 条记录的 Adult-v2 数据池和 SmolLM2-360M。[data/README.md](data/README.md) 给出数据准备命令。文本与原生图像训练使用 GPU。

```bash
cd results/paper && bash rebuild.sh
```

`rebuild.sh` 重算 13 列主表、11 个基准任务的综合统计、两个规模扩展任务、图 3，以及图 4 的信号移除、稳健性、验证噪声和验证规模结果，并将 16 个输出及合并数据中的绘图字段与 `results/paper/tables/` 比较。[results/README.md](results/README.md) 列出记录与输出。

`omniselect/tests/test_torch_select.py` 中的 torch k-means 测试使用论文 GPU 运行所用的 torch 2.11。[docs/REPRODUCING.md](docs/REPRODUCING.md) 列出协议字段与稳健性选项。

## 运行记录

每个格子写出一个目录，`decision.json` 最后写入。

```
results_and_logs/<batch>/<track>/<dataset>/<learner>/seed_<s>/
├── config.json         解析后的配置、命令行、git sha、包版本、硬件
├── splits.json         数据池、构造、排序、确认和测试划分的 id 及其 sha256，污染标签
├── signals/            每个信号对每条池记录的分数
├── candidates/<name>/  selection.npz、per_unit/{con,rank,conf,test}.npz、scores.json
├── leaderboard.json    按排序效用排列的候选及其在各划分上的效用
├── decision.json       采纳规则、当选候选、审计统计量、预检、筛选轨迹
├── timings.json        各阶段秒数、缓存命中、全量训练当量
├── metrics.json        该赛道主表的各行
└── log.txt
```

逐单元文件保存每个候选在每个划分上的预测、目标和损失，因此新指标无需训练即可计算，每个选择都能重建并按其哈希核验。

```bash
python -m omniselect.store.recompute results_and_logs/smoke --metric balanced_accuracy --split test
python -m omniselect.store.replay results_and_logs/smoke
```

[docs/RUN_RECORD.md](docs/RUN_RECORD.md) 逐项说明每个文件和字段。

## 添加策略或数据族

一个策略是一个目录 `benchmark/Methods/<Name>/`，其中的 `method.py` 用 `omniselect.core.portfolio.registry.register` 注册函数 `fn(ctx, k) -> list[int]`，README 写明来源、保留的规则、适配方式和保真度标注。该函数从 `SelectionContext` 读取数据池、标签和信号，返回 k 个池索引。在 `benchmark/Methods/__init__.py` 中导入它，并把它加入 `omniselect/core/portfolio/membership.py` 的成员表，它就成为组合成员。

一个数据族是 `tracks/<family>/` 中 `Track` 的子类，配有保存常量的 `TrackConfig` 子类，并实现钩子 `load`（数据、污染和划分）、`signals`（三个信号和策略输入）和 `learner`（一个提供 `fit(subset, stage)` 与 `score(model, split)`、返回逐单元输出的对象）。在 `tracks/common/experiment.py` 的 `TRACKS` 中注册它。[docs/REPRODUCING.md](docs/REPRODUCING.md) 列出可选钩子。

## 仓库结构

| 目录 | 内容 |
|---|---|
| [`omniselect/`](omniselect/README.md) | 控制器，包括配置、信号、选择、裁决、采纳规则、组合和运行记录存储 |
| `tracks/` | 每个数据族一条赛道，以及单格驱动程序 `tracks/common/experiment.py` |
| `benchmark/` | 每个选择方法一个目录、独立运行脚本、数据集加载器 |
| [`data/`](data/README.md) | 数据集说明、获取脚本和文本数据池构建脚本 |
| `run_omniselect/` | shell 入口、队列模板、队列运行器、CSV 汇总 |
| `tools/` | 采纳规则回放、代理排序、运行记录审计 |
| `docs/` | 设计、代码导读、复现、运行记录格式、数据集来源 |
| [`results/`](results/README.md) | 论文运行记录、表格和重建脚本 |

[docs/CODE_MAP.md](docs/CODE_MAP.md) 给出阅读代码的顺序，[docs/DESIGN.md](docs/DESIGN.md) 说明协议的各项选择。

## 引用

```bibtex
@misc{omniselect2026,
  title  = {{OmniSelect}: Task-Driven Adjudication of Data Selection Strategies for
            Modality-Specific Foundation Models},
  author = {Qian, Zekai and Ding, Xiaoou and Zhou, Muyun and Wang, Hongzhi and Wang, Chen},
  year   = {2026},
  note   = {Submitted to PVLDB}
}
```

## 许可证

代码以 [MIT 许可证](LICENSE)发布。数据集、模型权重以及此处重新实现的已发表方法遵循各自条款，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
