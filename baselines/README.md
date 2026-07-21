# Baselines 与数据集清单(出处 · 为何选 · 忠实度认证)

> 本目录是全部外部方法的实现地。每个 baseline 自带独立 `method/`(纯选择逻辑)+ 对称
> runner,产出与 `run_mmdataselect/run_select.py` 相同的 manifest 格式,共用同一套
> train/eval。**改任何 baseline 实现前,必须先过机制测试**
> `pytest tests/test_baseline_fidelity.py`(8 项,它抓出过两处不忠实实现)。
> 逐 baseline 认证证据与服务器精确复现协议见 `experiments/BASELINE_FIDELITY_AUDIT.md`。

## 一、忠实度认证等级(每个 baseline 都标了级)

| 等级 | 含义 | 证据 |
|---|---|---|
| **T1 官方代码对齐** | 与官方实现同池同参照对照出一致性数字 | `experiments/dsir_official_fidelity.log` |
| **T2 原文数据集定性复现** | 在原论文自己的数据集上复现其核心结论 | `experiments/fidelity_clean_cifar.log`、`tab_aicl_original_datasets.log` |
| **T3 机制测试** | 受控数据上验证方法的定义性机制,pytest 固化防回归 | `tests/test_baseline_fidelity.py` |

逐数字复现原文报告值(如 Data-Diet 的 ResNet-18 精确数)绑定原训练协议,本地不可达,
已写成服务器协议(AUDIT 文档末节):Data-Diet ResNet18 原配方、CCS 高剪枝对表、
DeepCore 官方库直跑、DSIR 官方 `data-selection` 包换入、Tab-AICL 冷启动 AULC。

## 二、baseline 总表(runner 里的 METHODS 名 → 出处 → 为何选 → 认证)

| METHODS 名 | 方法 | 出处(会/年) | 为何选 | 认证 |
|---|---|---|---|---|
| `random` / `full` | 随机 / 全量 | 惯例 | 必备下界 / 上界参考 | - |
| `auth_only` | 纯真实性 | QuRating(ICML24)、Ultra-FineWeb(2025)、FineWeb-Edu 的质量过滤范式 | 三信号通道之一,代表质量过滤路线 | T3 |
| `influence_only` / `if_mates` | 纯影响力 | LESS(ICML24)、PDS(2025)、MATES(NeurIPS24) | 通道之一,代表 model-aware 路线 | T3 |
| `coreset` | 纯覆盖(kmeans) | SemDeDup / Entropy-Law 去冗余思想 | 通道之一,代表覆盖路线 | T3 |
| `herding` | Herding | Welling, **ICML 2009** | 几何 coreset 鼻祖,DeepCore 标配 | T2+T3 |
| `kcenter` | k-center greedy | Sener & Savarese, **ICLR 2018** | coreset / 主动学习经典 | T2+T3 |
| `el2n` | EL2N | Paul et al., **NeurIPS 2021**(Data Diet) | 分数剪枝最常被引,审稿人必问 | T2+T3 |
| `grand` | GraNd(期望梯度范数) | 同上 NeurIPS 2021 | 同上;早期版弱与公开复现文献(arXiv 2303.14753)一致,如实引注 | T2+T3 |
| `ccs` | CCS | Zheng et al., **ICLR 2023** | 高剪枝率当前 SOTA,原文结论已在其 CIFAR-10 主场复现 | T2+T3 |
| `semdedup` | SemDeDup | Abbas et al., 2023(arXiv/workshop,引用注明) | 语义去重代表,LLM 管线标配;初版被机制测试抓出不忠实,已按原文簇内阈值规则重写 | T3 |
| `density` | Density 采样 | Sachdeva et al., 2024(arXiv, Google) | 覆盖采样代表;已改按密度反比概率采样(忠实) | T3 |
| `quadmix` | QuaDMix 式联合 | 2024(arXiv,无官方代码,注明按原文目标复现) | 质量×多样性联合,与本文论点最贴身 | T3 |
| `dmf` | 动态融合 DMF | Yang et al., 2025 | 验证反馈乘性调权,最强融合对手 | T3 |
| `mmdataselect` | 固定权重融合 | 自建对照 | 消融:权重写死会怎样 | - |
| `dsir` | DSIR | Xie et al., **NeurIPS 2023** | 文本分布匹配标准基线;与官方包秩相关 0.855、top-50% 选择重叠 85.5% | **T1** |
| `zip` | Entropy-Law / ZIP | Yin et al., 2024(arXiv,注明无会议) | 压缩比选择,官方 repo `USTC-StarTeam/ZIP` | 部分 T3 |
| `quality_ppl` | 质量过滤消融 | Ultra-FineWeb 式(注意其官方是 fastText 分类器) | 文本臂质量过滤对照 | - |
| `tabpfn_coreset/margin/hybrid` | Tab-AICL 三规则 | 2026(arXiv) | 表格上给 TabPFN 选上下文的直接对口前作;**单次选择适配版**,原文迭代 AULC 协议下的相对序未复现(协议差异,论文已披露),精确协议归服务器 | T2(差异已声明) |
| `mmds_adapt` | 投票式控制器(本文) | - | 组合含上述全部本体,构造上不弱于每个 | - |

**`auth_only`、`dmf`、`kcenter`、`semdedup` 排除于论文正文"vs 11 baseline"对照表之外。** 这四者是
控制器自身候选组合的成员(信号消融分量 / 融合消融分量 / 覆盖类候选),仍在此目录正常实现、正常被
`mmds_adapt` 在构造上纳入并调用,只是不作为独立对照行出现在最终结果表里——这是按定义
排除,不是按输赢排除。完整规则见 `experiments/canonical_tables_seed0.json` 的
`_meta.internal_only_excluded` 字段。

## 三、数据集总表(出处 → 为何选)

| 数据集 | 跑法 | 出处 | 为何选 |
|---|---|---|---|
| CIFAR-100 / CIFAR-10 | `VIS_DATASET=uoft-cs/cifar100` 或 `cifar10` | Krizhevsky 2009 | 图像经典,且**正是 EL2N/GraNd/CCS/herding 原文数据集**,在对手主场对比 |
| ETTh1 / ETTh2 / ETTm1 | `TS_DATASET=ETTh1|ETTh2|ETTm1` | Informer(**AAAI 2021 最佳论文**)引入的 LSF 标准 | 时序预测默认坐标系 |
| TEP 田纳西伊斯曼 | `scripts/run_tep_experiment.py` | Downs & Vogel 1993 | 化工故障诊断金标准(导师点名) |
| DaISy CSTR / steamgen | `TS_DATASET=daisy_cstr|daisy_steamgen`(数据 `data/daisy/`) | KU Leuven SISTA 识别库 98-002/98-003 | 导师点名;控制器双双严格最优 |
| OpenML electricity | `TAB_DATASET=electricity` | OpenML 常用 | 表格主臂,TabPFN 生态常用 |
| ionosphere / phoneme | `TAB_DATASET=ionosphere|phoneme` | UCI/OpenML 经典 | **Tab-AICL 原文 20 数据集中的两个**,为验证它接入 |
| 文本池(general/math/code) | `scripts/run_experiment.py` | FineWeb-Edu、FineMath、Stack-Edu(HF 官方策展) | 用户点名的数学与代码数据集来源 |
| 服务器榜单级 | 见 `SERVER_PROMPT.md` 第 5 节 | DataComp(NeurIPS23 D&B)、DCLM(NeurIPS24 D&B)、GIFT-Eval、ImageNet、OpenML-CC18、GSM8K/HumanEval | 提交制/固定协议,环境无关可比 |

## 四、怎么跑

```bash
./runall.sh setup    # 环境 + pytest + 数据自检
./runall.sh smoke    # 五条链路各 1-seed,~15 分钟
./runall.sh local    # 全部本地 3-seed + 自动汇总
# 单跑:METHODS="random,el2n,grand,mmds_adapt" SEED=0 .venv/bin/python scripts/run_vision_experiment.py
```

结果日志在 `experiments/*_3seed.log`,汇总 `scripts/summarize_runs.py`。
论文里每个印出的数字都能在这些日志里找到出处。

## 五、目录契约(加新 baseline 必守)

```
baselines/<name>/
  method/            纯选择逻辑(独立代码,零 IO)
  run_<name>.py      对称 runner:读统一 jsonl -> 选择 -> 写 manifest
  README.md          一句话:是什么 + 引用
```

runner 写 `outputs/<exp_id>_<name>/manifests/{manifest.json,selected.jsonl}`,字段
`{n_total, n_selected, selected_ids, method, experiment_id}`,manifest 契约是唯一耦合。

三条铁律:(1) 新 baseline 必须作为候选策略传进
`AdaptiveController.select(..., extra_strategies=[...])`,只当对照列会毁掉"构造上不弱于
每个基线"的核心贡献;(2) 配 smoke test 进 `tests/`,机制类方法加 T3 测试进
`tests/test_baseline_fidelity.py`;(3) 配置不得偷偷偏向或削弱任何一方(如 DSIR 的
target 必须用所有方法都能拿到的同一份干净参照)。
