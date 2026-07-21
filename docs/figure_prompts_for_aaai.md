# AAAI 两张关键图的作图提示词(2026-07-19,数据全部来自 canonical,不得改动数字)

---

## 图 1(引言 teaser 图):用真实数据可视化"信号翻转+灾难性失效"与我们的解法

### 给作图 AI 的完整提示词(直接复制)

请为一篇 AAAI 双栏论文绘制引言首图(Figure 1),矢量输出(SVG 与 PDF),画布宽 7.0 英寸、高 2.6 英寸(通栏双栏宽),白底,全部文字用无衬线字体(Helvetica/Arial),字号 7-9pt,配色使用色盲安全托尔调色板:蓝 #4477AA、红 #EE6677、绿 #228833、黄 #CCBB44、灰 #BBBBBB。禁止 3D、禁止渐变、禁止剪贴画、禁止阴影。图的唯一信息是:没有任何固定的数据选择信号能跨模态通用,固定方法会在某个模态灾难性失败,而基于各模态自身验证的裁决从不失败。

布局为左右两个面板:

面板 (a),占左侧 62% 宽,标题 "No fixed signal wins everywhere"。画一个 6 行 4 列的热力图。列为四个模态任务,列标题两行:第一行模态名,第二行指标与方向:
- Image(CIFAR-100, acc ↑)
- Time series(ETTh1, MASE ↓)
- Process(TEP, macro-F1 ↑)
- Tabular(electricity, AUC ↑)
行为六个固定策略:Authenticity-only、Influence-only、Coverage(k-center)、Fixed-weight fusion、Score pruning(EL2N)、Random。
每个格子填该策略在该列内的名次颜色(按列内相对表现归一,绿=最好,黄=中游,红=最差),并在格内印真实数值(以下数字必须逐字使用):
- Image 列:Authenticity 0.426,Influence 0.358,Coverage 0.377,Fixed fusion 0.340,EL2N 0.256,Random 0.381
- Time 列:Authenticity 0.950,Influence 1.218,Coverage 1.083,Fixed fusion 1.019,EL2N 不适用(格子画斜纹并标 N/A),Random 1.056
- Process 列:Authenticity 0.397,Influence 0.364,Coverage 0.317,Fixed fusion 0.338,EL2N 0.084,Random 0.317
- Tabular 列:Authenticity 0.843,Influence 0.816,Coverage 0.874,Fixed fusion 0.860,EL2N 0.234,Random 0.867
注意 Time 列 MASE 越低越好,上色时方向取反。每列的最优格子加一颗小星星。四颗星星分别落在不同的行(Image 列星在 Authenticity,Time 列星在 Authenticity,Process 列星在 Coverage,Tabular 列星在 Coverage),用一条淡灰色折线把四颗星连起来,线旁标注 "best signal flips"。
三个灾难格子加粗红框并用短引线标注:EL2N 在 Process 的 0.084 标 "collapses below random(0.317)";Fixed fusion 在 Image 的 0.340 标 "below random";Authenticity 在 Tabular 的 0.843 标 "below random"。

面板 (b),占右侧 38% 宽,标题 "Per-modality adjudication never fails"。画同样四列的单行条形组:每列两根并排竖条,灰色条是该列固定策略里的最差值(Image 0.256,Time 1.218,Process 0.084,Tabular 0.816),蓝色条是我们的控制器(Image 0.424,Time 0.981,Process 0.408,Tabular 0.873),MASE 列画在反向轴上使"高条=好"在四列视觉一致,轴旁注明 "MASE inverted"。每根蓝条顶端标数值。面板底部一行小字:"Ours: top tier on all four, never worst; certified switch guarantee P(adopt harmful) ≤ 0.05"。
面板 (a) 与 (b) 之间画一个细箭头,箭头上标 "adjudicate per modality"。

图题(印在图下方,7pt):Figure 1: The best selection signal flips across modalities and fixed strategies fail catastrophically somewhere (a), while validation-driven adjudication stays in the top tier everywhere (b). All numbers are three-seed means from our unified equal-budget protocol.

---

## 图 2(方法框架图):四步算法在真实数据实例上的运行

### 给作图 AI 的完整提示词(直接复制)

请为同一篇 AAAI 论文绘制方法总览图(Figure 2),矢量输出(SVG 与 PDF),画布宽 7.0 英寸、高 3.2 英寸,白底无衬线字体 7-9pt,同一托尔调色板(蓝 #4477AA、红 #EE6677、绿 #228833、黄 #CCBB44、灰 #BBBBBB),禁止 3D、渐变、剪贴画、阴影。风格参照 NeurIPS/ICML 方法总览图:从左到右四个编号阶段带,一条真实数据实例贯穿全图,每个阶段内放微型真实数据可视化而不是空泛图标,模块间用细箭头连接,关键决策点高亮。

全图从左到右四个阶段带,每带顶部一个编号圆片和标题:

阶段 ① Signal construction。左端画输入池:一个横向长条代表 ETTh1 候选池(3000 windows),条内用四种彩色小刻度标注注入的 40% 受控噪声(corrupt、duplicate、flat、shuffle 四色小段,60% 灰色为干净),条下标 "pool: 3,000 windows, 40% controlled corruption, per-sample tags"。从池引出三条箭头到三个小卡片,各含一个微型直方图:Authenticity(分布右偏,高分=真实)、Influence(参照模型误差取负)、Coverage(表示空间散布)。卡片下各一行小字说明信号回答的问题:"is it genuine?" "does it help the model?" "is it redundant?"。

阶段 ② Candidate portfolio。三信号箭头汇入一个纵向候选组合面板,面板里排列候选胶囊(chip):auth Top-K、influence Top-K、cluster coverage、fusion grid(一个 3×3 小网格示意权重格点)、executed baselines(herding、k-center、EL2N、CCS、DSIR、SemDeDup 六个小胶囊排两列,外框标注 "baselines run as candidates, never read as results")、random(do-not-select floor,画成灰色)。面板底部小字:"portfolio contains every compared baseline by construction"。

阶段 ③ Construction half V1: rank & freeze。画一个竖直虚线把独立验证集图标切成两半,左半标 V1 (construct),右半标 V2 (adjudicate),两半之间画时间轴与 purge 间隙(时序场景标 "chronological split + purge")。V1 侧放一个微型排行榜(五行,条形长度示意 V1 增益),第一名参照行用蓝色锁图标标 "frozen reference b*"(示例值标 auth),往下一名非参照行用黄色锁标 "frozen challenger family(top-K, preregistered)"。榜下小字:"ranking uses V1 only; the pair is frozen before V2 is opened"。

阶段 ④ Certified adjudication on V2。核心决策卡片:画一个横向数轴,零点竖线,右侧一个绿色区间从 τ 开始;画一个蓝色圆点表示配对改善均值估计,并画出其单边置信下界(LCB)误差线。给两个真实实例(上下两个小数轴):
- 实例一(seed 0):estimate +0.060,radius 0.113,LCB −0.053 落在 τ 左侧,决策标签灰色 "KEEP reference"(旁注 "insufficient evidence")。
- 实例二(seed 2,漂移种子):估计点明显越过 τ,决策标签绿色 "SWITCH (certified)",旁注真实效果 "MASE 1.028 → 0.945"。
卡片右侧列三值决策图例:SWITCH_CERTIFIED(绿)、KEEP_REFERENCE(灰)、ABSTAIN(黄),下方一行小字 "no empirical fallback"。
最右端输出:选中子集图标(900 selected ids)加两个徽章:绿色徽章 "Theorem: P(adopt harmful) ≤ δ = 0.05";蓝色徽章 "metric-specific radii: accuracy EB, AUC U-statistic, time-series conditional Hoeffding; TEP keep-reference"。

四个阶段带之间的箭头上依次标注:scores、candidates、frozen pair、certificate。

图题(7pt):Figure 2: OmniSelect as a four-step algorithm, shown on a real ETTh1 run. Signals build a portfolio that contains every compared baseline as an executed candidate; the construction half V1 ranks and freezes an ordered pair; the adjudication half V2 certifies the paired improvement with a metric-specific one-sided radius and switches only with a certificate, so adopting a harmful challenger has probability at most δ.

---

## 附:两图共同的顶会规范清单(给作图 AI 的附加约束)
1. 所有数字均来自论文 canonical,不得四舍五入到不同位数,不得编造新数;
2. 文字总量控制:图 1 不超过 60 个词,图 2 不超过 90 个词(不含图题);
3. 元素对齐用网格,箭头全部水平或垂直折线,不用曲线;
4. 导出时字体转曲或嵌入,线宽不小于 0.6pt,可缩印为灰度后仍可辨(靠明度差不是纯色相差);
5. 两图与正文术语逐词一致:authenticity/influence/coverage、construction half V1、adjudication half V2、frozen reference、certified switch、keep reference、abstain。
