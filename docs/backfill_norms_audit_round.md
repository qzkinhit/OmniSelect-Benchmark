# 回填措辞规范(审计轮,2026-07-15 定版,回填双稿前必读)

服务器周结果写进论文时,以下口径为硬约束,与数字同权重。

## 1. 测试集主张(全稿统一)

只写三件事:验证集上的构造保证(命题2 的原始表述,含 2ε 泛化误差界)、多 seed
mean±std 实证、每条臂的真实名次。禁止出现"test 上保证不弱于全部基线"
"strictly wins every modality"一类表述。Chronos 与表格臂中控制器非最优的行
(ETTh1 adapt 0.8765 vs DMF 0.8733、cstr adapt vs full、steamgen seed0、表格
dsdm 0.8758 vs adapt 0.8734)如实入表并按论文一贯口径分析。

## 2. TEP 检测指标

FDR/FAR 只以校准形式报告:写"在独立验证集上取目标 FAR=5% 的校准工作点",
同时报告该工作点的**实际测试 FAR**(逐 seed),再给 FDR@FAR{1,5,10}% 与 AUROC。
不得写成"每个 seed 均满足 FAR≤5%"。未校准的 22 类 argmax FDR/FAR(冒烟里的
0.98/0.97)不入论文。

## 3. 文本池的域定义(防夸大)

服务器规模池 = **五个文本化域**:general=fineweb-edu、math=finemath-4plus、
code=**codeparrot-clean-valid(不是 Stack-Edu,全稿不得写 Stack-Edu)**、
image=coco-karpathy 的 caption 文本、table=adult 行序列化文本。
image/table 两域是文本化代理域,**不得**当作独立的图像基础模型或表格基础模型
证据引用;它们只服务于文本臂的跨域混合池。

## 4. 消融命名

G 批只能称 **drop-channel reduced-portfolio ablation**(剔通道的同时组合缩为
双通道网格+random,搜索空间同步变化);若要称纯通道消融,须补 3 通道
core-only 对照(同候选类型/同搜索预算/同 random 参照)后方可。

## 5. 复现规模的如实标注

ImageNet-100@112px/40ep = **缩规模定性链路**,不得称原文完整协议;
全量 CIFAR-10(45k+160ep)才可称 Data-Diet/CCS 原协议复现,且 GraNd 为
线性头梯度范数代理(err×‖φ‖,已按审计修正),多初始化平均 SCORE_RUNS=3。
CIFAR 只有 seed0 时不得写"3-seed"。

## 6. 冒烟数字

任何 smoke 数字不作效果声明,只证链路。

## 7. ImageNet-100 结果的表述边界(Codex 19:12)

seed0 的结果只能写"缩规模 ImageNet-100 上的积极初步结果,控制器在该 seed 的验证
裁决选中 CCS"。不得写"定理已兑现""论文级结论""忠实复现官方协议"。若最终只有
两个 seed,标注 reduced-scale qualitative validation,不进正式三 seed 主表。

## 8. 终止条件(关机前必须全部满足)

D_VALIDATED_OK + E_VALIDATED_OK + ARTIFACTS_VALIDATED_OK +
CONTROLLER_REVALIDATED_OK + BACKUP_VERIFIED_OK 五标记齐全,且 outputs/
experiments/批脚本/环境快照/manifest 连同完整 SHA256 回传本地并双端核验一致。
