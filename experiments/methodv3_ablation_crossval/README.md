# 独立协议下的通道丢弃消融交叉复现

这批产物来自另一套独立代码路径(theory-aligned controller 分支,已废弃的两设置证书门
设计,与本仓库当前的单一控制器不同),协议为 xmodal paired protocol,93 组独立运行
(5 数据集族 × 7 种消融 × 3 seed,87 OK + 6 诚实 FAIL_CLOSED)。

保留原因:其中的通道丢弃消融(drop_signal_authenticity/influence/redundancy)在 TEP21
上的结果与本仓库主协议的通道丢弃消融(`experiments/channel_drop_ablation_3seed.log`)
方向和幅度一致(丢真实性影响最大,其次影响力,覆盖最小),构成独立代码路径的交叉复现,
是论文消融小节的证据来源之一。其余内容(证书门 on/off、回退机制等消融)对应已废弃的
两设置设计,不用于当前论文叙述,仅作为可追溯的原始产物保留。

6 个 FAIL_CLOSED 全部是时序协议下丢真实性导致守卫锚点无法解析,控制器直接拒绝输出
(失效即安全机制的设计,不是意外故障)。
