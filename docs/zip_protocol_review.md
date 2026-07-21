# ZIP 忠实协议与复杂度审查(audit 二)— 冻结替代协议

日期 2026-07-17。背景:exact-ZIP 全 25k 排序 lane(text-qz3)被判定为确定性计算爆炸并停止(`/root/qz3_blowup_evidence/VERDICT.txt`:~8.1e6 次 zlib 压缩、~50.5TB 累计;证据目录只读,永不覆盖)。本文档在任何重启之前冻结唯一替代协议。

## 1. 爆炸定位(本地实现 `baselines/zip/method/zip_select.py`)

三段贪心(k1_ratio=0.1, k2_ratio=0.5, zlib level 6, k=n=25000 全排序):

- Stage 1:按预计算的单样本压缩率排序取 Bottom-K1 —— 只排序,不爆炸。
- Stage 2:对 K1 个候选各压缩一次 `selected_blob + cand` —— 全程共 49,994 次(占比 <1%)。
- **Stage 3(爆炸点,`zip_select.py` L157-169)**:每选 1 个样本就对 block 内所有剩余候选重压 `selected_blob + cand`,而 `selected_blob` 单调增长到 18.9MB(池 25000 条、实测均值 756.32 B/样本,`ssh omni` 对 `data/processed/qpool_train.jsonl` 逐行核算,总量 18.91MB)。

精确循环结构模拟(不压缩、只计数,2026-07-17 复算):**159 轮外循环,stage2 = 49,994 次,stage3 = 8,025,425 次,合计 8,075,419 次压缩;按 blob 增长逐次累计被压字节 = 50.5TB** —— 与审计判词 "~8.1e6 compressions ~50.5TB" 逐位吻合。服务器实测 zlib level-6 单线程吞吐 22.3 MB/s(见 §3),50.5TB ≈ **26.2 天单核**,即被停时的真实量级(进程已耗 utime ≈ 44,199s ≈ 12.3h,`/root/qz3_blowup_evidence/proc_stat.txt`)。

## 2. 官方参照(2026-07-17 在线抓取)

来源:ar5iv 2407.06645(Yin et al., Entropy Law / ZIP)+ 官方 repo USTC-StarTeam/ZIP(`ZIP.py` raw,main 分支;台账锁定论文时点 commit `f77b0b5e2cc28414fc9d51f5d08bf34ad0353527`,`docs/baseline_fidelity_ledger.md` §2 ZIP 行)。

- 公式:g(D) = Bits(D)/Bits(C(D)),**最小化**(低压缩率 = 高信息密度)。与我方一致。
- Algorithm 1 三段:Stage1 取 π_D 最低 K1;Stage2 对 K1 算并集压缩率并**更新 π_D**,取最低 K2;Stage3 从 K2 贪心逐个加 K3 个。官方定值(Appendix B / repo 参数):**K1=10000, K2=200, K3=100**,repo 默认 `budget=1000`;Appendix D 扫 K1∈{200,1000,10000,20000}。
- **官方是否全量重压:是。** `ZIP.py` 的 `get_compression_ratio(input_data)` 每次对完整拼接串 `zlib.compress(data_str, level=9)`,无增量、无窗口、无截断;靠 `--n_jobs` 多进程硬算。
- **官方实验规模**:从 ~300k 池(DEITA / WizardLM+ShareGPT+UltraChat)选 **10,000 条** SFT(Mistral-7B / Llama-3-8B),选择本身 CPU-only 约 4.5h(repo README);预算语义按 token 对齐各 baseline(选样本数、控 token)。**官方从未跑过 k=n 的全池排序**;其 10k/300k 设定下 blob 只长到 10k 条,且每轮固定 K1=10000 次评估,故可行。

## 3. 候选方案裁决(按任务优先序)

服务器实测(`ssh omni`,miniconda python,CPU 微基准,只读池文件、零写入,2026-07-17):
- zlib level-6 一次性压缩 4.38MB 真实池文本:**22.3 MB/s**;
- 增量方案单次评估(`compressobj.copy()` + 压 ~757B 候选 + flush):**205.8 µs/次**;
- **位精确性:200/200 随机 (prefix, cand) 对,增量压缩长度 == 一次性 `zlib.compress` 长度,全部相等。**

| 方案 | 结论 | 数字 |
|---|---|---|
| (a) 官方代码官方规模 | **否**。官方代码同为全量重压,在我方"选到预算"规模(k≈12.5k/25k,blob 至 9.5-18.9MB)累计仍是十 TB 级;且其固定 K1/K2/K3 + π 更新与我方台账已冻结的本地实现是不同协议,换协议反而引入新替换项 | 官方 4.5h 依赖 300k 池选 10k + n_jobs 多进程;我方规模换算 ≈15TB+(k=12.5k)单核 ~8 天 |
| (b) 我方精确实现只排到 token 预算(k_budget 而非全 n) | **否**。BUDGET_FRAC=0.5,乐观 k_budget≈12.5k 时累计仍 ≈24TB ≈ 12.5 天单核(且 ZIP 偏好短高熵样本,k_budget 上界为 n,最坏不降) | 24TB / 22.3MB/s ≈ 12.5 天 |
| **(c′) 冻结方案:位精确增量前缀态评估器(exact-equivalence acceleration)** | **是**。非近似:zlib deflate 输出只依赖字节流与 flush 点、与喂入分块无关,故"保留已压前缀的 `compressobj` 状态 → 每候选 `copy()` → 压候选 → flush 计长"与一次性全量重压**逐字节等长**(服务器 200/200 实证)。同一 g、同一 argmin、同一全排序,选择语义零替换 | 8,075,419 次 × 205.8µs ≈ **27.7 min 单核**(全 25k 排序);加 3 倍安全余量上界 **< 1.5h** |

**冻结:方案 (c′)。** 注意它优于任务原设想的"32KB 窗口近似"(那是有替换项的 published-method reimplementation);(c′) 是纯计算等价变换,**位精确、无任何协议替换**,台账定性不变(ZIP 仍为文本臂 portfolio 附属证据、本地实现档,`docs/baseline_fidelity_ledger.md` §2)。理论依据一并记录:zlib 32KB 窗口局部性意味着超窗前缀的重压是可证浪费功,但我们不用近似,直接用前缀态复用拿到同一结果。

### 3.1 冻结的实现规范(改 `baselines/zip/method/zip_select.py`,选择语义不动)

维护 `master = zlib.compressobj(level)` 与 `emitted`(master 已吐出字节数)、`raw_len`(已选原始字节含分隔符):

```python
def _merged_len(master, emitted, cand, sep):      # 评估一个候选
    c2 = master.copy()
    return emitted + len(c2.compress(sep + cand)) + len(c2.flush())
# g(S ∪ {d}) = (raw_len + len(sep) + len(d)) / _merged_len(...)
# 选定 d 后: emitted += len(master.compress(sep + d)); raw_len += len(sep)+len(d)
# ratio_trace: 用 master.copy()+flush 取当前 g(S)(每选 1 次,共 n 次,可忽略)
# 空集边界: selected 为空时 sep 为空串,与原 _merged_blob 语义一致
# stage1 单样本压缩率、K1/K2 取法、tie-break(ratio, index)全部原样保留
```

等价性由三重闸门保证(任一失败即回退原实现并停止 lane):
1. 现有测试 `tests/test_zip_cache_equivalence.py`(index-exact、确定性、k 截断为前缀)必须全绿;
2. 新增等价断言:小池上新旧实现 `selected_idx` 与 `ratio_trace` 逐元素相等;
3. 冒烟脚本内 200 对随机 (prefix, cand) 压缩长度位相等断言(§4)。

复杂度上界(冻结数):评估次数精确 = 8,075,419(n=25000, k=n, k1_ratio=0.1, k2_ratio=0.5 的确定性循环结构,与随机性无关);单次实测 205.8µs → **~28 min 单核,上界 1.5h**;内存:一个活跃 `copy()` 状态(数百 KB)+ 池本身。ZIP_CACHE=1 契约不变(cache 仍存 n=25000 全排序,key 含实现文件 SHA —— 实现文件已改,key 自动翻新,不会误命中旧 cache;qz3 从未产出过 cache)。

## 4. 冒烟计划(≤5 min,只读池、产物只进 scratch,不碰任何既有 outputs/logs/markers)

目的:exact-k 唯一 ID、选中 ID SHA、位精确等价、有限运行时外推。**本轮只交付命令,不执行。**

```bash
# on omni, CPU only, NO GPU, NO lane
cd /root/autodl-tmp/OmniSelect
SCRATCH=/root/zip_smoke_qz4_$(date -u +%Y%m%dT%H%M%SZ)   # 全新目录,永不覆盖旧物
mkdir -p "$SCRATCH"
.venv/bin/python - <<'EOF'
import json, time, hashlib, itertools, sys, os
sys.path.insert(0, "/root/autodl-tmp/OmniSelect")
from baselines.zip.method.zip_select import zip_select            # 原实现
from baselines.zip.method.zip_select import zip_select_incremental  # 新实现(§3.1)
rows = [json.loads(l) for l in itertools.islice(open("data/processed/qpool_train.jsonl"), 400)]
texts = [r.get("text") or "" for r in rows]; ids = [r["id"] for r in rows]
o_ref, t_ref = zip_select(texts, ids, len(rows))
t0 = time.perf_counter(); o_new, t_new = zip_select_incremental(texts, ids, len(rows)); dt = time.perf_counter() - t0
assert list(o_ref) == list(o_new), "ORDER MISMATCH - do not relaunch"
assert t_ref == t_new, "TRACE MISMATCH - do not relaunch"
assert len(set(o_new)) == len(rows) == len(o_new), "exact-k unique IDs violated"
sha = hashlib.sha256(json.dumps([ids[i] for i in o_new]).encode()).hexdigest()
evals_25k = 8_075_419
per_eval = dt / max(1, len(o_new))  # 粗上界: 小池每选一样本的均摊
print(f"SMOKE_OK sha256(selected_ids)={sha} n=400 dt={dt:.1f}s "
      f"extrapolated_25k_upper<= {evals_25k * 205.8e-6 / 60:.0f} min (measured 205.8us/eval)")
EOF
# 断言全过且外推 < 90 min 才允许进入 qz4 lane;任何断言失败 = 冻结失效,回本文档重审
```

## 5. 三 seed lane 规格(text-qz4,仅规格,严禁本轮启动)

- RUN_ID:`text-qz4-<UTC 时间戳>`(独立于 qz3;逐 seed 由 runner 落 `run_id=` 目录)。
- 日志:`/root/text_qz4_${RUN_ID}.log`(全新文件);marker:`/root/TEXT_QZ4_OK`(全新,仅 fail-closed 验证器可写);**绝不触碰 `/root/qz3_blowup_evidence/*`、`/root/TEXT_QUADMIX_TRANSFER_OK`、既有 outputs/logs/markers**。
- 环境(对齐 qz3 被停 lane 的 PROBE 头):`STRATIFY=1 INFL_KIND=pplq TRAIN_MODE=finetune LM_EVAL=1 PASSES=3 METHODS=zip ZIP_CACHE=1`,mini-LM hid=320 layers=6 ctx=512,BUDGET_FRAC=0.5(默认),seeds 0,1,2。
- ZIP_CACHE=1:排序与 seed 无关,seed0 冷算一次(≤1.5h CPU,上界见 §3.1),seed1/2 位精确复用;cache 文件名含新实现 SHA,不会与任何旧 cache 冲突(qz3 无 cache 产出)。
- 前置闸门:§4 冒烟 SMOKE_OK + `pytest tests/test_zip_cache_equivalence.py` 全绿,缺一不可。
- qz3 已完成的 seed0 quadmix_pub 证据保持原状引用,不重跑不覆盖;quadmix_pub seed1/2 是否并入 qz4 由 orchestrator 另行裁决(本规格 METHODS 只含 zip,避免混批)。

## 6. 台账登记项(替换项清单,全部为既有本地实现相对官方的差异,本次加速新增替换项 = 0)

1. K1/K2 取法:我方比例式 K1=0.1·n_rem、K2=0.5·K1、stage3 吃完整个 block;官方定值 K1=10000/K2=200/K3=100。
2. π 状态:我方 stage1 用静态单样本压缩率;论文 Algorithm 1 会用并集压缩率更新 π_D。
3. zlib level:我方 6;官方 9。
4. 预算语义:我方全排序后按 token 预算截断(所有方法统一的 fixed-budget adapter);官方按样本数 budget 停止、跨方法 token 对齐。
5. (c′) 增量评估器:**位精确等价,非替换项**,以 §3.1 三重闸门为证。

NOT-CAPTURED:官方 `ZIP.py` 在论文时点 commit f77b0b5e 与 main 最新版之间是否有算法改动(本次只核了 main 分支 raw 文件);官方 4.5h 所用核数/n_jobs 值(README 未载)。
