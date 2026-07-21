# Dataset provenance ledger

Corrected claim discipline. The accurate statement is three-part:
1. The ORIGINAL downloaded artifacts are kept immutable and their SHA256 recorded below.
2. Experiments consume a SEPARATELY GENERATED, fully tagged, seeded derived corruption pool
   (quality-variance pool). The pool is NOT "original data" and is never claimed to be; its
   generation is deterministic (config + seed recorded) and does not modify the source files.
3. Candidate / clean-reference / held-out splits are disjoint; equal budget; >=3 seeds.

Verification status legend: [SHA-OK] exact bytes hashed on server; [REV-OK] pinned upstream
revision recorded; [PENDING] evidence not yet captured (listed here, to be closed before
submission).

## Vision

| Dataset | Source | Pinned version evidence | Raw-artifact SHA256 | Labels | Status |
|---|---|---|---|---|---|
| CIFAR-10 | HF `uoft-cs/cifar10` (Krizhevsky 2009) | HF snapshot revision `0b2714987fa478483af9968de7c934580d0bb9a2` | parquet shards inside pinned snapshot | 10 classes | [REV-OK]; per-shard [SHA-OK] → `provenance_evidence/cifar_shards_sha256.txt` |
| CIFAR-10 (CCS anchor lane, 2026-07-17) | torchvision `datasets.CIFAR10` via 官方 CCS loader(AutoDL 公共集拷入 `/root/autodl-tmp/ccs_official/data/cifar10/cifar-10-python.tar.gz`) | tar.gz md5 `c58f30108f718f92721af3b95e74349a`(= torchvision 官方校验值,md5 verified) | 官方 loader 原样读取 | 10 classes | 仅用于 official released-implementation anchor lane(`docs/ccs_anchor_protocol.md`、`experiments/ccs_anchor_canonical.json`),与主台架 HF snapshot 独立 |
| CIFAR-100 | HF `uoft-cs/cifar100` | HF snapshot revision `aadb3af77e9048adbea6b47c21a81e47dd092ae5` | parquet shards inside pinned snapshot | 100 fine labels | [REV-OK]; per-shard [SHA-OK] → `provenance_evidence/cifar_shards_sha256.txt` |
| CIFAR-100N | github.com/UCSC-REAL/cifar-10-100n (Wei et al. ICLR 2022) | `CIFAR-100_human.pt` | `bd2d80409754d420292d622e15e25248ba21e37d27429efac49c8da723f44394` | real crowd noisy_label + clean_label; order verified aligned to HF train | [SHA-OK] |

## Time series / process

| Dataset | Source | File as consumed | SHA256 | Target | Status |
|---|---|---|---|---|---|
| ETTh1 | github.com/zhouhaoyi/ETDataset | data/processed/etth1.csv (cached copy of upstream CSV) | `5c155a1b14dcafcdc64f76b86c30637b80c5db42f2454d70da341cd7a8305575` | OT forecast, MASE | [SHA-OK]; upstream byte-diff [EVIDENCE-OK] → `provenance_evidence/ett_upstream_bytediff.txt` (MISMATCH: cached copy is a float re-serialization of upstream, same rows/values, ~1e-17 diff) |
| ETTh2 | same | data/processed/etth2.csv | `14964a31bcfab7cdb8e5499962525fc58c719dc90c41f9a39ddf80f3def72f52` | OT forecast | [SHA-OK]; byte-diff [EVIDENCE-OK] (same benign re-serialization mismatch) |
| ETTm1 | same | data/processed/ettm1.csv | `093cc4efd56a6bf68fb20cc93a2a79a4fbb06f02c8f4e7e5efa5520cc68afce6` | OT forecast | [SHA-OK]; byte-diff [EVIDENCE-OK] (same benign re-serialization mismatch) |
| DaISy CSTR 98-002 | KU Leuven SISTA DaISy | data/daisy/cstr.dat | `0ffdda8a1b962d377dc34371be105bd9dcaef7fcca40554e666841efeec6b84d` | concentration forecast | [SHA-OK] |
| DaISy steamgen 98-003 | same | data/daisy/steamgen.dat | `7f1e66031197c9502c7c7583b313b6349b7da678644410f4902d18b743eabc23` | drum pressure (col 5) | [SHA-OK] |
| TEP | Braatz ML-ready d00..d21(+_te) | data/tep/*.dat, sorted concat | `f1df2998f0417cf934a4fcc998a5bb86ab1baf6c66779eea20a6ada5bf91a72f` (concat digest); per-file list [SHA-OK] → `provenance_evidence/tep_files_sha256.txt` (44 files) | normal + faults 1-21 | [SHA-OK concat + per-file] |
| OpenML electricity | OpenML data_id=151, `fetch_openml("electricity", version=1)` | sklearn openml cache | data/151.gz `ba8aae2c819f97561ea3cc12281cd2af8a32dfa53d9e11e0e92da52a2b9dbe51`, features/151.gz `7b97ed83bb9144c6fdfd6f388ba32c496ce22a1608f158f3f423e77da0491ff8` | binary UP/DOWN | [SHA-OK api cache]; ARFF payload [SHA-OK] electricity.arff.gz `60df61719ff2065ae144ae7788a2a1fca603e6c5bc83c5bd82bb3b63b0d28c74` → `provenance_evidence/openml151_sha256.txt` |

## Text pool (five textualized domains): pool_manifest.json (complete, strongest evidence)

Pinned HF revisions + per-shard SHA256 + train/heldout SHA all machine-recorded in
`data/processed/pool_manifest.json` (not present in the current checkout; only found in
older `server_backup_*/omni/data/processed/pool_manifest.json` snapshots, so the SHAs below
cannot currently be re-verified locally): train_sha256=`84e174db...`, heldout_sha256=`1e4a45c9...`, 6 shards.

| Domain | Source | Pinned revision (recorded in manifest) |
|---|---|---|
| general | HF `HuggingFaceFW/fineweb-edu` | `87f09149ef473420...` |
| math | HF `HuggingFaceTB/finemath` (finemath-4plus) | `e92b25a616738fe9...` |
| code | HF `codeparrot/codeparrot-clean-valid` (NOT Stack-Edu; every doc must say codeparrot) | `4db92d2ec0c1b4c4...` |
| image proxy | HF `yerevann/coco-karpathy` captions | `448fdb1bc7b2d09e...` |
| table proxy | HF `mstz/adult` income CSV | `f90a6fc5f6efb865...` |

Models: SmolLM2-135M/360M, chronos-bolt-tiny, TabPFN-v2 (official pkg), CLIP ViT-B/32.

## Non-numeric completion (2026-07-16, audit item 二.4): counts / labels / preprocessing / noise config

All numbers below were measured on server `omni` (`/root/autodl-tmp/OmniSelect`) on 2026-07-16
with `/root/miniconda3/bin/python` one-liners (jsonl row counters, `wc -l`, ARFF parse, npz/pt
loads); nothing is quoted from papers.

### Text pool (qpool)

- `data/processed/qpool_train.jsonl`: 25,000 records; exactly 5,000 per domain
  (general / math / code / image / table). `meta.quality`: high=15,000, low=10,000 (60/40).
  `meta.noise` injection kinds: clean ""=15,000; crossdomain=2,500; lowtier=2,500;
  template=2,500; truncation=2,500.
- `data/processed/qpool_heldout.jsonl`: 2,000 records; 400 per domain; all `quality=high`,
  all `noise="heldout"` (clean held-out namespace `{dom}hH{i}`, disjoint from train IDs).
- Label definition: `meta.quality` in {high, low} is the pool-level quality tier;
  `meta.noise` names the injection kind (empty = untouched source text).
- Preprocessing: rows drawn from the pinned-revision shards in `pool_manifest.json`;
  char-length filter MINC=150 / MAXC=2000.
- Noise/seed pointer: builder `scripts/build_pool_failclosed.py` with config
  POOL_HI=3000, HELD_PER=400, LOW_FRAC=0.4 (recorded in `pool_manifest.json` → `config`).
  The builder is fail-closed and deterministic (fixed slicing, no RNG call in the builder;
  shortfall = hard error, no fallback); low-kind IDs use fixed offsets 10k/20k/30k/40k.

### Vision

- CIFAR-10 as consumed: `data/cifar10_np/cifar10.npz`: Xtr (50000,32,32,3) uint8,
  ytr (50000,), Xte (10000,32,32,3), yte (10000,); 10 classes.
- CIFAR-100N: `data/cifar_n/CIFAR-100_human.pt`: 50,000 aligned label pairs
  (clean_label / noisy_label + coarse variants); measured real human noise rate = 0.402;
  100 fine classes. Label definition: `flip` tag where noisy != clean, else `high`.
- Preprocessing: CLIP ViT-B/32 image embeddings cached to
  `data/processed/vision_*_clip-vit-base-patch32_p{P}v{V}t{T}_s{seed}.npz`
  (keys Xp/Xval/Xt, 512-d float32; e.g. p4000v2000t2000 at seeds 0/1/2).
- Noise-instance config: `scripts/run_vision_experiment.py`: NOISE_FRAC=0.40 default;
  synthetic kinds flip / near-duplicate / hard-ambiguous, each tagged;
  injection RNG `np.random.default_rng(seed + 7)`, SEED from env (runs use seeds 0-2).

### Time series / process

- ETTh1 / ETTh2: 17,420 data rows each (17,421 lines incl. header); ETTm1: 69,680 data rows.
  Columns: `date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT`; forecasting target = OT.
- DaISy: `cstr.dat` 7,500 rows; `steamgen.dat` 9,600 rows (targets as in table above).
- TEP: 44 files. `d00.dat` is the upstream-transposed normal-operation train file
  (52 lines x 500 fields); `d01..d21.dat` are 480 rows x 52 vars (faulty train);
  all `d*_te.dat` are 960 rows (test). Labels: normal + fault classes 1-21 (by file).
- Noise-instance config: `scripts/run_timeseries_experiment.py`: NOISE_FRAC=0.40 default;
  kinds corrupt / flat / shuffle / near-duplicate (one quarter of low each), tagged;
  injection RNG `np.random.default_rng(seed + 7)`, SEED from env.

### Tabular

- OpenML electricity (data_id 151): ARFF payload = 45,312 rows, 8 feature attributes +
  binary `class {UP,DOWN}`; class distribution UP=19,237 / DOWN=26,075.
- Noise-instance config: `scripts/run_tabular_experiment.py`: NOISE_FRAC=0.40 default;
  kinds label-flip / feature-corruption / near-duplicate, tagged;
  injection RNG `np.random.default_rng(seed + 7)`, SEED from env (default 0; runs use 0-2).

## Gaps still open (each must close before submission or be disclosed)

- [CLOSED 2026-07-17, POST_DATA_LOCK] per-seed train/val/test split ID manifests + SHA:
  the four arm runners now carry an env-gated `SPLIT_EXPORT_DIR` dump inside their seeded
  split logic; a select-only replay (run_id=`split-export-20260717`, METHODS=random,
  PAIRED_RNG=1, CPU-only, arm configs identical to the pubcore lanes) emitted
  `experiments/split_manifests/{vision,timeseries,tep,tabular}/split_ids_seed{0,1,2}.json`
  (full pool/val/test ids + TEP calibration ids + generating rng recipe string).
  Verification: the replay recomputed the pool/validation/test arrays in-process and its
  pairing `arrays_sha256` matches run_id=`pubcore-paired-20260716T1754` EXACTLY for all
  12 arm x seed pairs → all 12 manifests are **RUNTIME_VERIFIED** (verdicts + per-file
  SHA recorded in `experiments/POST_DATA_LOCK_inventory.json` →
  `split_manifest_verification`). Text lane: **NOT_CAPTURED**. QZ4 emitted no split
  manifest at runtime; per the no-fabrication rule these ids are not re-derived after
  the fact (its ledger cells stay PASS_WEAK).
- [EVIDENCE-OK] license text/evidence snapshots captured 2026-07-16 into
  `provenance_evidence/licenses/`. CIFAR (cs.toronto.edu: no license, citation-required; HF card
  "unknown"); ETDataset (repo LICENSE verbatim = CC-BY-ND 4.0); OpenML electricity (API licence
  field "Public"); HF cards: fineweb-edu = odc-by v1.0, finemath = ODC-By v1.0, codeparrot-clean-valid
  = per-file original licenses (MIT/Apache/GPL/BSD/…), mstz/adult = "cc".
  CLOSED 2026-07-16: COCO upstream terms snapshotted verbatim →
  `provenance_evidence/licenses/coco_termsofuse.txt` ("The annotations in this dataset …
  are licensed under a Creative Commons Attribution 4.0 License"; images remain under
  Flickr ToU (we use captions only). yerevann/coco-karpathy HF card itself still has no
  license field; upstream CC-BY 4.0 governs the annotations.
  CLOSED 2026-07-16: DaISy usage terms snapshotted →
  `provenance_evidence/licenses/daisy_usage_terms.txt` (citation-required academic-use
  statement from daisydata.html, incl. the recommended De Moor citation template; no
  formal license text is published).
- [CLOSED 2026-07-17, POST_DATA_LOCK, with an explicit manifest-only carve-out]
  raw/derived/cache inventory built server-side: `experiments/POST_DATA_LOCK_inventory.json`.
  (a) Raw sources (CIFAR HF parquet shards, ETT csvs, DaISy dats, TEP 44 dats, OpenML-151
  sklearn cache, 6 text-pool HF shards) recorded with path/size/SHA256/pinned revision;
  every small local file was byte-recomputed on the current server and matches the recorded
  evidence SHAs (0 mismatches); a 5-file byte-recomputation sample on the large HF-side
  artifacts (cifar10 train, cifar100 train, finemath-4plus shard, coco.parquet, adult csv)
  all match. (b) Derived pools: qpool train/heldout SHA re-verified against
  `pool_manifest.json` (match); per-arm corruption pools are generated deterministically
  in-runner and never persisted → **NOT_CAPTURED as files, RECIPE_RECORDED** (config +
  seed recipe + split-id manifests make them reproducible; the generated pools'
  `arrays_sha256` match the pubcore batch). (c) Embedding/caches: all 40 `*.npz` under
  `data/` and `outputs/` hashed IN PLACE (2.1 GiB, not copied). HF hub cache =
  **manifest+revision only**: publicly re-downloadable at the pinned revisions, bytes not
  locally backed up on a second machine. NOT claimed as a dual-end byte backup.
  (d) zip order cache `data/processed/zip_order_cache_2d363c3edb698abe.json` hashed.
  Dual-end backup of the irreplaceable SMALL artifacts (12 split manifests + inventory +
  12 split-export results.json + the two new tool scripts, 27 files): local
  `post_data_lock_backup_20260717T1430/`, per-file SHA verified both ends 27/27 →
  marker `POST_DATA_LOCK_VERIFIED_OK` written (server `experiments/` + local backup dir).
- Noise/corruption pool generation: config + seed recorded per run; pool is derived data, tagged
  per record (flip/corrupt/dup/hard/truncation/template/crossdomain/lowtier).


## 完整性表述(2026-07-17 核查 1215 第 5 条)

当前 POST_AUDIT2(308 文件,双端逐文件 SHA 一致)封存的是**实验 artifact**(日志/结果/
文档/脚本),manifest 中 data/ 计数为 0,因此只能声明 **experiment artifact backup
closed**,不能声明 full data-integrity lock closed。

2026-07-17 更新(POST_DATA_LOCK,audit 1345 四.3):上文两项 PENDING 已按其新状态关闭:
(1) 四臂 x seed 0/1/2 的 split-id manifest 已由 runner 运行期真实导出并经 pairing
`arrays_sha256` 与 pubcore 批次逐一精确比对,全部 **RUNTIME_VERIFIED**;text lane
**NOT_CAPTURED**(QZ4 运行期未导出,不事后重推)。(2) raw/derived/cache 清单落地为
`experiments/POST_DATA_LOCK_inventory.json`,小文件全部在当前服务器逐字节重算并与既有
证据 SHA 一致,大 HF 工件抽样 5 个逐字节重算一致;embedding caches 已就地全量 SHA(40 个
npz,2.1 GiB,不复制字节);HF cache 仅 manifest+revision(公开可重下,**不声明**双端字节
备份)。不可替代的小工件(27 文件)已双端逐文件 SHA 备份到本地
`post_data_lock_backup_20260717T1430/`,marker `POST_DATA_LOCK_VERIFIED_OK` 双端写入。
表述与实际情况一致:derived corruption pool 是由公开原始数据按 config+seed 确定性生成、逐条
打标的派生数据,**不声称**直接在未修改的原始数据上训练;QZ4 缺失的 split/noise/init SHA
不事后伪造,其两格维持 PASS_WEAK。
