# Data & artifacts index

Everything the paper consumes, where it comes from, how it is pinned, and what is /
is not in git. Primary source: `docs/dataset_provenance.md` (SHA + revision ledger)
and `docs/provenance_evidence/` (hash lists + license snapshots). The small TEP and
DaISy files currently present under `data/` are checksum-verified research copies;
large image, ETT, OpenML, text-pool and model/cache artifacts are not stored in Git.

## 1. Datasets

All experiments consume a **derived quality-variance pool** generated deterministically
from the pinned raw sources (config + seed recorded per run; pool never claimed to be
original data; noise kinds tagged per record). Recipes: the arm runners'
noise-injection blocks (`NOISE_FRAC=0.40` default, RNG `np.random.default_rng(seed+7)`)
and, for text, `scripts/build_pool_failclosed.py` (fail-closed, no RNG in the builder).

| Dataset | Public source | Pinned revision / version | SHA256 (as consumed) | Labels / target | License (snapshot in `docs/provenance_evidence/licenses/`) |
|---|---|---|---|---|---|
| CIFAR-10 | HF `uoft-cs/cifar10` | HF snapshot `0b2714987fa478483af9968de7c934580d0bb9a2` | per-shard → `docs/provenance_evidence/cifar_shards_sha256.txt` | 10 classes | No formal license, citation-required (cs.toronto.edu snapshot); HF card "unknown" |
| CIFAR-100 | HF `uoft-cs/cifar100` | HF snapshot `aadb3af77e9048adbea6b47c21a81e47dd092ae5` | per-shard → same file | 100 fine labels | same as CIFAR-10 |
| CIFAR-100N | github.com/UCSC-REAL/cifar-10-100n | `CIFAR-100_human.pt` | `bd2d80409754d420292d622e15e25248ba21e37d27429efac49c8da723f44394` | real crowd noisy + clean labels, order-verified vs HF train; measured noise rate 0.402 | license unverified, distributed as download pointer + SHA only, file not in release |
| ETTh1 | github.com/zhouhaoyi/ETDataset | `data/processed/etth1.csv` (cached copy; benign float re-serialization vs upstream, diff evidence recorded) | `5c155a1b14dcafcdc64f76b86c30637b80c5db42f2454d70da341cd7a8305575` | OT forecast, MASE | CC-BY-ND 4.0 (repo LICENSE verbatim) |
| ETTh2 | same | `data/processed/etth2.csv` | `14964a31bcfab7cdb8e5499962525fc58c719dc90c41f9a39ddf80f3def72f52` | OT forecast | CC-BY-ND 4.0 |
| ETTm1 | same | `data/processed/ettm1.csv` | `093cc4efd56a6bf68fb20cc93a2a79a4fbb06f02c8f4e7e5efa5520cc68afce6` | OT forecast | CC-BY-ND 4.0 |
| DaISy CSTR [98-002] | KU Leuven SISTA DaISy | `data/daisy/cstr.dat` | `0ffdda8a1b962d377dc34371be105bd9dcaef7fcca40554e666841efeec6b84d` | concentration forecast | No formal license published; citation-required academic-use statement snapshotted (`daisy_usage_terms.txt`); the small research copy is tracked with its checksum |
| DaISy steamgen [98-003] | same | `data/daisy/steamgen.dat` | `7f1e66031197c9502c7c7583b313b6349b7da678644410f4902d18b743eabc23` | drum pressure (col 5) | same as CSTR |
| TEP | Braatz ML-ready `d00..d21(+_te)` | 44 `.dat` files, per-file list → `docs/provenance_evidence/tep_files_sha256.txt` | sorted-concat digest `f1df2998f0417cf934a4fcc998a5bb86ab1baf6c66779eea20a6ada5bf91a72f` | normal + fault classes 1–21 | license unverified; the small ML-ready research files are tracked with per-file checksums and source citation |
| OpenML electricity | OpenML data_id=151 (`fetch_openml("electricity", version=1)`) | OpenML API cache | ARFF payload `60df61719ff2065ae144ae7788a2a1fca603e6c5bc83c5bd82bb3b63b0d28c74` → `docs/provenance_evidence/openml151_sha256.txt` | binary UP/DOWN (45,312 rows) | OpenML API licence field "Public" |
| Text pool: general | HF `HuggingFaceFW/fineweb-edu` | pinned revision in `data/processed/pool_manifest.json` (`87f09149ef473420...`) | per-shard SHA in the same manifest | quality/noise tags (see below) | ODC-By v1.0 |
| Text pool: math | HF `HuggingFaceTB/finemath` (finemath-4plus) | `e92b25a616738fe9...` | same manifest | " | ODC-By v1.0 |
| Text pool: code | HF `codeparrot/codeparrot-clean-valid` (NOT Stack-Edu) | `4db92d2ec0c1b4c4...` | same manifest | " | per-file original licenses (MIT/Apache/GPL/BSD/…) |
| Text pool: image proxy | HF `yerevann/coco-karpathy` (captions only) | `448fdb1bc7b2d09e...` | same manifest | " | COCO annotations CC-BY 4.0 (upstream terms snapshotted); HF card has no license field |
| Text pool: table proxy | HF `mstz/adult` income CSV | `f90a6fc5f6efb865...` | same manifest | " | HF card "cc" |

Derived text pool: `data/processed/qpool_train.jsonl`: 25,000 records, 5,000 per
domain, 60/40 high/low quality, noise kinds clean/crossdomain/lowtier/template/
truncation; heldout 2,000 (400 per domain, disjoint ID namespace). Pool-level SHAs
(`train_sha256=84e174db...`, `heldout_sha256=1e4a45c9...`) recorded in
`data/processed/pool_manifest.json` (not present in the current checkout; only found in
older server backup snapshots, so these SHAs cannot currently be re-verified locally).

Downstream models (official weights/packages, not reproductions): SmolLM2-135M/360M,
chronos-bolt-tiny, TabPFN-v2 (official `tabpfn` package), CLIP ViT-B/32, DLinear
(standard two-linear-layer structure).

## 2. Split-ID manifests (small, versioned)

Runtime-exported pool/validation/test id lists (+ TEP calibration ids + the generating
RNG recipe string), one file per arm × seed:

```
experiments/split_manifests/{vision,timeseries,tep,tabular}/split_ids_seed{0,1,2}.json   (12 files)
```

Status: **RUNTIME_VERIFIED**: a select-only replay
(`run_id=split-export-20260717`, `PAIRED_RNG=1`, arm configs identical to the pubcore
lanes) recomputed the arrays in-process and its pairing `arrays_sha256` matches
`run_id=pubcore-paired-20260716T1754` exactly for all 12 arm × seed pairs. Per-file
SHA256 and verdicts: `experiments/POST_DATA_LOCK_inventory.json` →
`split_manifest_verification` (see also `docs/dataset_provenance.md`, POST_DATA_LOCK
section). All 12 split manifests are tracked in this repository.

Text lane: **NOT_CAPTURED**: the text run exported no split manifest at runtime and
ids are not re-derived after the fact (no-fabrication rule); its coverage cells stay
PASS_WEAK.

## 3. What is in git vs. what ships as a release archive

In git (small, canonical):

- `experiments/*.json` ledgers: `master_coverage.json`, `canonical_tables.json`,
  `results_matrix.json`, `characteristic_metrics_v4/v5.json`,
  `controller_current_canonical_v5.json`, `ccs_anchor_canonical.json`,
  `selection_manifest_verdicts.json`, `text_controls_stats.json`.
- Verified run logs under `experiments/` (log SHA256 recorded per view in
  `results_matrix.json`).
- `docs/dataset_provenance.md` + `docs/provenance_evidence/` (per-shard/per-file SHA
  lists, upstream byte-diff evidence, license snapshots).
- `environment/pip_freeze_server_vgpu.txt` (environment lock) and
  `migration_manifest_vgpu_20260717T142822.json` (hardware + canonical-file SHAs).
- All code: `src/`, `scripts/`, `baselines/`, `tools/`, `configs/`, `tests/`.

NOT in git (large or third-party; gitignored):

- Large/raw datasets other than the explicitly listed TEP and DaISy files
  (`data/raw/`, `data/processed/`, image/HF/OpenML/text payloads), re-obtainable
  from the pinned sources above and verifiable against the recorded SHA256s. The HF
  hub cache is manifest+revision only (publicly re-downloadable at the pinned
  revisions); it is **not** claimed as a byte backup.
- The large unfiltered `outputs/` tree: only the 42 whitelisted small result JSONs under
  `results_canonical/` are tracked; embedding caches (40 `.npz`, 2.1 GiB, hashed in
  place in the POST_DATA_LOCK inventory) and model files are not.
- The 308-file POST_AUDIT2 **experiment-artifact** backup (logs / results / docs /
  scripts; contains no `data/` files. It is an experiment-artifact backup, not a full
  data-integrity lock).
- `experiments/POST_DATA_LOCK_inventory.json` is tracked; the large files it inventories
  are not automatically redistributed.

These ship as a **versioned release archive with a per-file SHA256 manifest**.
URL: **pending. Not yet published; the archive is not downloadable until a versioned release is cut and its URL is committed here**. Redistribution carve-outs: CIFAR-100N, DaISy, and TEP files
are excluded from the archive (license unverified, download pointer + SHA only);
ETT CSVs are CC-BY-ND 4.0 (redistribution of unmodified copies with attribution).

### Score-cache checkpoints (original-protocol baselines only)

`baselines/deepcore_original/run_original_protocol.py` (the from-scratch ResNet-18
protocol for CIFAR-10/ImageNet-100) computes a shared early-checkpoint scoring pass
once per (dataset, seed, code state) and caches it to
`outputs/score_ckpt_<dataset>_s<seed>_e10_r3_<code_sha12>.npz` (EL2N/GraNd scores +
512-d penultimate features + labels). Every one of the 12 comparison methods on that
protocol reads from this same cache; only the downstream training + evaluation is run
independently per method. This is the standard way EL2N/GraNd/coreset-style methods
are defined (a scoring function shared by all rank/cluster/stratify rules built on top
of it), not a shortcut specific to this repo.

The cache is auto-detected (`if os.path.exists(cache): load else: compute`), so a
fresh clone works with no extra step, just slower (recomputing costs ~9 min for
CIFAR-10, ~21 min for ImageNet-100 per seed). To skip recomputation, drop the matching
`.npz` from the release archive into `outputs/` before running — file is ~99 MB
(CIFAR-10) / ~250 MB (ImageNet-100), too large for git, hence release-archive-only.
The `<code_sha12>` suffix ties a cache file to the exact byte content of
`run_original_protocol.py` it was computed with; a cache from a different code state
is silently ignored (recomputed fresh) rather than mismatched, so this is safe.

## 4. Controlled-noise recipes (pointers)

- Vision: `scripts/run_vision_experiment.py`: flip / near-duplicate / hard-ambiguous.
- Time series: `scripts/run_timeseries_experiment.py`: corrupt / flat / shuffle /
  near-duplicate.
- Process: `scripts/run_tep_experiment.py`: label-flip / sensor-corruption /
  near-duplicate.
- Tabular: `scripts/run_tabular_experiment.py`: label-flip / feature-corruption /
  near-duplicate.
- Text: `scripts/build_pool_failclosed.py`: crossdomain / lowtier / template /
  truncation (POOL_HI=3000, HELD_PER=400, LOW_FRAC=0.4, MINC=150 / MAXC=2000; config
  echoed into `pool_manifest.json`).

All: `NOISE_FRAC=0.40` default, injection RNG `np.random.default_rng(seed + 7)`,
per-record tags; per-arm corruption pools are generated deterministically in-runner
and never persisted (RECIPE_RECORDED; their `arrays_sha256` match the pubcore batch).
