# Processed inputs

Files read by the tracks and ignored by Git:

- `etth1.csv`, `etth2.csv`, `ettm1.csv`: ETDataset at commit `1d16c8f4f943`, re-saved by pandas
  (`python data/fetch_data.py --only ett` checks the raw and the saved sha256).
- `qpool_train.jsonl`, `qpool_heldout.jsonl`, `pool_manifest.json`: the five-domain text pool
  (`python data/build_text_pool.py`, verified by `python data/fetch_data.py --verify-only --only text`).
- caches written by the tracks: `vision_*.npz` (CLIP features and labels), `native_scores_*.npz`
  (score-run EL2N, GraNd and penultimate features), `text_influence_pplq_*.npz` and
  `text_transfer_*.npz` (text signals).
