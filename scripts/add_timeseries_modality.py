"""Append a 6th modality (timeseries) to the quality-variance pool, mirroring the
60%-high / 40%-controlled-noise design of build_quality_pool.py.

Timeseries source: autogluon/chronos_datasets kernel_synth (the Chronos pretraining
corpus), each example a numeric series serialized to text. Noise injected four ways
(truncation / template / crossdomain / lowtier), same as the other modalities, so the
selection signals have something to act on. Appends to qpool_{train,heldout}.jsonl and
removes the influence cache so it is recomputed for the extended pool.

    python scripts/add_timeseries_modality.py
"""
from __future__ import annotations

import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from mmdataselect.datatypes import Modality, UnifiedRecord  # noqa: E402
from mmdataselect.utils.io import read_jsonl, write_jsonl  # noqa: E402

DOMAIN_TS = "timeseries"
HI_PER, HELD_PER = 200, 80
LOW_FRAC = 0.40
MINC, MAXC = 150, 2000
WIN = 60  # values per serialized window


def _series(ex):
    s = ex.get("target")
    if not isinstance(s, (list, tuple)) or len(s) < 24:
        return None
    vals = [x for x in s if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(vals) < 24:
        return None
    return "time series: " + ", ".join(f"{float(x):.2f}" for x in vals[:WIN])


def _stream_ts(take):
    from datasets import load_dataset

    ds = load_dataset("autogluon/chronos_datasets", name="training_corpus_kernel_synth_1m",
                      split="train", streaming=True, trust_remote_code=True)
    out = []
    for ex in ds:
        t = _series(ex)
        if t and MINC <= len(t) <= MAXC:
            out.append(t[:MAXC])
        if len(out) >= take:
            break
    return out


def _rec(i, text, quality, noise=""):
    return UnifiedRecord(id=f"ts{quality[0]}{i:05d}", modality=Modality.TEXT, domain=DOMAIN_TS,
                         text=text, meta={"quality": quality, "noise": noise})


def main() -> int:
    train_path = os.path.join(_REPO, "data/processed/qpool_train.jsonl")
    held_path = os.path.join(_REPO, "data/processed/qpool_heldout.jsonl")
    train = [UnifiedRecord.from_dict(d) for d in read_jsonl(train_path)]
    held = [UnifiedRecord.from_dict(d) for d in read_jsonl(held_path)]
    if any(r.domain == DOMAIN_TS for r in train):
        print("timeseries already in pool; nothing to do")
        return 0
    # general text (for the cross-domain noise type), reused from the existing pool
    gen_txt = [r.text for r in train if r.domain == "general"][:64]

    hi = _stream_ts(HI_PER + HELD_PER)
    if len(hi) < HI_PER + HELD_PER:
        print(f"only got {len(hi)} timeseries records; aborting")
        return 3
    held_rows, hi_rows = hi[:HELD_PER], hi[HELD_PER:]
    for i, t in enumerate(held_rows):
        held.append(_rec(i, t, "high", "heldout"))
    for i, t in enumerate(hi_rows):
        train.append(_rec(i, t, "high"))

    n_low = int(round(HI_PER * LOW_FRAC / (1 - LOW_FRAC)))
    per = max(1, n_low // 4)
    # 1) truncation
    for i, t in enumerate(hi_rows[:per]):
        train.append(_rec(10000 + i, t[: 80 + (i % 70)], "low", "truncation"))
    # 2) template (few seeds duplicated with a changed tail)
    seeds = hi_rows[per: per + max(1, per // 12)]
    c = 0
    for s_i, seed in enumerate(seeds):
        for r in range(12):
            if c >= per:
                break
            train.append(_rec(20000 + c, f"{seed} [v{r}#{s_i}]", "low", "template"))
            c += 1
    # 3) crossdomain (general text mislabeled as timeseries)
    for i in range(per):
        if gen_txt:
            train.append(_rec(30000 + i, gen_txt[i % len(gen_txt)], "low", "crossdomain"))
    # 4) lowtier (heavy truncation of the HI tail)
    for i, t in enumerate(hi_rows[per: 2 * per]):
        train.append(_rec(40000 + i, t[:120], "low", "lowtier"))

    write_jsonl((r.to_dict() for r in train), train_path)
    write_jsonl((r.to_dict() for r in held), held_path)
    cache = os.path.join(_REPO, "data/processed/qpool_influence.npz")
    if os.path.exists(cache):
        os.remove(cache)
        print("removed stale influence cache (will recompute for 6-modality pool)")
    n_ts = sum(1 for r in train if r.domain == DOMAIN_TS)
    print(f"OK appended timeseries: train+={n_ts} heldout+={HELD_PER} | total train={len(train)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
