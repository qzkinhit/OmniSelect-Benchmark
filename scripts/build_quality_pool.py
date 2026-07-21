"""Build a *quality-variance* pool so data selection can actually move downstream loss.

Per modality (general / math / code) the training pool is 60% high-quality + 40%
controlled low-quality, so both influence (spots degraded/low-value text) and
redundancy (spots duplicates) have something to do. A clean, high-quality held-out
set per modality is the evaluation target.

Low-quality is injected four ways (each record tagged in meta):
  truncation   - high-quality text cut to 80-150 chars (broken reasoning/code);
  template     - a few records duplicated ~12x with a changed trailing number;
  crossdomain  - general text mislabeled as math/code (distribution mismatch);
  lowtier      - genuinely lower-quality source rows (finemath-3plus for math).

Caches data/processed/qpool_{train,heldout}.jsonl. Sizes via env (small by default
for a fast verify; scale up POOL_HI/HELD_PER for the full overnight run).
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from mmdataselect.datatypes import DOMAIN_CODE, DOMAIN_GENERAL, DOMAIN_MATH, Modality, UnifiedRecord  # noqa: E402
from mmdataselect.utils.io import write_jsonl  # noqa: E402
from mmdataselect.utils.logger import get_logger  # noqa: E402

log = get_logger("build_quality_pool")

HI_PER = int(os.environ.get("POOL_HI", "200"))        # high-quality per modality (train)
HELD_PER = int(os.environ.get("HELD_PER", "80"))      # clean held-out per modality
LOW_FRAC = 0.40                                        # low-quality fraction of the pool
MINC, MAXC = 150, 2000

# Extra modalities beyond math/code: domain is a free string, no datatypes change.
DOMAIN_IMAGE, DOMAIN_TABLE = "image", "table"
# Which modalities to build (extend with image,table to test modality-agnosticism).
MODALITIES = os.environ.get("MODALITIES", "general,math,code").split(",")


def _field(fields):
    return lambda ex: next((ex[f] for f in fields if ex.get(f)), None)


def _caption(ex):  # image-text: join the per-image captions into one record
    s = ex.get("sentences")
    return " ".join(str(x) for x in s) if isinstance(s, (list, tuple)) else s


def _table_row(ex):  # table: serialize one row to text
    return ". ".join(f"{k} is {v}" for k, v in ex.items() if v not in (None, ""))


HI_SRC = {
    DOMAIN_GENERAL: ("HuggingFaceFW/fineweb-edu", None, _field(["text"])),
    DOMAIN_MATH: ("HuggingFaceTB/finemath", "finemath-4plus", _field(["text"])),
    DOMAIN_CODE: ("codeparrot/codeparrot-clean-valid", None, _field(["content"])),
    DOMAIN_IMAGE: ("yerevann/coco-karpathy", None, _caption),
    DOMAIN_TABLE: ("mstz/adult", None, _table_row),
}
LOWTIER_SRC = {DOMAIN_MATH: ("HuggingFaceTB/finemath", "finemath-3plus", _field(["text"]))}


def _stream(path, config, extract, take, skip=0):
    from datasets import load_dataset

    ds = load_dataset(path, name=config, split="train", streaming=True)
    out, seen = [], 0
    for ex in ds:
        try:
            raw = extract(ex)
        except Exception:
            raw = None
        if not raw:
            continue
        t = " ".join(str(raw).split())
        if MINC <= len(t) <= MAXC:
            seen += 1
            if seen <= skip:
                continue
            out.append(t[:MAXC])
        if len(out) >= take:
            break
    return out


def _rec(domain, i, text, quality, noise=""):
    return UnifiedRecord(
        id=f"{domain[:2]}{quality[0]}{i:05d}",
        modality=Modality.TEXT,
        domain=domain,
        text=text,
        meta={"quality": quality, "noise": noise},
    )


def main() -> int:
    train_path = os.path.join(_REPO, "data/processed/qpool_train.jsonl")
    held_path = os.path.join(_REPO, "data/processed/qpool_heldout.jsonl")
    if os.path.exists(train_path) and os.path.exists(held_path):
        log.info("cached qpool exists; delete to rebuild")
        print("QPOOL CACHED")
        return 0

    n_low = int(round(HI_PER * LOW_FRAC / (1 - LOW_FRAC)))  # low count so low/(hi+low)=LOW_FRAC
    train, held = [], []
    general_hi_cache = []

    for domain in MODALITIES:
        if domain not in HI_SRC:
            log.warning("unknown modality %s; skipping", domain)
            continue
        path, cfg, extract = HI_SRC[domain]
        hi = _stream(path, cfg, extract, HI_PER + HELD_PER)
        if not hi:
            log.warning("%s: no high-quality source; skipping modality", domain)
            continue
        held_rows, hi_rows = hi[:HELD_PER], hi[HELD_PER:]
        for i, t in enumerate(held_rows):
            held.append(_rec(domain, i, t, "high", "heldout"))
        for i, t in enumerate(hi_rows):
            train.append(_rec(domain, i, t, "high"))
        if domain == DOMAIN_GENERAL:
            general_hi_cache = hi_rows[:]

        # ---- inject low-quality (four kinds, ~evenly) ----
        per = max(1, n_low // 4)
        # 1) truncation degradation
        for i, t in enumerate(hi_rows[:per]):
            train.append(_rec(domain, 10000 + i, t[: 80 + (i % 70)], "low", "truncation"))
        # 2) template redundancy (few seeds duplicated ~12x with a changed number)
        seeds = hi_rows[per : per + max(1, per // 12)]
        c = 0
        for s_i, seed in enumerate(seeds):
            for r in range(12):
                if c >= per:
                    break
                train.append(_rec(domain, 20000 + c, f"{seed} [v{r}#{s_i}]", "low", "template"))
                c += 1
        # 3) cross-domain noise (general text mislabeled as this modality)
        if domain != DOMAIN_GENERAL and general_hi_cache:
            for i in range(per):
                src = general_hi_cache[i % len(general_hi_cache)]
                train.append(_rec(domain, 30000 + i, src, "low", "crossdomain"))
        # 4) low-tier source (math: finemath-3plus; others: heavy truncation of HI tail)
        if domain in LOWTIER_SRC:
            lp, lc, lextract = LOWTIER_SRC[domain]
            for i, t in enumerate(_stream(lp, lc, lextract, per, skip=HI_PER + HELD_PER)):
                train.append(_rec(domain, 40000 + i, t, "low", "lowtier"))
        else:
            for i, t in enumerate(hi_rows[per : 2 * per]):
                train.append(_rec(domain, 40000 + i, t[:120], "low", "lowtier"))
        log.info("%s: hi=%d low~%d heldout=%d", domain, len(hi_rows), n_low, len(held_rows))

    if not train:
        log.error("empty pool")
        return 3
    write_jsonl((r.to_dict() for r in train), train_path)
    write_jsonl((r.to_dict() for r in held), held_path)
    from collections import Counter

    qd = Counter((r.domain, r.meta["quality"]) for r in train)
    print(f"QPOOL OK | train={len(train)} heldout={len(held)}")
    for (dom, q), n in sorted(qd.items()):
        print(f"   {dom:8} {q:4} : {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
