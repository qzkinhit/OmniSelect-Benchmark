"""Text pool and held-out splits for the five-domain track (general, math, code, image, table).

Each domain's held-out records are split in file order. Canonical: the first half is the
reference sample (influence and DSIR target), the next quarter is the controller slice divided
into V1 (con) and V2 (rank) by a seeded id hash, the last quarter is the report split. v2 divides the
held-out records in the totals ``sizes`` (con 400, rank 600, conf 500, report 500 of the 2,000
registered records): per domain the report share is the tail in file order and con, rank and conf
are consecutive slices of the seeded id-hash order of the rest. The reference sample is con.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from omniselect.core.datatypes import UnifiedRecord


def split_controller_records(records: list[UnifiedRecord], seed: int) -> tuple[list, list]:
    """Canonical V1/V2 split of the controller slice: order by sha256('methodv3-text-split-v1|seed|id')."""
    ordered = sorted(
        records,
        key=lambda record: hashlib.sha256(f"methodv3-text-split-v1|{int(seed)}|{record.id}".encode("utf-8")).digest(),
    )
    cut = len(ordered) // 2
    if cut == 0 or cut == len(ordered):
        raise ValueError("controller records cannot form non-empty V1 and V2")
    return ordered[:cut], ordered[cut:]


@dataclass
class TextSplits:
    """Held-out records per split plus the per-domain reference sample."""

    con: list = field(default_factory=list)
    rank: list = field(default_factory=list)
    conf: list = field(default_factory=list)
    report: list = field(default_factory=list)
    reference: list = field(default_factory=list)
    layout: str = "canonical"


def _shares(n: int, sizes: tuple[int, int, int, int]) -> list[int]:
    """Largest-remainder split of n records in proportion to ``sizes``."""
    total = float(sum(sizes))
    raw = [n * s / total for s in sizes]
    base = [int(r) for r in raw]
    order = sorted(range(len(sizes)), key=lambda i: -(raw[i] - base[i]))
    for i in order[: n - sum(base)]:
        base[i] += 1
    return base


def split_heldout(held: list[UnifiedRecord], seed: int, mode: str,
                  sizes: tuple[int, int, int, int] = (400, 600, 500, 500)) -> TextSplits:
    """Canonical or v2 split of the held-out records (see the module docstring)."""
    by_domain: dict[str, list] = {}
    for record in held:
        by_domain.setdefault(record.domain, []).append(record)
    out = TextSplits(layout="canonical" if mode != "three_way" else "v2")
    for domain in sorted(by_domain):
        rs = by_domain[domain]
        if mode != "three_way":
            half = max(1, len(rs) // 2)
            rest = rs[half:]
            q = max(1, len(rest) // 2)
            out.report.extend(rest[q:])
            out.reference.extend(rs[:half])
            v1, v2 = split_controller_records(rest[:q], seed)
            out.con.extend(v1)
            out.rank.extend(v2)
            continue
        n_con, n_rank, n_conf, n_report = _shares(len(rs), sizes)
        head, report = rs[: len(rs) - n_report], rs[len(rs) - n_report:]
        out.report.extend(report)
        ordered = sorted(head, key=lambda r: hashlib.sha256(f"omniselect-text-v2|{int(seed)}|{r.id}".encode()).digest())
        out.con.extend(ordered[:n_con])
        out.rank.extend(ordered[n_con:n_con + n_rank])
        out.conf.extend(ordered[n_con + n_rank:])
    if mode == "three_way":
        out.reference = list(out.con)
    return out


def rank_to_budget(order: list[int], tok_counts: list[int], budget: int) -> list[int]:
    """Prefix of ``order`` whose token count first reaches ``budget``."""
    sel, used = [], 0
    for i in order:
        sel.append(int(i))
        used += tok_counts[int(i)]
        if used >= budget:
            break
    return sel


def stratified_cut(order: list[int], domains: list[str], tok_counts: list[int],
                   dom_budget: dict[str, int]) -> list[int]:
    """Per-domain prefix of ``order`` up to each domain's token budget, domains in sorted order."""
    selected: list[int] = []
    for domain in sorted(dom_budget):
        domain_order = [int(i) for i in order if domains[int(i)] == domain]
        selected.extend(rank_to_budget(domain_order, tok_counts, dom_budget[domain]))
    return selected
