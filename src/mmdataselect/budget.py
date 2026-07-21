"""Budget resolution — turn a config budget spec into a concrete keep-count.

Selection is always budget-constrained. A budget is expressed as one of:

* ``fraction`` — keep ``value`` x N records   (default, e.g. 0.5)
* ``records``  — keep exactly ``value`` records
* ``tokens``   — keep as many records as fit under an (approximate) token cap
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence


@dataclass
class Budget:
    kind: str = "fraction"  # "fraction" | "records" | "tokens"
    value: float = 0.5

    @classmethod
    def from_cfg(cls, cfg: Optional[Dict[str, Any]]) -> "Budget":
        b = {}
        if isinstance(cfg, dict):
            b = cfg.get("budget", cfg) or {}
        return cls(kind=str(b.get("kind", "fraction")), value=float(b.get("value", 0.5)))

    def resolve(self, n_total: int, token_counts: Optional[Sequence[int]] = None) -> int:
        """Return the number of records (k) to keep from a pool of ``n_total``."""
        if n_total <= 0:
            return 0
        if self.kind == "records":
            return max(1, min(n_total, int(self.value)))
        if self.kind == "tokens":
            if not token_counts:
                return max(1, int(round(0.5 * n_total)))
            avg = sum(token_counts) / max(1, len(token_counts))
            return max(1, min(n_total, int(self.value / max(1.0, avg))))
        # fraction (default)
        return max(1, min(n_total, int(round(float(self.value) * n_total))))
