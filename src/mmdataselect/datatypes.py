"""The modality-agnostic data record — the single coupling between modalities.

Every source (vision-language / math / code / general text) is standardized into a
:class:`UnifiedRecord` by ``tools/standardize`` *before* any signal is computed, so
that all downstream signals and selectors operate on one common representation and
never branch on the original modality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

# Domains kept as plain strings so new domains need no code change.
DOMAIN_GENERAL = "general"
DOMAIN_MATH = "math"
DOMAIN_CODE = "code"


class Modality(str, Enum):
    """Source modality of a record. ``str`` mix-in keeps JSON round-trips trivial."""

    TEXT = "text"
    IMAGE_TEXT = "image_text"
    AUDIO_TEXT = "audio_text"


@dataclass
class UnifiedRecord:
    """One standardized data point.

    Attributes
    ----------
    id : str
        Stable, globally unique identifier (carried through manifests).
    modality : Modality
        Source modality; signals stay modality-agnostic, this is metadata only.
    domain : str
        Coarse content domain (``DOMAIN_GENERAL`` / ``DOMAIN_MATH`` / ``DOMAIN_CODE`` / ...).
    text : str
        The modality's textualized content (caption+OCR for image-text, code text,
        problem+solution for math). All signals are computed on this field.
    meta : dict
        Free-form provenance / auxiliary fields (source path, token count, image ref).
    """

    id: str
    modality: Modality
    domain: str
    text: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        mod = self.modality.value if isinstance(self.modality, Modality) else str(self.modality)
        return {"id": self.id, "modality": mod, "domain": self.domain, "text": self.text, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UnifiedRecord":
        mod = d.get("modality", Modality.TEXT)
        if not isinstance(mod, Modality):
            mod = Modality(mod)
        return cls(
            id=str(d["id"]),
            modality=mod,
            domain=d.get("domain", DOMAIN_GENERAL),
            text=d.get("text", "") or "",
            meta=d.get("meta", {}) or {},
        )
