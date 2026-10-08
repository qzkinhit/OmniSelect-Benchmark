"""UnifiedRecord: the record type shared by signals, selectors and the text track.

A record has a stable ``id``, a ``modality`` tag, a ``domain`` string and a ``text`` field on
which the text signals are computed, plus free-form ``meta``. Non-text tracks create records with
empty text and use them only for their count and domain. ``to_dict`` and ``from_dict`` convert to
and from JSON rows.
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
        Source modality. Signals do not read it.
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
