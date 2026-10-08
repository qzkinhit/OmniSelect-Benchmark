"""Text authenticity: a weighted sum of completeness, non-degeneracy and structural validity.

Completeness ramps from 0 at 80 characters to 1 at 400. Non-degeneracy averages the unique-word
and unique-bigram ratios. Structural validity checks that code parses (``ast``) and contains code
keywords, that math contains math tokens, and that general text has plausible word lengths and
punctuation. Weights default to (0.35, 0.30, 0.35). No model is used.
"""
from __future__ import annotations

import ast
import re
from typing import Sequence

import numpy as np

from omniselect.core.datatypes import DOMAIN_CODE, DOMAIN_MATH, UnifiedRecord
from omniselect.core.signals.base import Signal

_WORD = re.compile(r"\w+")
_MATH = re.compile(r"[0-9]|[+\-*/=^%<>]|\\[a-zA-Z]+|sqrt|theorem|prove|equation|solve|integral|derivative")
_CODE_KW = ("def ", "return", "import ", "class ", "for ", "while ", "lambda", "self.", "()", "{", "}", "::", "=>")


def _completeness(t: str) -> float:
    # soft ramp: <80 chars -> 0, >=400 chars -> 1 (truncated text scores low)
    return float(np.clip((len(t) - 80) / 320.0, 0.0, 1.0))


def _non_degeneracy(t: str) -> float:
    words = _WORD.findall(t.lower())
    if len(words) < 4:
        return 0.3
    uniq = len(set(words)) / len(words)
    bigrams = list(zip(words, words[1:]))
    ubg = len(set(bigrams)) / max(1, len(bigrams))
    return float(np.clip(0.5 * uniq + 0.5 * ubg, 0.0, 1.0))


def _structural(t: str, domain: str) -> float:
    if domain == DOMAIN_CODE:
        kw = 1.0 if any(k in t for k in _CODE_KW) else 0.0
        try:
            ast.parse(t)
            parses = 1.0
        except Exception:
            parses = 0.0
        return float(0.5 * parses + 0.5 * kw)
    if domain == DOMAIN_MATH:
        return float(np.clip(len(_MATH.findall(t)) / 8.0, 0.0, 1.0))
    # general: sentence-like (reasonable word length + has punctuation)
    words = _WORD.findall(t)
    if not words:
        return 0.0
    avg = sum(len(w) for w in words) / len(words)
    base = 1.0 if 2.5 <= avg <= 9.0 else 0.4
    return float(base * (1.0 if any(p in t for p in ".!?,;") else 0.6))


class AuthenticitySignal(Signal):
    """Model-free text authenticity with weights (completeness, non-degeneracy, structural validity)."""

    name = "authenticity"

    def __init__(self, weights=(0.35, 0.30, 0.35)):
        self.weights = weights

    def score(self, records: Sequence[UnifiedRecord]) -> np.ndarray:
        """One authenticity score in [0, 1] per record."""
        wc, wd, ws = self.weights
        out = np.zeros(len(records))
        for i, r in enumerate(records):
            t = r.text or ""
            out[i] = wc * _completeness(t) + wd * _non_degeneracy(t) + ws * _structural(t, r.domain)
        return out
