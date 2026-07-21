"""Modality-agnostic utility signals.

Every signal maps a list of :class:`~mmdataselect.datatypes.UnifiedRecord` to a
per-record value in ``[0, 1]`` (higher = more valuable to keep), so signals are
directly comparable and fusible across vision-language / math / code.
"""
from .base import Signal, minmax  # noqa: F401
from .redundancy import RedundancySignal, hashed_features, set_redundancy  # noqa: F401
from .influence import InfluenceSignal  # noqa: F401
from .authenticity import AuthenticitySignal  # noqa: F401

__all__ = [
    "Signal",
    "minmax",
    "RedundancySignal",
    "InfluenceSignal",
    "AuthenticitySignal",
    "hashed_features",
    "set_redundancy",
]
