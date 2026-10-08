"""OmniSelect: budgeted data selection adjudicated on a clean validation split.

The package exports the record type, the budget resolver, the configuration, and the
controller entry point. Importing it loads numpy only. Model code is imported by the
tracks under ``tracks/`` when a learner is constructed.
"""
from omniselect.core.datatypes import (  # noqa: F401
    DOMAIN_CODE,
    DOMAIN_GENERAL,
    DOMAIN_MATH,
    Modality,
    UnifiedRecord,
)
from omniselect.core.selection.budget import Budget  # noqa: F401

__version__ = "0.2.0"
__all__ = [
    "UnifiedRecord",
    "Modality",
    "Budget",
    "DOMAIN_GENERAL",
    "DOMAIN_MATH",
    "DOMAIN_CODE",
]
