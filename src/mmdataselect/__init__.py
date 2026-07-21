"""MMDataSelect — a unified, modality-agnostic data selection system.

Public surface (kept small and stable; runners and baselines depend on it):

    from mmdataselect.datatypes import UnifiedRecord, Modality
    from mmdataselect.budget import Budget
    from mmdataselect.api import select_pool        # high-level one-call API

See ``docs/architecture.md`` for the system/application separation discipline.
"""
from .datatypes import (  # noqa: F401
    DOMAIN_CODE,
    DOMAIN_GENERAL,
    DOMAIN_MATH,
    Modality,
    UnifiedRecord,
)
from .budget import Budget  # noqa: F401

__version__ = "0.1.0"
__all__ = [
    "UnifiedRecord",
    "Modality",
    "Budget",
    "DOMAIN_GENERAL",
    "DOMAIN_MATH",
    "DOMAIN_CODE",
]
