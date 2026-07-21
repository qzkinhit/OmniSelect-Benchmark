"""IF / MATES baseline — influence-driven Top-K data selection.

References:
* Koh & Liang, "Understanding Black-box Predictions via Influence Functions",
  ICML 2017.
* Yu et al., "MATES: Model-Aware Data Selection ... Data Influence Models",
  NeurIPS 2024 (``cxcscmu/MATES``).
"""
from .if_select import select

__all__ = ["select"]
