"""DMF baseline — dynamic multi-signal fusion (base variant).

The representative "fuse heterogeneous utility signals with adaptive, feedback-
driven weights" comparison: redundancy + influence signals blended by a single
dynamic weight, with every advanced fusion mechanism left off.
"""
from .dmf_select import build_fusion, select

__all__ = ["select", "build_fusion"]
