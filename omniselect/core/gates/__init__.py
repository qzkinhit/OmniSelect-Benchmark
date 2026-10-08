"""Decision gates.

margin and argmax compare ranking utilities. lcb, bootstrap and eprocess test paired per-unit
differences.
"""
from omniselect.core.gates.argmax import argmax_gate  # noqa: F401
from omniselect.core.gates.base import FamilyDecision, GateResult, run_family  # noqa: F401
from omniselect.core.gates.bootstrap import bootstrap_gate  # noqa: F401
from omniselect.core.gates.eprocess import eprocess_gate  # noqa: F401
from omniselect.core.gates.lcb import hoeffding_radius, lcb_gate  # noqa: F401
from omniselect.core.gates.margin import margin_gate  # noqa: F401
