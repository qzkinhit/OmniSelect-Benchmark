"""Empirical margin gate on the ranking split (canonical rule).

The challenger is adopted when its ranking utility exceeds the reference's by more than
``margin_frac * |u_rank(reference)|``. With margin_frac = 0.015 this is the switch rule of the
2026-07 runners. No confidence statement is attached to this rule.
"""
from __future__ import annotations

from omniselect.core.gates.base import GateResult


def margin_gate(
    reference: str, u_reference: float, challenger: str, u_challenger: float, margin_frac: float
) -> GateResult:
    """Adopt iff u_challenger > u_reference + margin_frac * |u_reference|."""
    tau = float(margin_frac) * abs(float(u_reference))
    adopted = float(u_challenger) > float(u_reference) + tau
    return GateResult(
        "margin",
        reference,
        challenger,
        bool(adopted),
        {
            "u_reference": float(u_reference),
            "u_challenger": float(u_challenger),
            "margin_frac": float(margin_frac),
            "tau": tau,
            "difference": float(u_challenger) - float(u_reference),
        },
    )
