"""Argmax gate: elect the higher ranking utility without a margin.

This is the decision rule of the canonical native ResNet-18 protocol, which elects the
validation argmax over the executed subsets. A challenger is adopted when its utility on the
ranking split is strictly larger than the reference's.
"""
from __future__ import annotations

from omniselect.core.gates.base import GateResult


def argmax_gate(reference: str, u_reference: float, challenger: str, u_challenger: float) -> GateResult:
    """Adopt iff u_challenger > u_reference."""
    adopted = float(u_challenger) > float(u_reference)
    return GateResult(
        "argmax",
        reference,
        challenger,
        bool(adopted),
        {
            "u_reference": float(u_reference),
            "u_challenger": float(u_challenger),
            "difference": float(u_challenger) - float(u_reference),
        },
    )
