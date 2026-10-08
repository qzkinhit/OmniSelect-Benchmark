"""Candidate roles: reference-eligible strategies and challenger-only candidates.

A reference-eligible strategy is a fixed selection rule executed as-is. The best of them on the
ranking split is the reference. Challenger-only candidates are built by the controller from the
construction split (grid finalists, consensus vote, coordinate ascent) and must pass the gate.
``role_of`` maps a candidate name and its stage to one of these labels for the run record.
"""
from __future__ import annotations

REFERENCE = "reference"
CHALLENGER = "challenger"
SCREENED_OUT = "screened_out"
BASELINE_ROW = "baseline_row"
DIAGNOSTIC = "diagnostic"            # method row that reads the corruption tags (clean oracle)
ROLES = (REFERENCE, CHALLENGER, SCREENED_OUT, BASELINE_ROW, DIAGNOSTIC)


def role_of(is_reference: bool, stage: str) -> str:
    """reference, challenger or screened_out for a controller candidate."""
    if stage == "screened_out":
        return SCREENED_OUT
    return REFERENCE if is_reference else CHALLENGER
