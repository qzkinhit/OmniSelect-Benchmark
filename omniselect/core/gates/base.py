"""Gate result type and the ordered-family evaluation shared by every gate.

A gate compares one challenger with the reference and returns ``GateResult``. The family
runner tests up to K challengers in their ranking order and adopts the first that passes.
The statistical gates receive K so that their thresholds cover the whole family. The result
records every tested pair so tools/gate_replay.py can recompute the decision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence


@dataclass
class GateResult:
    """Outcome of one gate application to one (reference, challenger) pair."""

    kind: str
    reference: str
    challenger: str
    adopted: bool
    statistics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return asdict(self)


@dataclass
class FamilyDecision:
    """Result of testing an ordered family of challengers against one reference."""

    kind: str
    reference: str
    elected: str
    adopted: bool
    k_tested: int
    tests: list[GateResult]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "kind": self.kind,
            "reference": self.reference,
            "elected": self.elected,
            "adopted": self.adopted,
            "k_tested": self.k_tested,
            "tests": [t.to_dict() for t in self.tests],
        }


def run_family(
    kind: str,
    reference: str,
    challengers: Sequence[str],
    test_one: Callable[[str, int], GateResult],
    *,
    test_all: bool = False,
) -> FamilyDecision:
    """Apply ``test_one(challenger, K)`` in order and adopt the first challenger that passes.

    K is ``len(challengers)``. An empty family elects the reference. With ``test_all`` every
    challenger is tested and recorded, and the election is still the first one that passes.
    """
    k = len(challengers)
    tests: list[GateResult] = []
    elected = None
    for name in challengers:
        result = test_one(name, k)
        tests.append(result)
        if result.adopted and elected is None:
            elected = name
            if not test_all:
                break
    if elected is not None:
        return FamilyDecision(kind, reference, elected, True, k, tests)
    return FamilyDecision(kind, reference, reference, False, k, tests)
