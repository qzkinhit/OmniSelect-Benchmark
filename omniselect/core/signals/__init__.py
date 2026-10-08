"""Signal channels: authenticity, influence and redundancy, and the shared min-max normalization."""
from omniselect.core.signals.base import Signal, minmax  # noqa: F401
from omniselect.core.signals.redundancy import RedundancySignal, hashed_features, set_redundancy  # noqa: F401
from omniselect.core.signals.influence import InfluenceSignal  # noqa: F401
from omniselect.core.signals.authenticity import AuthenticitySignal  # noqa: F401

__all__ = [
    "Signal",
    "minmax",
    "RedundancySignal",
    "InfluenceSignal",
    "AuthenticitySignal",
    "hashed_features",
    "set_redundancy",
]
