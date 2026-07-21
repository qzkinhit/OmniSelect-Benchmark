"""Multi-Actor fusion — each signal is an actor, a console dynamically weights them."""
from .console import MultiActorConsole  # noqa: F401

__all__ = ["MultiActorConsole"]
