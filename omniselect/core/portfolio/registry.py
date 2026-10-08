"""Strategy registry: name -> selection function with its family, fidelity and requirements.

A strategy is ``fn(ctx, k) -> list[int]`` over a ``SelectionContext``. The benchmark package
registers every method under ``benchmark/Methods/<Name>/`` through ``register`` when
``ensure_loaded`` imports it, so the controller package itself imports no method code. The
controller treats every registered strategy that the membership table admits as a candidate.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class SelectionContext:
    """Everything a strategy may read. Absent inputs are None.

    features: pool representation used by geometric strategies. auth, influence, redundancy:
    raw channel scores. labels: observed pool labels (classification). proba: probe class
    probabilities with columns aligned to label ids. construction_gain: utility on the
    construction split (DMF, DsDm). extras: track-specific inputs (for example ``imp_dyn``).
    """

    n: int
    seed: int
    features: np.ndarray
    auth: Optional[np.ndarray] = None
    influence: Optional[np.ndarray] = None
    redundancy: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    proba: Optional[np.ndarray] = None
    records: Optional[list] = None
    construction_gain: Optional[Callable[[list], float]] = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Strategy:
    """A registered selection strategy."""

    name: str
    fn: Callable[[SelectionContext, int], list]
    method_dir: str
    family: str                     # builtin | control | external | new
    fidelity: str                   # exact | published-core transfer | published-update transfer |
                                    # local implementation | proxy | qualitative protocol transfer
    requires: frozenset = frozenset()
    display: str = ""


REGISTRY: dict[str, Strategy] = {}
_LOADED = False


def register(
    name: str,
    *,
    method_dir: str,
    family: str,
    fidelity: str,
    requires: tuple[str, ...] = (),
    display: str = "",
) -> Callable[[Callable[[SelectionContext, int], list]], Callable[[SelectionContext, int], list]]:
    """Decorator that registers ``fn`` under ``name``. A second registration of a name raises."""

    def decorate(fn: Callable[[SelectionContext, int], list]) -> Callable[[SelectionContext, int], list]:
        if name in REGISTRY and REGISTRY[name].fn is not fn:
            raise ValueError(f"strategy {name!r} is already registered")
        REGISTRY[name] = Strategy(name, fn, method_dir, family, fidelity, frozenset(requires), display or name)
        return fn

    return decorate


def ensure_loaded() -> None:
    """Import ``benchmark.Methods`` once so its methods register themselves."""
    global _LOADED
    if not _LOADED:
        importlib.import_module("benchmark.Methods")
        _LOADED = True


def get(name: str) -> Strategy:
    """Return the registered strategy ``name``."""
    ensure_loaded()
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


def missing_inputs(strategy: Strategy, ctx: SelectionContext) -> list[str]:
    """Names in ``strategy.requires`` that the context does not provide."""
    out = []
    for key in sorted(strategy.requires):
        value = getattr(ctx, key, None) if hasattr(ctx, key) else ctx.extras.get(key)
        if value is None:
            out.append(key)
    return out
