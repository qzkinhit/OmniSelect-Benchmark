"""Shared RNG discipline of paired runs.

``reset_for_strategy`` resets Python, numpy and torch RNGs to stable_seed(seed, 'select', name)
before a strategy runs, so a selection does not depend on the order strategies are called.
``fit_seed`` is the stage seed of a learner fit (stable_seed(seed, stage) under paired RNG,
the plain run seed otherwise). Both reuse omniselect.tools.pairing.
"""
from __future__ import annotations


from omniselect.tools.pairing import reset_rng, stable_seed


def reset_for_strategy(seed: int, name: str, paired: bool) -> None:
    """reset_rng(seed, 'select', name) when ``paired``."""
    if paired:
        reset_rng(seed, "select", name)


def fit_seed(seed: int, stage: str, paired: bool, reset: bool = True) -> int:
    """Seed of a fit at ``stage``. With ``reset`` the global RNGs are reset to it."""
    if not paired:
        return int(seed)
    return int(reset_rng(seed, stage) if reset else stable_seed(seed, stage))
