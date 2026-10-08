"""GLISTER (Killamsetty et al., AAAI 2021): greedy subset selection by a Taylor estimate of the V_con log-likelihood.

The per-record gradients g_i = phi_i e_i^T of a warm-started last layer (``extras['last_layer']``)
are fixed at the warm-start parameters Theta_0. In each of ``rounds`` rounds the records with the
largest gains g_i . G_V are added, where G_V is the mean V_con gradient at
Theta_S = Theta_0 - eta sum_{s in S} g_s. G_V is recomputed after every round (the R-greedy variant
of the authors' code). eta = 1 / (k mean ||phi||^2). Returns exactly k distinct pool indices in
selection order.
"""
from __future__ import annotations

import numpy as np

from benchmark.Methods._gradients import LastLayer
from omniselect.core.portfolio.registry import SelectionContext, register


def glister(model: LastLayer, k: int, *, rounds: int = 20, eta: float | None = None) -> list[int]:
    """GLISTER R-greedy selection of k records on the last-layer model."""
    n = len(model.phi)
    k = min(int(k), n)
    if k <= 0:
        return []
    errors = model.errors(model.phi, model.target)          # fixed at Theta_0, as in the authors' code
    step = model.step_size() / k if eta is None else float(eta)
    per_round = max(1, int(np.ceil(k / max(int(rounds), 1))))
    remaining = np.ones(n, dtype=bool)
    selected: list[int] = []
    grad_sum = np.zeros_like(model.theta)
    while len(selected) < k:
        theta = model.theta - step * grad_sum
        e_val = model.errors(model.phi_val, model.target_val, theta)
        g_val = model.phi_val.T @ e_val / max(len(model.phi_val), 1)           # (d, m) mean V_con gradient
        gains = np.sum((model.phi @ g_val) * errors, axis=1)
        gains[~remaining] = -np.inf
        take = min(per_round, k - len(selected))
        chosen = [int(i) for i in np.argsort(-gains, kind="stable")[:take]]
        selected.extend(chosen)
        remaining[chosen] = False
        grad_sum += model.phi[chosen].T @ errors[chosen]
    return selected


@register("glister", method_dir="GLISTER", family="new", fidelity="published-core transfer",
          requires=("last_layer",), display="GLISTER")
def glister_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """GLISTER on the track's warm-started last layer against V_con."""
    model = ctx.extras["last_layer"]
    model = model() if callable(model) else model
    return glister(model, k, rounds=int(ctx.extras.get("glister_rounds", 20)))
