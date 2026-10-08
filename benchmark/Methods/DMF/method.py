"""DMF (Bai et al., ACL 2025): multi-actor weighting of the three channels, fused top-k.

``dmf_published_update`` (dmf_pub) applies Eqs. 6 to 8: fused top-k by the actor weights, each
weight moved by eta (R_j - mean R) and projected onto the simplex, for ``rounds`` rounds, with
rewards from the construction split. ``dmf_dynamic`` (dmf) is the multiplicative-weight proxy of
the historical batches. It is removed from the unified portfolio. ``text_dmf`` holds the text-lane
console baseline.
"""
from __future__ import annotations

import numpy as np

from omniselect.core.portfolio.registry import SelectionContext, register
from omniselect.core.signals.base import minmax


def dmf_dynamic(channel_scores: np.ndarray, k: int, val_reward, *, rounds: int = 6, eta: float = 0.5,
                seed: int = 0) -> list:
    """Multiplicative-weight proxy. Returns the fused selection with the best reward."""
    S = np.asarray(channel_scores, float)
    m, n = S.shape
    k = min(k, n)
    w = np.ones(m) / m
    best_sel, best_r = None, -np.inf
    for _ in range(rounds):
        fused = w @ S
        sel = list(np.argsort(-fused, kind="stable")[:k])
        r = float(val_reward(sel))
        if r > best_r:
            best_r, best_sel = r, sel
        rew = np.array([float(val_reward(list(np.argsort(-S[j], kind="stable")[:k]))) for j in range(m)])
        rew = (rew - rew.min()) / (rew.max() - rew.min() + 1e-9)
        w = w * np.exp(eta * rew)
        w /= w.sum()
    return best_sel if best_sel is not None else list(np.argsort(-(w @ S), kind="stable")[:k])


def dmf_published_update(channel_scores: np.ndarray, k: int, val_reward, *, rounds: int = 6,
                         eta: float = 0.5, seed: int = 0, return_trace: bool = False):
    """Eqs. 6 to 8 transfer with simplex projection. Returns the best fused selection (and trace)."""
    del seed
    S = np.asarray(channel_scores, dtype=float)
    if S.ndim != 2:
        raise ValueError("channel_scores must have shape (actors, samples)")
    m, n = S.shape
    k = min(int(k), n)
    theta = np.ones(m, dtype=float) / m
    best_sel, best_reward = None, -np.inf
    trace = []
    for step in range(int(rounds)):
        fused = theta @ S
        sel = [int(i) for i in np.argsort(-fused, kind="stable")[:k]]
        fused_reward = float(val_reward(sel))
        if fused_reward > best_reward:
            best_reward, best_sel = fused_reward, sel
        actor_rewards = np.array([
            float(val_reward([int(i) for i in np.argsort(-S[j], kind="stable")[:k]]))
            for j in range(m)
        ])
        mean_reward = float(actor_rewards.mean())
        raw_next = theta + float(eta) * (actor_rewards - mean_reward)
        projected = np.maximum(raw_next, 0.0)
        if projected.sum() <= 1e-12:
            projected = np.ones(m, dtype=float)
        projected /= projected.sum()
        trace.append({"round": step, "theta": theta.copy(),
                      "actor_rewards": actor_rewards.copy(),
                      "mean_reward": mean_reward, "raw_next": raw_next.copy(),
                      "projected_next": projected.copy(),
                      "fused_reward": fused_reward})
        theta = projected
    result = best_sel if best_sel is not None else [int(i) for i in np.argsort(-(theta @ S), kind="stable")[:k]]
    return (result, trace) if return_trace else result


def _channels(ctx: SelectionContext) -> np.ndarray:
    return np.stack([minmax(ctx.auth), minmax(ctx.influence), minmax(ctx.redundancy)], axis=0)


@register("dmf", method_dir="DMF", family="external", fidelity="proxy",
          requires=("auth", "influence", "redundancy", "construction_gain"), display="DMF proxy")
def dmf_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """dmf_dynamic over the three channels with ``extras['dmf_rounds']`` rounds."""
    return [int(i) for i in dmf_dynamic(_channels(ctx), k, val_reward=ctx.construction_gain,
                                        rounds=int(ctx.extras.get("dmf_rounds", 6)), seed=ctx.seed)]


@register("dmf_pub", method_dir="DMF", family="external", fidelity="published-update transfer",
          requires=("auth", "influence", "redundancy", "construction_gain"), display="DMF-pub")
def dmf_pub_strategy(ctx: SelectionContext, k: int) -> list[int]:
    """dmf_published_update over the three channels, six rounds, eta 0.5."""
    return dmf_published_update(_channels(ctx), k, val_reward=ctx.construction_gain, rounds=6, seed=ctx.seed)
