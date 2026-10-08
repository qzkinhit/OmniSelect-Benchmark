"""RegMix (Liu et al., ICLR 2025) and Perplexity-Correlations (Thrush et al., ICLR 2025), text probes.

``regmix_mixture`` regresses a construction-split metric on Dirichlet domain mixtures and returns
the predicted-best mixture. ``perpcorr_select`` allocates the budget across domains by softmax of
standardized domain gains and takes low reference-perplexity records first. Both are group-level
proxies that never enter the main table. They are not registered as portfolio members.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def regmix_mixture(domain_ids: Sequence, train_eval_fn, *, n_probe: int = 12, seed: int = 0) -> dict:
    """Predicted-best domain mixture of a linear regression over ``n_probe`` probe mixtures."""
    doms = sorted(set(domain_ids))
    D = len(doms)
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(D), size=n_probe)
    y = np.array([float(train_eval_fn({d: float(w[j]) for j, d in enumerate(doms)})) for w in W])
    Xc = W - W.mean(axis=0, keepdims=True)
    coef, *_ = np.linalg.lstsq(Xc, y - y.mean(), rcond=None)
    cand = rng.dirichlet(np.ones(D), size=4096)
    best = cand[np.argmax(cand @ coef)]
    return {d: float(best[j]) for j, d in enumerate(doms)}


def perpcorr_select(ppl: np.ndarray, domain_ids: Sequence, domain_gain: dict, k: int) -> list:
    """Domain allocation by softmax of standardized gains, lowest perplexity first within a domain."""
    doms = sorted(set(domain_ids))
    g = np.array([float(domain_gain.get(d, 0.0)) for d in doms])
    gz = (g - g.mean()) / (g.std() + 1e-9)
    alloc = np.exp(gz) / np.exp(gz).sum()
    dom_arr = np.asarray(list(domain_ids))
    out = []
    for j, d in enumerate(doms):
        idx = np.where(dom_arr == d)[0]
        take = min(len(idx), int(round(alloc[j] * k)))
        order = idx[np.argsort(ppl[idx], kind="stable")]
        out.extend(int(i) for i in order[:take])
    rest = [i for i in np.argsort(ppl, kind="stable") if i not in set(out)]
    out.extend(int(i) for i in rest[: max(0, k - len(out))])
    return out[:k]
