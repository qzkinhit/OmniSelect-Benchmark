"""Statistical gate tests on a PairedSample and the confirmation-split audit of v2 cells.

``test_paired`` applies lcb, bootstrap or eprocess to one PairedSample with the configured reading
order (stratified bootstrap replicates and stratified e-process sequences come from
core/gates/paired.py). ``audit_family`` computes, for the reference and each ordered challenger,
the paired mean gain, the bootstrap 90% interval, the win share, the e-value with K fixed to the
configured family size, and the flag e >= K / delta. The audit never changes the election.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from omniselect.core.gates.base import GateResult
from omniselect.core.gates.bootstrap import bootstrap_gate
from omniselect.core.gates.eprocess import eprocess_gate
from omniselect.core.gates.lcb import lcb_gate
from omniselect.core.gates.paired import PairedSample


def test_paired(kind: str, reference: str, challenger: str, sample: PairedSample, *, k: int, seed: int,
                delta: float, eps: float, n_boot: int, p_beat_min: float, order: str) -> GateResult:
    """GateResult of lcb, bootstrap or eprocess on ``sample`` with reading order ``order``."""
    d, w = sample.differences, sample.weights
    if kind == "lcb":
        return lcb_gate(reference, challenger, d, w, delta, k)
    design = sample.design(order)
    if kind == "bootstrap":
        return bootstrap_gate(reference, challenger, d, w, n_boot=n_boot, p_beat_min=p_beat_min, k=k, seed=seed,
                              boot_means=sample.bootstrap_means(order, n_boot, seed), design=design)
    if kind == "eprocess":
        return eprocess_gate(reference, challenger, d, w, delta=delta, eps=eps, k=k, seed=seed,
                             sequence=sample.sequence(order, seed), design=design)
    raise ValueError(f"no paired test for gate kind {kind!r}")


def audit_family(reference: str, challengers: Sequence[tuple[str, PairedSample]], *, k: int, seed: int,
                 delta: float, eps: float, n_boot: int, order: str, split: str) -> dict[str, Any]:
    """Audit entries of the ordered challengers against ``reference`` on ``split``.

    Each entry holds mean_gain (weighted mean difference), boot_interval90 (5% and 95% quantiles of
    the bootstrap replicates), p_win, e_value, log_e_value, threshold K / delta, certified
    (e >= K / delta), stop_index (units read when the wealth first reached the threshold), n_units,
    n_read and the reading design.
    """
    entries = []
    k = max(int(k), 1)
    for name, sample in challengers:
        ep = eprocess_gate(reference, name, sample.differences, sample.weights, delta=delta, eps=eps, k=k,
                           seed=seed, sequence=sample.sequence(order, seed), design=sample.design(order))
        means = sample.bootstrap_means(order, n_boot, seed)
        stats = ep.statistics
        entries.append({
            "challenger": name,
            "mean_gain": sample.mean(),
            "boot_interval90": [float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))],
            "p_win": float(np.mean(means > 0.0)),
            "e_value": stats["e_value"],
            "log_e_value": stats["log_e_value"],
            "threshold": stats["threshold"],
            "certified": bool(stats["log_e_value"] >= np.log(stats["threshold"]) - 1e-12),
            "stop_index": stats["stop_index"],
            "n_units": stats["n_units"],
            "n_read": stats["n_read"],
            "reading": stats["reading"],
        })
    return {"split": split, "reference": reference, "K": k, "delta": float(delta), "eps": float(eps),
            "n_boot": int(n_boot), "reading_order": order, "seed": int(seed), "entries": entries,
            "note": "diagnostic, the election does not read it"}
