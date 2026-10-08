"""Track side of the alignment channel: from a warm-started last layer to channel scores and gradient factors.

``from_last_layer`` evaluates the errors e = dLoss / dOutput of a LastLayer (benchmark/Methods/_gradients.py)
on the pool and on V_con at its warm-start parameters and returns the alignment scores of
omniselect.core.signals.alignment, the factors (phi, e) of the per-record gradients for the
gradient coverage space, and a description of the source.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from omniselect.core.signals.alignment import gradient_alignment


def from_last_layer(model, source: str) -> dict[str, Any]:
    """Alignment scores, gradient factors and source of one LastLayer."""
    err = model.errors(model.phi, model.target)
    err_val = model.errors(model.phi_val, model.target_val)
    scores = gradient_alignment(model.phi, err, model.phi_val, err_val)
    return {"scores": scores, "phi": np.asarray(model.phi, dtype=float), "err": err,
            "source": source, "n_val": int(len(model.phi_val))}
