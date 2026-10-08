"""Alignment channel of protocol v2.1: cosine of each record's last-layer gradient with the mean V_con gradient.

For a last layer Theta (d x m) on features phi, a record's loss gradient is g_i = phi_i e_i^T with
e_i = dLoss / dOutput at the warm-start parameters. With G_V = (1 / n_val) sum_v phi_v e_v^T the mean
V_con gradient, align_i = <g_i, G_V> / (||g_i|| ||G_V||) = phi_i^T G_V e_i / (||phi_i|| ||e_i|| ||G_V||).
The numerator is GLISTER's round-0 gain, so the channel is that gain normalized per record and by
||G_V||. The computation forms Phi G_V (n x m) and never the d x m gradient of a record. The channel
reads the pool and V_con only.
"""
from __future__ import annotations

import numpy as np


def mean_gradient(phi_val: np.ndarray, err_val: np.ndarray) -> np.ndarray:
    """G_V = Phi_val^T E_val / n_val, shape (d, m)."""
    phi_val = np.asarray(phi_val, dtype=float)
    return phi_val.T @ np.asarray(err_val, dtype=float) / max(len(phi_val), 1)


def gradient_alignment(phi: np.ndarray, err: np.ndarray, phi_val: np.ndarray, err_val: np.ndarray) -> np.ndarray:
    """align_i in [-1, 1] for every pool record (0 where a gradient vanishes)."""
    phi = np.asarray(phi, dtype=float)
    err = np.asarray(err, dtype=float)
    g_val = mean_gradient(phi_val, err_val)
    numerator = np.sum((phi @ g_val) * err, axis=1)
    denominator = np.linalg.norm(phi, axis=1) * np.linalg.norm(err, axis=1) * np.linalg.norm(g_val)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-300)
