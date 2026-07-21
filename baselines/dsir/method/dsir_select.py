"""DSIR selection logic — pure numpy, CPU-only, no torch/transformers.

DSIR (Xie et al., NeurIPS 2023) selects raw data whose feature distribution looks
like a *target* distribution. We use the modality-agnostic ``text`` field and a
hashed word n-gram bag-of-features as the cheap, deterministic feature space (the
paper uses hashed n-grams too). With those features we:

1. fit a unigram-style multinomial ``q`` over the whole raw pool (proposal) and a
   multinomial ``p`` over the target subset (here: ``math`` + ``code`` records);
2. score each raw example by its importance-weight log-ratio
   ``log w_i = sum_f c_{i,f} * (log p_f - log q_f)`` (the per-feature log p/q,
   accumulated over the example's own n-gram counts);
3. resample ``k`` examples with the Gumbel top-k trick on those log-weights — an
   exact, seedable equivalent of sampling-without-replacement proportional to
   ``w_i`` (DSIR's importance resampling), gracefully degrading to deterministic
   top-k when ``noise=0``.

Everything is a pure function of ``(texts, ids, k, ...)`` so the runner stays a thin
I/O shell, mirroring the other baselines.
"""
from __future__ import annotations

import re
import zlib
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Word-ish tokens: keep alphanumerics, drop punctuation/whitespace. Lowercased so
# casing does not fragment the bag-of-features.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def hashed_ngram_counts(
    texts: Sequence[str],
    *,
    dim: int = 4096,
    ngram: int = 2,
) -> np.ndarray:
    """Hashed word n-gram (1..``ngram``) count matrix, shape ``(len(texts), dim)``.

    Uses ``zlib.crc32`` as the hash so features are stable across runs/processes
    (unlike Python's salted ``hash``), keeping selection reproducible. Counts are
    raw (un-normalized) because DSIR's log-ratio score sums per-feature evidence
    over each example's own token counts.
    """
    X = np.zeros((len(texts), dim), dtype=np.float64)
    for i, text in enumerate(texts):
        toks = _tokenize(text)
        for n in range(1, ngram + 1):
            for j in range(len(toks) - n + 1):
                gram = " ".join(toks[j : j + n]).encode("utf-8")
                X[i, zlib.crc32(gram) % dim] += 1.0
    return X


def _multinomial(counts: np.ndarray, *, smoothing: float = 1.0) -> np.ndarray:
    """Laplace-smoothed multinomial over hashed features from summed counts."""
    total = counts.sum() + smoothing * counts.shape[0]
    return (counts + smoothing) / total


def importance_log_weights(
    X: np.ndarray,
    target_mask: np.ndarray,
    *,
    smoothing: float = 1.0,
    target_counts: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-example DSIR log importance-weight ``log(p/q)`` accumulated over n-grams.

    ``q`` is fit on the whole pool (raw proposal). ``p`` is fit on an explicit
    ``target_counts`` feature histogram when given (the canonical DSIR setup: a
    *clean external target sample* such as the held-out reference), otherwise on the
    in-pool ``target_mask`` rows. The per-feature log-ratio is dotted with each
    example's own count vector, then centered to mean 0 for numerical stability (a
    constant shift does not change resampling/top-k order).
    """
    pool_counts = X.sum(axis=0)
    tgt_counts = target_counts if target_counts is not None else X[target_mask].sum(axis=0)

    q = _multinomial(pool_counts, smoothing=smoothing)
    p = _multinomial(tgt_counts, smoothing=smoothing)

    log_ratio = np.log(p) - np.log(q)          # (dim,)
    log_w = X @ log_ratio                       # (n,)
    log_w -= log_w.mean()
    return log_w


def _target_mask(domains: Optional[Sequence[str]], n: int) -> np.ndarray:
    """Rows treated as the DSIR target. Prefer math/code domains; else fall back
    to the whole pool (so the method still runs on an undomained pool)."""
    if domains is None:
        return np.ones(n, dtype=bool)
    mask = np.array([str(d).lower() in ("math", "code") for d in domains], dtype=bool)
    if not mask.any():
        mask = np.ones(n, dtype=bool)
    return mask


def dsir_select(
    texts: Sequence[str],
    ids: Sequence[str],
    k: int,
    *,
    domains: Optional[Sequence[str]] = None,
    target_texts: Optional[Sequence[str]] = None,
    dim: int = 4096,
    ngram: int = 2,
    smoothing: float = 1.0,
    noise: float = 1.0,
    seed: int = 0,
) -> Tuple[List[int], np.ndarray]:
    """Select ``k`` indices by DSIR importance resampling.

    Parameters
    ----------
    texts, ids : the standardized ``text`` field and stable ids (parallel).
    k : number of examples to keep (already budget-resolved by the runner).
    domains : per-record domain; ``math``/``code`` rows define the target ``p`` when
        no ``target_texts`` is given (legacy in-pool target).
    target_texts : an explicit clean target sample (e.g. the multi-modal held-out
        reference). When given, ``p`` is fit on these — the canonical, fair DSIR
        target — instead of in-pool domain rows.
    noise : Gumbel scale. ``>0`` = importance *resampling* without replacement
        (proportional to ``w_i``); ``0`` = deterministic importance top-k.

    Returns
    -------
    (selected_idx, log_weights) — indices into the pool and the raw per-example
    log importance-weights (for diagnostics / the manifest).
    """
    n = len(texts)
    if n == 0 or k <= 0:
        return [], np.zeros(0, dtype=np.float64)
    k = min(k, n)

    X = hashed_ngram_counts(texts, dim=dim, ngram=ngram)
    mask = _target_mask(domains, n)
    tgt_counts = None
    if target_texts is not None and len(target_texts) > 0:
        tgt_counts = hashed_ngram_counts(target_texts, dim=dim, ngram=ngram).sum(axis=0)
    log_w = importance_log_weights(X, mask, smoothing=smoothing, target_counts=tgt_counts)

    # Gumbel top-k == sampling without replacement proportional to exp(log_w).
    rng = np.random.default_rng(seed)
    keys = log_w.copy()
    if noise > 0:
        gumbel = -np.log(-np.log(rng.uniform(low=1e-12, high=1.0, size=n)))
        keys = keys + noise * gumbel
    idx = np.argsort(-keys, kind="stable")[:k]
    return [int(i) for i in idx], log_w
