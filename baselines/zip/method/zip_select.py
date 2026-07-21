"""ZIP / Entropy-Law selection logic — model-free, pure CPU (zlib + stdlib).

ZIP (Yin et al., "Entropy Law: The Story Behind Data Compression and LLM
Performance", USTC-StarTeam/ZIP) selects a low-redundancy subset by *minimizing*
the compression ratio of the chosen set: a set that compresses well is repetitive
(low information), so a set that compresses *poorly* carries more distinct
information. The compression ratio of a set ``D`` is

    g(D) = Bits(D) / Bits(C(D))

where ``C`` is a lossless compressor (here ``zlib``) and ``Bits(.)`` the byte
size; a *lower* ``g`` means the set is harder to compress, i.e. more informative
/ less redundant. ZIP greedily builds the kept set to keep ``g`` as low as
possible, faithful to Algorithm 1 of the paper, in three nested stages:

* **Stage 1 (global)** — score every remaining sample by its *own* sample-level
  compression ratio and take the Bottom-K1 (the K1 with the lowest individual
  ratio = highest information density) as a coarse candidate pool. This
  initializes the "information-redundancy state".
* **Stage 2 (local, coarse)** — for each Stage-1 candidate, compute the *merged*
  ratio ``g(selected ∪ {d})`` and keep the Bottom-K2 with the smallest merged
  ratio (the ones that least inflate the kept set's compressibility).
* **Stage 3 (local, fine)** — greedily add Stage-2 candidates one by one, each
  time recomputing the marginal merged ratio against the (growing) selected set,
  always taking the current argmin, until the K2 block is consumed or the budget
  ``k`` is met.

The outer loop repeats Stages 1–3 until ``k`` samples are selected. Everything is
a pure function of ``(texts, ids, k, ...)`` so the runner stays a thin I/O shell,
mirroring the other baselines.
"""
from __future__ import annotations

import zlib
from typing import List, Optional, Sequence, Tuple

# Pad each sample with a newline so concatenation keeps record boundaries; this
# matches the "join with a separator" convention used elsewhere in the codebase.
_SEP = b"\n"


def _bits(blob: bytes) -> int:
    """Raw size of ``blob`` in bytes (``Bits(.)`` in the Entropy-Law notation)."""
    return len(blob)


def _compressed_bits(blob: bytes, *, level: int = 6) -> int:
    """Size of the losslessly compressed ``blob`` (``Bits(C(.))``)."""
    if not blob:
        return 0
    return len(zlib.compress(blob, level))


def compression_ratio(blob: bytes, *, level: int = 6) -> float:
    """Set compression ratio ``g = Bits(D) / Bits(C(D))`` for a byte blob.

    Higher ``g`` = more compressible = more redundant / less information. An empty
    blob has no information, so we return ``0.0`` (it can never lower the ratio of
    a non-empty set and is never preferred). The ratio is ``>= 1`` for any real
    compressor on non-trivial input.
    """
    raw = _bits(blob)
    if raw == 0:
        return 0.0
    comp = _compressed_bits(blob, level=level)
    if comp == 0:
        return 0.0
    return raw / comp


def _encode(texts: Sequence[str]) -> List[bytes]:
    """Pre-encode each sample's text to bytes once (so we never re-encode)."""
    return [((t or "")).encode("utf-8") for t in texts]


def _merged_blob(selected_blob: bytes, cand: bytes) -> bytes:
    """Concatenate the current selected blob with one candidate's bytes."""
    if not selected_blob:
        return cand
    return selected_blob + _SEP + cand


def _sample_ratios(blobs: Sequence[bytes], *, level: int) -> List[float]:
    """Per-sample compression ratio ``g({d})`` for every candidate (Stage 1 key)."""
    return [compression_ratio(b, level=level) for b in blobs]


class _PrefixState:
    """Bit-exact incremental evaluator for g(S + sep + d) (docs/zip_protocol_review.md).

    zlib's streamed deflate (Z_NO_FLUSH chunks + one final Z_FINISH) emits output
    byte-identical to one-shot ``zlib.compress`` at the same level, so copying the
    prefix compressor state and finishing with only the candidate reproduces the
    EXACT compressed length of the concatenation without recompressing the prefix.
    This changes only the COST of computing the same number (O(prefix) -> O(cand)
    per evaluation); selection semantics are untouched. Equivalence is enforced by
    element-wise order+trace tests against the reference path (ZIP_INCREMENTAL=0)."""

    def __init__(self, level: int = 6):
        self._obj = zlib.compressobj(level)
        self._out_len = 0   # compressed bytes already emitted for the prefix
        self._raw_len = 0   # raw byte length of the prefix (== len(selected_blob))

    def _feed(self, cand: bytes) -> bytes:
        return cand if self._raw_len == 0 else _SEP + cand

    def merged_ratio(self, cand: bytes) -> float:
        """== compression_ratio(_merged_blob(selected_blob, cand), level=level)."""
        feed = self._feed(cand)
        raw = self._raw_len + len(feed)
        if raw == 0:
            return 0.0
        c = self._obj.copy()
        comp = self._out_len + len(c.compress(feed)) + len(c.flush())
        return raw / comp if comp else 0.0

    def advance(self, cand: bytes) -> None:
        """Commit ``cand`` to the prefix (mirrors selected_blob growth)."""
        feed = self._feed(cand)
        self._out_len += len(self._obj.compress(feed))
        self._raw_len += len(feed)

    def set_ratio(self) -> float:
        """== compression_ratio(selected_blob, level=level) for the current prefix."""
        if self._raw_len == 0:
            return 0.0
        c = self._obj.copy()
        comp = self._out_len + len(c.flush())
        return self._raw_len / comp if comp else 0.0


def zip_select(
    texts: Sequence[str],
    ids: Sequence[str],
    k: int,
    *,
    k1_ratio: float = 0.1,
    k2_ratio: float = 0.5,
    level: int = 6,
    seed: int = 0,
) -> Tuple[List[int], List[float]]:
    """Select ``k`` low-redundancy indices via the three-stage ZIP greedy.

    Parameters
    ----------
    texts, ids : the standardized ``text`` field and stable ids (parallel).
    k : number of samples to keep (already budget-resolved by the runner). When
        ``k == len(texts)`` the full greedy *ordering* of the pool is returned, so
        the caller can truncate it under a token budget.
    k1_ratio : Stage-1 pool size as a fraction of the *remaining* candidates
        (Bottom-K1 by sample-level ratio). ``K1 = max(1, round(k1_ratio * n_rem))``.
    k2_ratio : Stage-2 block size as a fraction of the Stage-1 pool
        (Bottom-K2 by merged ratio). ``K2 = max(1, round(k2_ratio * K1))``.
    level : ``zlib`` compression level (deterministic; default 6).
    seed : kept for interface symmetry / reproducible tie-breaking; ties in the
        ratio are already broken deterministically by original index.

    Returns
    -------
    (selected_idx, ratio_trace) — the selected pool indices in greedy order, and,
    for diagnostics, the running set compression ratio ``g(selected)`` recorded
    after each pick (parallel to ``selected_idx``).
    """
    n = len(texts)
    if n == 0 or k <= 0:
        return [], []
    k = min(k, n)

    blobs = _encode(texts)
    # Sample-level ratios are fixed (each is g({d})); precompute once for Stage 1.
    sample_ratio = _sample_ratios(blobs, level=level)

    import os as _os
    _incr = _os.environ.get("ZIP_INCREMENTAL", "1") == "1"
    state = _PrefixState(level) if _incr else None

    remaining: List[int] = list(range(n))
    selected: List[int] = []
    selected_blob = b""
    ratio_trace: List[float] = []

    while len(selected) < k and remaining:
        # ---- Stage 1 (global): Bottom-K1 by own sample-level compression ratio.
        # Lowest ratio = least compressible alone = highest information density.
        n_rem = len(remaining)
        k1 = max(1, int(round(k1_ratio * n_rem)))
        k1 = min(k1, n_rem)
        # Stable sort by (sample_ratio, original index) so ties are deterministic.
        stage1 = sorted(remaining, key=lambda i: (sample_ratio[i], i))[:k1]

        # ---- Stage 2 (local, coarse): Bottom-K2 by merged ratio g(S ∪ {d}).
        k2 = max(1, int(round(k2_ratio * len(stage1))))
        k2 = min(k2, len(stage1))
        merged = [
            ((state.merged_ratio(blobs[i]) if _incr else
              compression_ratio(_merged_blob(selected_blob, blobs[i]), level=level)), i)
            for i in stage1
        ]
        merged.sort(key=lambda t: (t[0], t[1]))
        stage2 = [i for _, i in merged[:k2]]

        # ---- Stage 3 (local, fine): greedily add Stage-2 candidates one at a
        # time, recomputing the marginal merged ratio against the growing set and
        # always taking the current argmin, until the block is consumed or k met.
        block = set(stage2)
        while block and len(selected) < k:
            best_i = -1
            best_ratio = float("inf")
            for i in block:
                g = (state.merged_ratio(blobs[i]) if _incr else
                     compression_ratio(_merged_blob(selected_blob, blobs[i]), level=level))
                if g < best_ratio or (g == best_ratio and (best_i < 0 or i < best_i)):
                    best_ratio = g
                    best_i = i
            selected.append(best_i)
            if _incr:
                state.advance(blobs[best_i])
                selected_blob = _merged_blob(selected_blob, blobs[best_i])
                ratio_trace.append(state.set_ratio())
            else:
                selected_blob = _merged_blob(selected_blob, blobs[best_i])
                ratio_trace.append(compression_ratio(selected_blob, level=level))
            block.discard(best_i)
            remaining.remove(best_i)

    return selected, ratio_trace


# Unified baseline entrypoint: every baseline exposes ``select(...)`` returning a
# list of selected pool indices. ZIP is model-free and ignores ``records`` meta
# beyond the ``text`` field, so accept either pre-extracted ``texts``+``ids`` or a
# sequence of ``UnifiedRecord``-like objects (anything exposing ``.text``/``.id``).
def select(
    records: Optional[Sequence[object]] = None,
    k: Optional[int] = None,
    *,
    texts: Optional[Sequence[str]] = None,
    ids: Optional[Sequence[str]] = None,
    k1_ratio: float = 0.1,
    k2_ratio: float = 0.5,
    level: int = 6,
    seed: int = 0,
) -> List[int]:
    """Return the selected pool indices ``List[int]`` (the unified baseline API).

    Provide either ``records`` (objects with ``.text`` / ``.id``) or the parallel
    ``texts`` + ``ids``. ``k`` is the number to keep; if omitted it defaults to the
    full pool, in which case the *complete* greedy ordering is returned (so the
    caller can truncate it under a token budget). Tie-breaking is deterministic.
    """
    if texts is None:
        if records is None:
            return []
        texts = [getattr(r, "text", "") or "" for r in records]
        if ids is None:
            ids = [getattr(r, "id", str(i)) for i, r in enumerate(records)]
    if ids is None:
        ids = [str(i) for i in range(len(texts))]
    n = len(texts)
    if k is None:
        k = n
    selected_idx, _ = zip_select(
        texts,
        ids,
        int(k),
        k1_ratio=k1_ratio,
        k2_ratio=k2_ratio,
        level=level,
        seed=seed,
    )
    return selected_idx
