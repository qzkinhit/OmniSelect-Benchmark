"""Conflict-aware dynamic fusion controller.

Heterogeneous utility signals (redundancy / information density vs downstream
influence) frequently give *conflicting* per-record orderings: a sample can be
high-loss (high influence) yet near-duplicate of the already-kept set (high
redundancy). A plain static linear blend averages such conflicts into a
meaningless middle score. This controller treats each signal as a feedback
*channel* and fuses them with weights that are driven by their measured gain on a
small held-out probe, with four robustness/adaptivity mechanisms beyond a static
blend:

* sample-level conflict gating (resolve conflicts per record, not only per signal);
* held-out-gain rewards smoothed by an EMA (black-box, no per-sample gradients,
  so it transfers across text / image-text / code);
* a curriculum prior that shifts emphasis from coverage early to influence late;
* group-wise weights keyed by ``domain`` / ``modality`` (math and code can fuse
  signals at a different ratio than general text);
* a trust-region, variance-normalized weight update with a soft exploration floor.

All mechanisms default OFF, so with default construction this controller is
numerically identical to a softmax-weighted linear blend (keeps prior behavior).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..datatypes import UnifiedRecord
from ..signals.base import Signal, minmax


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class MultiActorConsole:
    def __init__(
        self,
        actors: Sequence[Tuple[str, Signal]],
        weights: Optional[Sequence[float]] = None,
        lr: float = 0.5,
        *,
        ema_beta: float = 0.0,            # >0 -> EMA-smooth noisy held-out rewards
        trust_region: Optional[float] = None,  # cap on |delta theta| per update (+ std-norm)
        min_weight: float = 0.0,          # soft exploration floor on each weight
        conflict_gate: bool = False,      # sample-level conflict resolution
        anneal: float = 0.0,              # curriculum prior strength
        group_key: Optional[Union[str, Callable[[UnifiedRecord], str]]] = None,
    ):
        if not actors:
            raise ValueError("MultiActorConsole needs at least one actor")
        self.names: List[str] = [name for name, _ in actors]
        self.signals: List[Signal] = [sig for _, sig in actors]
        m = len(actors)
        self.theta = np.array(weights if weights is not None else [0.0] * m, dtype=float)
        self.lr = float(lr)
        self.ema_beta = float(ema_beta)
        self.trust_region = trust_region
        self.min_weight = float(min_weight)
        self.conflict_gate = bool(conflict_gate)
        self.anneal = float(anneal)
        self.group_key = group_key
        self._ema: Dict[str, float] = {n: 0.0 for n in self.names}
        self.theta_group: Dict[str, np.ndarray] = {}
        self.last_conflict: float = 0.0

    # ---- weights ----
    def weights(self) -> np.ndarray:
        return _softmax(self.theta)

    def _weights_for(self, gid: Optional[str]) -> np.ndarray:
        if gid is None:
            return _softmax(self.theta)
        return _softmax(self.theta_group.setdefault(gid, self.theta.copy()))

    def _idx(self, name: str) -> Optional[int]:
        return self.names.index(name) if name in self.names else None

    def _group_ids(self, records: Sequence[UnifiedRecord]) -> Optional[List[str]]:
        if self.group_key is None:
            return None
        if callable(self.group_key):
            f = self.group_key
        elif self.group_key == "modality":
            f = lambda r: str(r.modality.value if hasattr(r.modality, "value") else r.modality)  # noqa: E731
        else:  # "domain"
            f = lambda r: r.domain  # noqa: E731
        return [f(r) for r in records]

    def _annealed(self, theta: np.ndarray, progress: float) -> np.ndarray:
        bias = np.zeros_like(theta)
        ri, ii = self._idx("redundancy"), self._idx("influence")
        if ri is not None:
            bias[ri] = 1.0 - progress  # early -> coverage/redundancy
        if ii is not None:
            bias[ii] = progress        # late -> influence
        return theta + self.anneal * bias

    # ---- scoring ----
    def actor_scores(self, records: Sequence[UnifiedRecord]) -> np.ndarray:
        return np.stack([minmax(s.score(records)) for s in self.signals], axis=0)

    def importance(
        self,
        records: Sequence[UnifiedRecord],
        scores: Optional[np.ndarray] = None,
        *,
        progress: Optional[float] = None,
    ) -> np.ndarray:
        S = scores if scores is not None else self.actor_scores(records)  # (m, N)
        gids = self._group_ids(records)
        out = np.zeros(S.shape[1], dtype=float)
        groups = set(gids) if gids is not None else {None}
        for gid in groups:
            mask = np.ones(S.shape[1], dtype=bool) if gids is None else np.array([g == gid for g in gids])
            th = self._weights_for(gid)
            if progress is not None and self.anneal > 0:
                th = _softmax(self._annealed(np.log(th + 1e-9), float(progress)))
            out[mask] = (th[:, None] * S[:, mask]).sum(axis=0)
        if self.conflict_gate and S.shape[0] > 1:
            spread = S.max(axis=0) - S.min(axis=0)  # 0 = agree, 1 = max conflict
            g = 1.0 - spread
            self.last_conflict = float(np.mean(spread))
            out = g * out + (1.0 - g) * 0.5  # conflicted samples pulled to neutral
        return out

    # ---- weight update from measured held-out gains ----
    def update(self, rewards: Dict[str, float], *, group: Optional[str] = None) -> np.ndarray:
        if self.ema_beta > 0:
            for n in self.names:
                self._ema[n] = (1 - self.ema_beta) * self._ema[n] + self.ema_beta * float(rewards.get(n, 0.0))
            r = np.array([self._ema[n] for n in self.names], dtype=float)
        else:
            r = np.array([float(rewards.get(n, 0.0)) for n in self.names], dtype=float)

        adv = r - r.mean()
        if self.trust_region is not None:
            sd = r.std()
            if sd > 1e-8:
                adv = adv / sd
            step = np.clip(self.lr * adv, -self.trust_region, self.trust_region)
        else:
            step = self.lr * adv

        th = self.theta if group is None else self.theta_group.setdefault(group, self.theta.copy())
        th = th + step
        if self.min_weight > 0:
            w = np.clip(_softmax(th), self.min_weight, None)
            th = np.log(w / w.sum() + 1e-9)
        if group is None:
            self.theta = th
        else:
            self.theta_group[group] = th
        return _softmax(th)
