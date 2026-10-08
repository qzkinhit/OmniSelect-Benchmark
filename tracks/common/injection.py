"""Controlled corruption of the training pool and of the validation labels, per data family.

The canonical mechanisms reproduce the 2026-07 runners bit for bit: vision (label flip, duplicate
marker, hard marker, with duplicated embeddings), forecasting windows (Gaussian corruption, flat,
shuffle, near-duplicate), process and tabular rows (label flip, feature corruption,
near-duplicate). The unseen mechanisms (``mechanism_set=unseen``) corrupt the same records
(default_rng(seed + 7) picks them) by mechanisms the detectors were not designed for: vision blur,
JPEG and class-dependent flips, forecasting scale drift, bias drift and stuck sensor, rows stuck
features and class-dependent flips. ``contaminate_validation`` adds symmetric, natural (human) or
class-prior-shift noise to the validation labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MECHANISM_SETS = ("canonical", "unseen")
VAL_NOISE_KINDS = ("none", "symmetric", "natural", "prior_shift")
VAL_NOISE_ALIASES = {"class_prior_shift": "prior_shift"}
# Asymmetric CIFAR-10 noise (Patrini et al., CVPR 2017): truck to automobile, bird to airplane, deer to
# horse, cat and dog swapped. The other classes move to the next class id.
CIFAR10_CLASS_MAP = {9: 1, 2: 0, 4: 7, 3: 5, 5: 3, 0: 1, 1: 2, 6: 7, 7: 8, 8: 9}


def inject_vision_labels(labels: np.ndarray, seed: int, noise_frac: float, n_classes: int = 100):
    """Split round(noise_frac * n) records in three: flipped label, duplicate marker, hard marker.

    Returns (observed labels, tags, duplicate indices). ``n_classes`` is the range the flipped
    label is drawn from. The canonical runner used 100 on every dataset.
    """
    rng = np.random.default_rng(seed + 7)
    n = len(labels)
    obs = labels.copy()
    tag = np.array(["high"] * n, dtype=object)
    n_low = int(round(noise_frac * n))
    low_idx = rng.permutation(n)[:n_low]
    per = max(1, n_low // 3)
    flip, dup, hard = low_idx[:per], low_idx[per:2 * per], low_idx[2 * per:]
    for i in flip:
        obs[i] = rng.integers(n_classes)
        tag[i] = "flip"
    for i in dup:
        tag[i] = "dup"
    for i in hard:
        tag[i] = "hard"
    return obs, tag, dup


def duplicate_embeddings(X: np.ndarray, dup_idx: np.ndarray) -> np.ndarray:
    """Copy the first eighth of ``dup_idx`` onto every duplicate slot with 0.01 noise, then renormalize.

    The noise of slot i comes from default_rng(i). Operates on a copy.
    """
    X = X.copy()
    if len(dup_idx) > 0:
        seeds = dup_idx[: max(1, len(dup_idx) // 8)]
        for j, i in enumerate(dup_idx):
            X[i] = X[seeds[j % len(seeds)]] + 0.01 * np.random.default_rng(i).standard_normal(X.shape[1])
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return X


def inject_windows(X: np.ndarray, Y: np.ndarray, seed: int, noise_frac: float):
    """Forecasting windows: a quarter each of corrupt, flat, shuffle, and the rest near-duplicate."""
    rng = np.random.default_rng(seed + 7)
    n, length = X.shape
    Xn, Yn = X.copy(), Y.copy()
    tag = np.array(["high"] * n, dtype=object)
    n_low = int(round(noise_frac * n))
    low = rng.permutation(n)[:n_low]
    per = max(1, n_low // 4)
    corrupt, flat, shuf, dup = low[:per], low[per:2 * per], low[2 * per:3 * per], low[3 * per:]
    sd = X.std() + 1e-8
    for i in corrupt:
        Xn[i] = Xn[i] + rng.standard_normal(length) * sd * 1.5
        tag[i] = "corrupt"
    for i in flat:
        Xn[i] = np.full(length, Xn[i].mean())
        tag[i] = "flat"
    for i in shuf:
        Xn[i] = rng.permutation(Xn[i])
        tag[i] = "shuffle"
    if len(dup):
        seeds = dup[: max(1, len(dup) // 8)]
        for j, i in enumerate(dup):
            Xn[i] = Xn[seeds[j % len(seeds)]] + 0.01 * rng.standard_normal(length) * sd
            Yn[i] = Yn[seeds[j % len(seeds)]]
            tag[i] = "dup"
    return Xn, Yn, tag


def inject_rows(X: np.ndarray, labels: np.ndarray, seed: int, n_classes: int, noise_frac: float):
    """Process and tabular rows: a third each of label flip, feature corruption, near-duplicate."""
    rng = np.random.default_rng(seed + 7)
    n = len(labels)
    obs = labels.copy()
    Xn = X.copy()
    tag = np.array(["high"] * n, dtype=object)
    n_low = int(round(noise_frac * n))
    low = rng.permutation(n)[:n_low]
    per = max(1, n_low // 3)
    flip, corr, dup = low[:per], low[per:2 * per], low[2 * per:]
    for i in flip:
        obs[i] = rng.integers(n_classes)
        tag[i] = "flip"
    for i in corr:
        Xn[i] = Xn[i] + rng.standard_normal(X.shape[1]) * 2.0
        tag[i] = "corrupt"
    if len(dup):
        seeds = dup[: max(1, len(dup) // 8)]
        for j, i in enumerate(dup):
            Xn[i] = Xn[seeds[j % len(seeds)]] + 0.01 * rng.standard_normal(X.shape[1])
            tag[i] = "dup"
    return Xn, obs, tag


def flip_validation_labels(labels: np.ndarray, rate: float, modulus: int, max_offset: int) -> np.ndarray:
    """Canonical VAL_NOISE contamination: a Bernoulli(rate) mask from default_rng(10000 + int(100 rate)),
    flipped to (label + U{1, max_offset - 1}) mod ``modulus``. Vision used max_offset 10 and
    modulus max(label) + 1. TEP used max_offset = modulus = number of classes."""
    rng = np.random.default_rng(10_000 + int(rate * 100))
    mask = rng.random(len(labels)) < rate
    out = labels.copy()
    out[mask] = (out[mask] + rng.integers(1, max_offset, mask.sum())) % int(modulus)
    return out


@dataclass
class UnseenPlan:
    """Records chosen for corruption and the mechanism applied to each (index arrays into the pool)."""

    groups: dict[str, np.ndarray] = field(default_factory=dict)

    def tags(self, n: int) -> np.ndarray:
        """Per-record tag: 'high' or the mechanism name."""
        tag = np.array(["high"] * n, dtype=object)
        for name, idx in self.groups.items():
            tag[idx] = name
        return tag


def _pick(seed: int, n: int, noise_frac: float, names: tuple[str, ...],
          eligible: Optional[dict[str, np.ndarray]] = None) -> UnseenPlan:
    """round(noise_frac * n) records from default_rng(seed + 7), split evenly over ``names``.

    ``eligible`` restricts a mechanism to the given records (class-dependent flips need a class with
    a different target). The draw order equals the canonical injectors, so the same records are chosen
    whenever no restriction applies.
    """
    rng = np.random.default_rng(seed + 7)
    n_low = int(round(noise_frac * n))
    low = rng.permutation(n)[:n_low]
    per = max(1, n_low // len(names)) if n_low else 0
    plan = UnseenPlan()
    used: set[int] = set()
    for j, name in enumerate(names):
        part = low[j * per:(j + 1) * per] if j < len(names) - 1 else low[j * per:]
        if eligible is not None and name in eligible:
            allowed = set(int(i) for i in eligible[name]) - used
            part = np.array([i for i in part if int(i) in allowed], dtype=np.int64)
            short = (per if j < len(names) - 1 else len(low) - j * per) - len(part)
            spare = [i for i in rng.permutation(sorted(allowed - set(int(x) for x in low))) if i not in used]
            part = np.concatenate([part, np.asarray(spare[:max(short, 0)], dtype=np.int64)])
        plan.groups[name] = np.asarray(part, dtype=np.int64)
        used.update(int(i) for i in part)
    return plan


def class_flip_target(labels: np.ndarray, n_classes: int, superclass: Optional[np.ndarray] = None,
                      class_map: Optional[dict[int, int]] = None) -> np.ndarray:
    """Class-dependent flip target of every label (-1 where the label keeps its class).

    With ``superclass`` (fine class -> superclass, CIFAR-100) the target is the next fine class of the
    same superclass in cyclic order. With ``class_map`` the map is used. Otherwise c -> (c + 1) mod C.
    """
    y = np.asarray(labels).astype(int)
    if superclass is not None:
        sup = np.asarray(superclass).astype(int)
        nxt = np.empty(len(sup), dtype=int)
        for s in np.unique(sup):
            members = np.sort(np.flatnonzero(sup == s))
            nxt[members] = np.roll(members, -1)
        return nxt[y]
    if class_map is not None:
        return np.array([class_map.get(int(c), -1) for c in y])
    return (y + 1) % int(n_classes)


def unseen_vision_labels(labels: np.ndarray, seed: int, noise_frac: float, n_classes: int,
                         superclass: Optional[np.ndarray] = None,
                         class_map: Optional[dict[int, int]] = None) -> tuple[np.ndarray, np.ndarray, UnseenPlan]:
    """A third each of Gaussian blur, JPEG and class-dependent flip. Returns (labels, tags, plan).

    Blur and JPEG change the images (the track re-encodes them), the flip changes the observed label.
    """
    target = class_flip_target(labels, n_classes, superclass, class_map)
    eligible = {"class_flip": np.flatnonzero((target >= 0) & (target != np.asarray(labels)))}
    plan = _pick(seed, len(labels), noise_frac, ("blur", "jpeg", "class_flip"), eligible)
    obs = np.asarray(labels).copy()
    obs[plan.groups["class_flip"]] = target[plan.groups["class_flip"]]
    return obs, plan.tags(len(labels)), plan


def blur_image(image, sigma: float = 1.5):
    """PIL Gaussian blur with standard deviation ``sigma`` pixels."""
    from PIL import ImageFilter

    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def jpeg_image(image, quality: int = 20):
    """JPEG round trip at ``quality`` through an in-memory buffer."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def unseen_windows(X: np.ndarray, Y: np.ndarray, seed: int, noise_frac: float, scale: float = 1.3,
                   bias: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A third each of multiplicative scale drift, additive bias drift and stuck sensor on (X, Y).

    Each fault acts on the recorded segment (input window followed by target). Scale drift multiplies
    step t of the L + H segment by 1 + (scale - 1) t / (L + H - 1). Bias drift adds bias * sd(X) times
    t / (L + H - 1). Stuck sensor holds the value at a position p drawn in [L / 2, L) for every later
    step. Returns (X, Y, tags).
    """
    n, L = X.shape
    H = Y.shape[1]
    plan = _pick(seed, n, noise_frac, ("scale_drift", "bias_drift", "stuck"))
    rng = np.random.default_rng(seed + 8)
    Xn, Yn = X.copy(), Y.copy()
    ramp = np.linspace(0.0, 1.0, L + H)
    sd = float(X.std()) + 1e-8
    for name, idx in plan.groups.items():
        for i in idx:
            seg = np.concatenate([Xn[i], Yn[i]])
            if name == "scale_drift":
                seg = seg * (1.0 + (float(scale) - 1.0) * ramp)
            elif name == "bias_drift":
                seg = seg + float(bias) * sd * ramp
            else:
                p = int(rng.integers(L // 2, L))
                seg[p:] = seg[p]
            Xn[i], Yn[i] = seg[:L], seg[L:]
    return Xn, Yn, plan.tags(n)


def unseen_rows(X: np.ndarray, labels: np.ndarray, seed: int, n_classes: int, noise_frac: float,
                stuck_frac: float = 0.25, class_map: Optional[dict[int, int]] = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Half stuck features, half class-dependent flips. Returns (X, labels, tags).

    A stuck record has round(stuck_frac * d) columns (at least one, drawn per record) set to the pool
    median of each column. A flipped record moves to class_map[c] (binary default {1: 0}) or to
    (c + 1) mod C.
    """
    n, d = X.shape
    if class_map is None and int(n_classes) == 2:
        class_map = {1: 0}
    target = class_flip_target(labels, n_classes, class_map=class_map)
    eligible = {"class_flip": np.flatnonzero((target >= 0) & (target != np.asarray(labels)))}
    plan = _pick(seed, n, noise_frac, ("stuck_feature", "class_flip"), eligible)
    rng = np.random.default_rng(seed + 8)
    Xn = X.copy()
    median = np.median(X, axis=0)
    width = max(1, int(round(float(stuck_frac) * d)))
    for i in plan.groups["stuck_feature"]:
        cols = rng.choice(d, size=width, replace=False)
        Xn[i, cols] = median[cols]
    obs = np.asarray(labels).copy()
    obs[plan.groups["class_flip"]] = target[plan.groups["class_flip"]]
    return Xn, obs, plan.tags(n)


def normalize_val_noise(kind: str) -> str:
    """Canonical name of a validation noise kind (class_prior_shift is accepted for prior_shift)."""
    kind = VAL_NOISE_ALIASES.get(kind, kind)
    if kind not in VAL_NOISE_KINDS:
        raise ValueError(f"validation noise kind must be one of {VAL_NOISE_KINDS}, got {kind!r}")
    return kind


def contaminate_validation(kind: str, labels: np.ndarray, rate: float, seed: int, *, modulus: int,
                           max_offset: int, natural: Optional[np.ndarray] = None
                           ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Validation labels after contamination, the validation positions kept, and a summary.

    symmetric: ``flip_validation_labels`` (the canonical VAL_NOISE rule). natural: the human labels
    ``natural`` of the same records (``rate`` is not used). prior_shift: a seeded half of the classes
    (default_rng(seed + 10007)) is the minority, and each minority record is replaced with probability
    ``rate`` by a uniformly drawn majority record, so the validation size is unchanged and the class
    priors shift. Positions index the original validation list (duplicates for prior_shift).
    """
    kind = normalize_val_noise(kind)
    y = np.asarray(labels)
    positions = np.arange(len(y))
    if kind == "none" or (kind in ("symmetric", "prior_shift") and rate <= 0):
        return y.copy(), positions, {"kind": kind, "rate": float(rate), "changed": 0}
    if kind == "symmetric":
        out = flip_validation_labels(y, rate, modulus, max_offset)
        return out, positions, {"kind": kind, "rate": float(rate), "changed": int(np.sum(out != y))}
    if kind == "natural":
        if natural is None:
            raise ValueError("natural validation noise needs human labels for the validation records")
        out = np.asarray(natural).astype(y.dtype)
        return out, positions, {"kind": kind, "rate": float(np.mean(out != y)), "changed": int(np.sum(out != y))}
    rng = np.random.default_rng(seed + 10_007)
    classes = np.unique(y)
    minority = set(int(c) for c in rng.permutation(classes)[: len(classes) // 2])
    is_minority = np.array([int(c) in minority for c in y])
    majority_pos = np.flatnonzero(~is_minority)
    replace = is_minority & (rng.random(len(y)) < float(rate))
    positions = positions.copy()
    positions[replace] = rng.choice(majority_pos, size=int(replace.sum()), replace=True)
    out = y[positions]
    prior = {int(c): float(np.mean(out == c)) for c in classes}
    return out, positions, {"kind": kind, "rate": float(rate), "changed": int(replace.sum()),
                            "minority_classes": sorted(minority), "class_prior": prior}
