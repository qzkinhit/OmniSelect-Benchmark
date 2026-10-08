"""Track interface: TrackConfig, TaskData, Signals and the Track base class.

A track supplies data loading with corruption injection, the three signal channels, one
downstream learner, and strategy overrides where a method needs track-specific inputs. The
driver in tracks/common/experiment.py calls these hooks in a fixed order and writes the run
record. TrackConfig holds the task constants. OmniSelectConfig holds the controller constants.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from omniselect.config.config import set_dotted
from omniselect.core.adjudication.splits import ValidationSplits
from omniselect.core.selection.cooperative import GATE_SCORES, SELECTORS, parse_rho
from omniselect.core.selection.fusion_grid import FusionGrid


@dataclass
class TrackConfig:
    """Task constants shared by every track. Subclasses add learner and data fields."""

    track: str = ""
    dataset: str = ""
    learner: str = ""
    seed: int = 0
    pool_n: int = 0
    val_n: int = 0
    test_n: int = 0
    budget_frac: float = 0.5
    noise_frac: float = 0.4
    knn: int = 15
    lam: float = 0.5                # diversity strength of the Fixed-fusion row
    auth_q: float = 0.25            # authenticity prefilter quantile of Fixed fusion and the grid
    w_infl: float = 0.5             # influence weight of the Fixed-fusion console
    influence_ref_n: int = 400      # canonical size of the influence reference sample
    influence_within_class: bool = False  # v2.3: rank the influence of each record inside its observed class
    dsdm_runs: int = 12
    dmf_rounds: int = 6
    methods: tuple[str, ...] = ()   # main-table rows. 'full' and 'mmds_adapt' are allowed
    grid_weights: Optional[tuple[tuple[float, ...], ...]] = None
    grid_q: tuple[float, ...] = (0.0, 0.25)
    grid_lam: tuple[float, ...] = (0.0, 0.25, 0.6)
    grid_protocols: tuple[str, ...] = ("canonical", "v2")
    coop_scores: tuple[str, ...] = ("authenticity", "cleanliness")   # v2.2 cooperative grid: gate scores
    coop_rhos: tuple[str, ...] = ("1.15", "1.3", "1.5")                  # gate widths (keep ratios)
    coop_selectors: tuple[str, ...] = ("herding", "kcenter")             # selectors inside the gate
    coop_ungated: tuple[str, ...] = ("herding", "kcenter")               # selectors of the rho=none cells
    device: str = "auto"
    selection_device: str = "cpu"   # herding, k-center, coreset: cpu (numpy, scikit-learn) | cuda | mps | torch_cpu
    data_root: str = "data"
    smoke: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Plain dict written to config.json."""
        return dataclasses.asdict(self)

    def with_overrides(self, **overrides: Any) -> "TrackConfig":
        """Copy with fields replaced. Values given as strings are cast to the field type."""
        out = dataclasses.replace(self)
        for key, value in overrides.items():
            if key == "methods" and isinstance(value, str):
                value = tuple(v for v in value.split(",") if v)
            set_dotted(out, key, value)
        return out

    def fusion_grid(self) -> FusionGrid:
        """Fusion grid of this task (weights None means the default simplex)."""
        weights = [tuple(w) for w in self.grid_weights] if self.grid_weights is not None else None
        return FusionGrid(weights=weights, q_grid=tuple(self.grid_q), lam_grid=tuple(self.grid_lam))

    def cooperative_grid(self) -> list[tuple[Optional[str], Optional[float], str]]:
        """(gate score, gate width, selector) of every cooperative grid cell: scores outer, then widths,
        then selectors, followed by the ungated cells (score and width None, vetoes only)."""
        unknown = [s for s in self.coop_selectors + self.coop_ungated if s not in SELECTORS]
        unknown += [s for s in self.coop_scores if s not in GATE_SCORES]
        if unknown:
            raise ValueError(f"cooperative grid entries must come from {SELECTORS} and {GATE_SCORES}, got {unknown}")
        cells: list[tuple[Optional[str], Optional[float], str]] = []
        for score in self.coop_scores:
            for rho in self.coop_rhos:
                if parse_rho(rho) is None:
                    raise ValueError("coop_rhos lists gated widths, the ungated cells come from coop_ungated")
                cells.extend((score, parse_rho(rho), sel) for sel in self.coop_selectors)
        return cells + [(None, None, sel) for sel in self.coop_ungated]


@dataclass
class TaskData:
    """Loaded task: ids, tags, splits and the arrays the track's learner and signals read."""

    pool_ids: list
    val_ids: list
    test_ids: list
    tags: np.ndarray
    splits: ValidationSplits
    budget: int
    utility: str
    records: list
    arrays: dict[str, Any] = field(default_factory=dict)
    labels: Optional[np.ndarray] = None
    labels_clean: Optional[np.ndarray] = None
    n_classes: Optional[int] = None
    blocks: Optional[dict[str, Any]] = None
    s0: float = 1.0
    block_steps: int = 0
    budget_cut: Optional[Callable[[list], list]] = None
    mechanisms: Optional[list] = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Pool size."""
        return len(self.pool_ids)


@dataclass
class Signals:
    """Channel scores and the inputs of the selection context."""

    auth: np.ndarray
    influence: np.ndarray
    redundancy: np.ndarray
    features: np.ndarray
    reference_ids: list
    reference_source: str
    proba: Optional[np.ndarray] = None
    extras: dict[str, Any] = field(default_factory=dict)
    saved: dict[str, np.ndarray] = field(default_factory=dict)
    secs: float = 0.0
    alignment: Optional[np.ndarray] = None    # fourth channel of protocol v2.1 (raw cosine)

    def channels(self) -> np.ndarray:
        """Stack (authenticity, influence, redundancy[, alignment]) with shape (3, n) or (4, n)."""
        rows = [self.auth, self.influence, self.redundancy]
        if self.alignment is not None:
            rows.append(self.alignment)
        return np.stack(rows, axis=0)


class Track:
    """Base class. Subclasses set the class attributes and implement the hooks."""

    name: str = ""
    utility: str = ""
    config_cls: type = TrackConfig
    datasets: dict[str, dict[str, Any]] = {}
    default_learner: str = ""
    canonical_sorted_scoring: bool = False
    sorted_training: bool = True        # reported fits train on sorted ids under the paired RNG

    def config(self, dataset: str, learner: Optional[str] = None, seed: int = 0, smoke: bool = False,
               **overrides: Any) -> TrackConfig:
        """TrackConfig of ``dataset`` with the canonical constants, then ``overrides``."""
        if dataset not in self.datasets:
            raise KeyError(f"unknown dataset {dataset!r} for track {self.name}; known: {sorted(self.datasets)}")
        base = dict(self.datasets[dataset])
        if smoke:
            base.update(base.pop("smoke", {}))
        else:
            base.pop("smoke", None)
        cfg = self.config_cls(track=self.name, dataset=dataset, learner=learner or self.default_learner,
                              seed=seed, smoke=smoke, **base)
        return cfg.with_overrides(**overrides) if overrides else cfg

    def omni_overrides(self, protocol: str, tcfg: TrackConfig) -> dict[str, Any]:
        """Track-specific OmniSelectConfig overrides applied on top of the protocol preset."""
        return {}

    def load(self, tcfg: TrackConfig, ocfg: Any) -> TaskData:
        """Load the data, inject corruption, build the validation splits."""
        raise NotImplementedError

    def signals(self, data: TaskData, tcfg: TrackConfig, ocfg: Any) -> Signals:
        """Authenticity, influence and redundancy per pool record."""
        raise NotImplementedError

    def learner(self, data: TaskData, tcfg: TrackConfig, ocfg: Any) -> Any:
        """Downstream learner implementing tracks.common.downstream.Learner."""
        raise NotImplementedError

    def alignment(self, data: TaskData, sig: Signals, tcfg: TrackConfig, ocfg: Any) -> Optional[dict[str, Any]]:
        """Alignment channel of protocol v2.1: {"scores", "phi", "err", "source"} or None when undefined."""
        return None

    def cooperative_inputs(self, data: TaskData, sig: Signals, tcfg: TrackConfig, ocfg: Any) -> dict[str, Any]:
        """Inputs of the cooperative candidates and the learned cleanliness score (protocol v2.2).

        ``veto_features`` and ``metric`` (cosine or euclidean) give the geometry of the vetoes and of
        the neighbour features, ``con_veto_features`` the V_con records in the same representation,
        ``selector_features`` the rows herding and k-center read, and ``cleanliness`` the per-track
        feature inputs of core/signals/cleanliness.py (``kind`` plus arrays). Token-budgeted tracks add
        ``budget_gate`` and ``token_orders``. Tracks without V_con features return no ``cleanliness``.
        """
        raise NotImplementedError(f"track {self.name} defines no cooperative inputs")

    def strategy_overrides(self, data: TaskData, sig: Signals, tcfg: TrackConfig, ocfg: Any) -> dict[str, Callable]:
        """Track-specific strategy functions that replace registry entries of the same name."""
        return {}
