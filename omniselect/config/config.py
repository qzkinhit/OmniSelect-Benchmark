"""OmniSelectConfig: every controller constant, with the canonical, v2, v2_1 and v2_2 protocol presets.

``v2_2`` is the protocol of the paper. ``v2`` sets three validation splits (construction, ranking,
confirmation), the train-once fit cache, scoring at the reported schedule, the margin gate on the
ranking split with a confirmation audit, the recorded headroom precheck, the unified portfolio and
an influence reference drawn from the construction split. ``v2_1`` is ``v2`` plus the alignment
channel, the four-channel fusion grid, the alignment_only member and the solver challengers.
``v2_2`` is ``v2`` plus the learned cleanliness score, the cooperative candidates, the coop_herding
member and the clean_top challenger. ``canonical`` reproduces the earlier two-split controller of
the 2026-07 runners (two validation halves, 1.5% margin on the ranking half, successive halving on
the construction half, no fit cache). Each protocol difference is one field, so an ablation changes
one field (``--set key=value`` on the driver). docs/DESIGN.md explains the choices.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Any

PROTOCOLS = ("canonical", "v2", "v2_1", "v2_2")
GATE_KINDS = ("margin", "argmax", "lcb", "bootstrap", "eprocess")
MEMBERSHIPS = ("canonical", "unified", "unified_v21", "unified_v22", "canonical_20260716", "canonical_ettm1_20260716")


@dataclass
class SplitConfig:
    """How the clean validation set is divided.

    mode: ``two_way`` gives con and rank halves (canonical V1 and V2), ``three_way`` gives
    con, rank and conf in ``fractions``, ``rank_only`` puts every validation record in rank
    (canonical native ResNet protocol, validation argmax without construction split).
    """

    mode: str = "two_way"
    fractions: tuple[float, float, float] = (0.5, 0.5, 0.0)
    perm_seed_offset: int = 41          # canonical runners: default_rng(seed + 41).permutation
    time_blocks: bool = False           # forecasting: con, rank, conf, test are contiguous blocks
    text_sizes: tuple[int, int, int, int] = (400, 600, 500, 500)  # v2 text: con, rank, conf, report


@dataclass
class CacheConfig:
    """Train-once cache keyed by (sha of sorted subset ids, learner fidelity key)."""

    by_subset_hash: bool = False


@dataclass
class FidelityConfig:
    """Scoring fits versus the reported fit.

    Canonical scoring fits use the shorter schedule of each runner (CLIP probe 150 iterations
    against 300 reported, DLinear 40 epochs against 60) and stage-specific seeds. With
    ``scoring_equals_reported`` every fit uses the reported schedule and one seed.
    """

    scoring_equals_reported: bool = False


@dataclass
class GateConfig:
    """Decision rule between the reference and the challengers.

    ``margin`` and ``argmax`` compare ranking utilities. ``lcb``, ``bootstrap`` and ``eprocess``
    use per-unit paired differences on ``split``. ``pair_split`` is the split on which references
    and challengers are ranked before the gate is applied. Bootstrap and eprocess test all
    ``k_challengers`` and elect the first that passes. ``reading_order`` sets how the bootstrap
    resamples and the e-process reads units (core/gates/paired.py). With ``audit`` the controller
    records the paired gain, a bootstrap 90% interval and the e-value of the reference against the
    top-K ranking challengers on ``audit_split`` without changing the election.
    """

    kind: str = "margin"
    split: str = "rank"
    pair_split: str = "rank"
    margin_frac: float = 0.015          # canonical switch margin, fraction of |u_rank(reference)|
    delta: float = 0.05
    eps: float = 0.0                    # e-process null: mean paired difference <= -eps
    reading_order: str = "stratified"   # stratified: class, pos-neg pair or domain draws | uniform: one shuffle
    audit: bool = False                 # v2: e-value and bootstrap interval of the top-K challengers on V_conf
    audit_split: str = "conf"
    n_boot: int = 1000
    p_beat_min: float = 0.9
    k_challengers: int = 1              # ordered family size tested by lcb, bootstrap, eprocess
    text_clip: float = 1.0              # text: per-record NLL improvement clipped to [-C, C]
    ts_block_steps: int = 0             # forecasting certificate unit length. 0 means L + H


@dataclass
class PrecheckConfig:
    """Headroom precheck on the ranking split.

    Records u_rank(full) - u_rank(random) and u_rank(best reference) - u_rank(random) on the bounded
    surrogate. With ``decide`` it elects random when the headroom is below the threshold:
    ``threshold_multiplier`` times r1 = sqrt(log(1/delta) / (2 n_eff)) for ``radius``, or
    ``threshold`` times the absolute mean surrogate of random for ``relative``. Every preset leaves
    ``decide`` off.
    """

    enabled: bool = False
    decide: bool = False                 # false: record the headroom only, never change the election
    threshold_kind: str = "radius"       # radius: multiplier * r1 | relative: threshold * |mean(random)|
    threshold_multiplier: float = 2.0
    threshold: float = 0.03              # used when threshold_kind == "relative"


@dataclass
class PortfolioConfig:
    """Membership table used for the reference-eligible strategies (core/portfolio/membership.py)."""

    membership: str = "canonical"       # one of MEMBERSHIPS


@dataclass
class InfluenceConfig:
    """Where the influence reference sample comes from."""

    reference: str = "pool_clean_tag"   # canonical | "v_con"


@dataclass
class ScreeningConfig:
    """Successive halving of the fusion grid on the construction split."""

    kind: str = "full"                  # full | low_fidelity | off
    sh_keep: int = 4
    low_fidelity_fraction: float = 0.4  # low_fidelity: share of each selection trained at the screen stage


@dataclass
class SynthesisConfig:
    """Challenger-only candidates built on the construction split."""

    consensus: bool = True              # vote over the top ``consensus_top`` candidates
    consensus_top: int = 3
    consensus_diversity_lam: float = 0.0          # > 0 adds the diversity-regularized vote (historical, off)
    consensus_complementarity_threshold: float = 0.0  # > 0 adds the overlap-gated vote (historical, off)
    coordinate_ascent: bool = True
    ca_rounds: int = 2
    ca_step: float = 0.15
    policy_search: bool = False         # historical group-relative search (negative result)
    held_half: bool = False             # coordinate ascent fits on one half of V_con, steps checked on the other
    infomax_solver: bool = False        # v2.1: one "<cell> solver=infomax" challenger per screening finalist


@dataclass
class FixesConfig:
    """Behaviour corrections that the canonical protocol must not apply.

    ``fixed_fusion_gate``: ``sentinel`` reproduces the canonical Fixed-fusion row, which writes
    -1e9 into gated records and min-max normalizes the whole vector, so the passing records get
    near-identical importance. ``finite_subset`` runs the budget selector on the passing records
    only, as the controller's fusion cells do.
    """

    fixed_fusion_gate: str = "sentinel"
    # ``canonical_100`` draws flipped vision labels from range(100) on every dataset, as the
    # 2026-07 vision runner did (on CIFAR-10 most flips land on classes absent from validation
    # and test). ``n_classes`` draws them from the dataset's own classes.
    vision_flip_range: str = "canonical_100"
    # ``zero`` keeps the canonical Density row, whose Gumbel keys used seed 0 on every run seed
    # in the vision, forecasting, TEP and tabular runners. ``run_seed`` passes the run seed.
    density_seed: str = "zero"


@dataclass
class RobustnessConfig:
    """Robustness conditions (tracks/common/injection.py). The defaults are the canonical values.

    ``mechanism_set``: canonical corruption or the unseen mechanisms. ``val_noise_kind``: none,
    symmetric, natural (CIFAR-N human labels on the validation records) or prior_shift, with
    ``val_noise_rate``. ``injection_ratio``: pool corruption fraction (negative keeps the track
    default). Which track reads which setting is ``ROBUSTNESS_SUPPORT`` in tracks/common/experiment.py.
    """

    mechanism_set: str = "canonical"    # canonical | unseen
    val_noise_kind: str = "none"        # none | symmetric | natural | prior_shift
    val_noise_rate: float = 0.0
    injection_ratio: float = -1.0       # negative: use the track default noise fraction


@dataclass
class SignalsConfig:
    """Channels of the fusion. ``alignment`` adds the fourth channel of protocol v2.1 (core/signals/alignment.py)."""

    alignment: bool = False


@dataclass
class CoverageConfig:
    """Similarity of the budget selector's redundancy penalty: feature | gradient | concat."""

    space: str = "feature"


@dataclass
class CooperativeConfig:
    """Cooperative candidates of protocol v2.2 (core/selection/cooperative.py).

    A gate keeps the ceil(rho k) records with the highest gate score (authenticity or the learned
    cleanliness), an outlier veto drops records whose robust z of the distance to the ``knn``-th
    neighbour exceeds ``outlier_z``, a duplicate veto keeps the cleanest member of each group linked
    below ``duplicate_ratio`` times the median nearest-neighbour distance, and herding or k-center
    picks k records inside. ``enabled`` adds the challenger grid of TrackConfig (``coop_scores`` x
    ``coop_rhos`` x ``coop_selectors`` plus the ungated ``coop_ungated`` cells), screened on V_con down
    to ``screen_keep`` finalists. The reference member coop_herding uses the authenticity gate with
    ``reference_rho`` and herding.
    """

    enabled: bool = False
    screen_keep: int = 3
    reference_rho: float = 1.3
    knn: int = 10
    outlier_veto: bool = True
    outlier_z: float = 3.0
    duplicate_veto: bool = True
    duplicate_ratio: float = 0.1


@dataclass
class CleanlinessConfig:
    """Learned cleanliness score of protocol v2.2 (core/signals/cleanliness.py).

    ``folds`` pool folds for the cross-fitted label model (LogisticRegression, ``logreg_iter``
    iterations) or ridge residual (``ridge_alpha``) and for the cross-fitted classifier scores.
    The classifier is HistGradientBoostingClassifier(``max_iter``, ``learning_rate``, ``max_leaf_nodes``).
    """

    folds: int = 5
    logreg_iter: int = 300
    ridge_alpha: float = 1.0
    max_iter: int = 200
    learning_rate: float = 0.05
    max_leaf_nodes: int = 15


@dataclass
class StoreConfig:
    """What the run record keeps per candidate.

    ``per_unit_scope``: ``all`` writes per_unit/<split>.npz for every candidate. ``finalists`` writes
    them for reference-eligible candidates, screening finalists, synthesized challengers and method
    rows, and keeps only selection.npz and scores.json (utilities and standard metrics computed at
    run time) for grid cells removed by screening.
    """

    per_unit_scope: str = "all"         # all | finalists


@dataclass
class OmniSelectConfig:
    """All controller parameters. Track constants (pool sizes, learners) live in TrackConfig."""

    protocol: str = "canonical"
    seed: int = 0
    paired_rng: bool = True             # reset RNGs per strategy and per fit stage (canonical batches)
    splits: SplitConfig = field(default_factory=SplitConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    fidelity: FidelityConfig = field(default_factory=FidelityConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    precheck: PrecheckConfig = field(default_factory=PrecheckConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    influence: InfluenceConfig = field(default_factory=InfluenceConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    fixes: FixesConfig = field(default_factory=FixesConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    signals: SignalsConfig = field(default_factory=SignalsConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    cooperative: CooperativeConfig = field(default_factory=CooperativeConfig)
    cleanliness: CleanlinessConfig = field(default_factory=CleanlinessConfig)
    report_all_candidates: bool = True  # reported fit and test score for every leaderboard entry
    save_models: str = "none"           # none | elected | all

    @classmethod
    def preset(cls, protocol: str, **overrides: Any) -> "OmniSelectConfig":
        """Return the canonical, v2, v2_1 or v2_2 preset with dotted-key ``overrides`` applied."""
        if protocol not in PROTOCOLS:
            raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
        cfg = cls(protocol=protocol)
        if protocol in ("v2", "v2_1", "v2_2"):
            cfg.splits = SplitConfig(mode="three_way", fractions=(0.3, 0.5, 0.2), time_blocks=True)
            cfg.cache = CacheConfig(by_subset_hash=True)
            cfg.fidelity = FidelityConfig(scoring_equals_reported=True)
            # The deployed gate is the margin rule on V_rank. V_conf is the audit split (docs/DESIGN.md).
            cfg.gate = GateConfig(kind="margin", split="rank", pair_split="rank", k_challengers=3, audit=True)
            cfg.precheck = PrecheckConfig(enabled=True, decide=False)
            cfg.portfolio = PortfolioConfig(membership="unified")
            cfg.influence = InfluenceConfig(reference="v_con")
            cfg.screening = ScreeningConfig(kind="full", sh_keep=4)
            cfg.synthesis = SynthesisConfig()
            cfg.fixes = FixesConfig(fixed_fusion_gate="finite_subset", vision_flip_range="n_classes",
                                    density_seed="run_seed")
        if protocol == "v2_1":
            cfg.signals = SignalsConfig(alignment=True)
            cfg.coverage = CoverageConfig(space="feature")
            cfg.synthesis = SynthesisConfig(infomax_solver=True)
            cfg.portfolio = PortfolioConfig(membership="unified_v21")
        if protocol == "v2_2":
            cfg.cooperative = CooperativeConfig(enabled=True)
            cfg.portfolio = PortfolioConfig(membership="unified_v22")
        return cfg.with_overrides(**overrides) if overrides else cfg

    def with_overrides(self, **overrides: Any) -> "OmniSelectConfig":
        """Return a copy with dotted keys replaced, for example ``{"gate.kind": "lcb"}``."""
        out = copy.deepcopy(self)
        for dotted, value in overrides.items():
            set_dotted(out, dotted.replace("__", "."), value)
        out.validate()
        return out

    def validate(self) -> None:
        """Raise ValueError for values outside the documented options."""
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}")
        if self.gate.kind not in GATE_KINDS:
            raise ValueError(f"gate.kind must be one of {GATE_KINDS}")
        if self.splits.mode not in ("two_way", "three_way", "rank_only"):
            raise ValueError("splits.mode must be two_way, three_way or rank_only")
        if self.screening.kind not in ("full", "low_fidelity", "off"):
            raise ValueError("screening.kind must be full, low_fidelity or off")
        if self.portfolio.membership not in MEMBERSHIPS:
            raise ValueError(f"portfolio.membership must be one of {MEMBERSHIPS}")
        if self.influence.reference not in ("pool_clean_tag", "v_con"):
            raise ValueError("influence.reference must be pool_clean_tag or v_con")
        if self.fixes.fixed_fusion_gate not in ("sentinel", "finite_subset"):
            raise ValueError("fixes.fixed_fusion_gate must be sentinel or finite_subset")
        if self.fixes.vision_flip_range not in ("canonical_100", "n_classes"):
            raise ValueError("fixes.vision_flip_range must be canonical_100 or n_classes")
        if self.fixes.density_seed not in ("zero", "run_seed"):
            raise ValueError("fixes.density_seed must be zero or run_seed")
        if self.robustness.mechanism_set not in ("canonical", "unseen"):
            raise ValueError("robustness.mechanism_set must be canonical or unseen")
        if self.robustness.val_noise_kind == "class_prior_shift":
            self.robustness.val_noise_kind = "prior_shift"
        if self.robustness.val_noise_kind not in ("none", "symmetric", "natural", "prior_shift"):
            raise ValueError("robustness.val_noise_kind must be none, symmetric, natural or prior_shift")
        if not 0.0 <= self.robustness.val_noise_rate < 1.0 or self.robustness.injection_ratio >= 1.0:
            raise ValueError("robustness.val_noise_rate must lie in [0, 1) and injection_ratio below 1")
        if self.coverage.space not in ("feature", "gradient", "concat"):
            raise ValueError("coverage.space must be feature, gradient or concat")
        if self.store.per_unit_scope not in ("all", "finalists"):
            raise ValueError("store.per_unit_scope must be all or finalists")
        if self.save_models not in ("none", "elected", "all"):
            raise ValueError("save_models must be none, elected or all")
        if self.gate.reading_order not in ("stratified", "uniform"):
            raise ValueError("gate.reading_order must be stratified or uniform")
        if not 0.0 <= self.gate.eps < 1.0:
            raise ValueError("gate.eps must lie in [0, 1)")
        if not 0.0 < self.gate.delta < 1.0:
            raise ValueError("gate.delta must lie in (0, 1)")
        if self.gate.k_challengers < 1:
            raise ValueError("gate.k_challengers must be at least 1")
        co, cl = self.cooperative, self.cleanliness
        if co.screen_keep < 2 or co.knn < 1 or co.reference_rho < 1.0:
            raise ValueError("cooperative.screen_keep must be at least 2, knn at least 1, reference_rho at least 1")
        if co.outlier_z <= 0.0 or not 0.0 < co.duplicate_ratio < 1.0:
            raise ValueError("cooperative.outlier_z must be positive and duplicate_ratio in (0, 1)")
        if cl.folds < 2 or cl.logreg_iter < 1 or cl.max_iter < 1 or cl.max_leaf_nodes < 2:
            raise ValueError("cleanliness.folds and max_leaf_nodes must be at least 2, the iteration counts positive")
        if cl.ridge_alpha < 0.0 or cl.learning_rate <= 0.0:
            raise ValueError("cleanliness.ridge_alpha must be >= 0 and learning_rate positive")
        if abs(sum(self.splits.fractions) - 1.0) > 1e-9:
            raise ValueError("splits.fractions must sum to 1")

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form written to config.json."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OmniSelectConfig":
        """Inverse of ``to_dict``. Keys of earlier schema versions are mapped by LEGACY_KEYS. Unknown keys raise."""
        cfg = cls()
        for key, value in flatten(data).items():
            if key in LEGACY_KEYS:
                key, value = LEGACY_KEYS[key](value)
                if key is None:
                    continue
            set_dotted(cfg, key, value)
        cfg.validate()
        return cfg


# Fields renamed or removed after run records were written. None drops the key when a record is read.
LEGACY_KEYS: dict[str, Any] = {
    "robustness.injection": lambda value: ("robustness.mechanism_set", value),
    "gate.eprocess_weighting": lambda value: (None, value),
    "synthesis.bayes": lambda value: (None, value),
    "synthesis.bayes_probes": lambda value: (None, value),
}


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Map a nested dict to dotted keys."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, name + "."))
        else:
            out[name] = value
    return out


def set_dotted(obj: Any, dotted: str, value: Any) -> None:
    """Assign ``value`` to the dataclass field named by ``dotted``, casting to the field type."""
    parts = dotted.split(".")
    target = obj
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise KeyError(f"unknown config section {dotted!r}")
        target = getattr(target, part)
    leaf = parts[-1]
    if not dataclasses.is_dataclass(target) or leaf not in {f.name for f in dataclasses.fields(target)}:
        raise KeyError(f"unknown config field {dotted!r}")
    current = getattr(target, leaf)
    setattr(target, leaf, _cast(value, current))


def _cast(value: Any, current: Any) -> Any:
    """Cast a CLI string or JSON value to the type of the field's current value."""
    if isinstance(current, bool):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("1", "true", "yes", "on"):
                return True
            if lowered in ("0", "false", "no", "off"):
                return False
            raise ValueError(f"cannot read {value!r} as a boolean")
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, tuple):
        if isinstance(value, str):
            value = [v for v in value.replace(",", " ").split() if v]
        return tuple(type(current[0])(v) if current else v for v in value)
    return value


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse CLI ``key=value`` pairs (``--set gate.kind=lcb``) into an overrides dict."""
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"override {pair!r} must have the form key=value")
        key, value = pair.split("=", 1)
        out[key.strip()] = value.strip()
    return out
