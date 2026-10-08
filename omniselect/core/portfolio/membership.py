"""Portfolio membership tables per track, with a reason string for every exclusion.

``canonical`` lists the reference strategies in the order each canonical (2026-07) runner passed
them to the controller. ``unified`` is the table of protocol v2, shared by all tracks.
``unified_v21`` adds alignment_only, and ``unified_v22`` adds coop_herding (reference) and
clean_top (challenger). The function ``membership`` returns one row per known strategy with its
role (reference or challenger), whether it is included on the track, and why an excluded strategy
is excluded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

TRACKS = ("vision", "native", "timeseries", "process", "tabular", "text")

CANONICAL_REFERENCES: dict[str, tuple[str, ...]] = {
    "vision": (
        "coreset", "auth_only", "auth2_only", "influence_only", "mmdataselect", "herding", "kcenter",
        "el2n", "grand", "ccs", "semdedup", "density", "quadmix_pub", "dmf", "dmf_pub", "d4", "dsdm",
        "random", "auth_bottom",
    ),
    "timeseries": (
        "mmdataselect", "auth2_only", "herding", "kcenter", "semdedup", "density", "quadmix_pub",
        "dmf", "dmf_pub", "d4", "dsdm", "random", "auth_bottom",
    ),
    "process": (
        "coreset", "auth_only", "auth2_only", "auth3_only", "influence_only", "mmdataselect", "herding",
        "kcenter", "el2n", "grand", "ccs", "semdedup", "density", "quadmix_pub", "dmf", "dmf_pub", "d4",
        "dsdm", "random", "auth_bottom",
    ),
    "tabular": (
        "mmdataselect", "auth2_only", "auth3_only", "herding", "kcenter", "el2n", "grand", "ccs",
        "semdedup", "density", "quadmix_pub", "dmf", "dmf_pub", "tabpfn_coreset", "tabpfn_margin",
        "tabpfn_hybrid", "d4", "dsdm", "random", "auth_bottom",
    ),
    "native": (
        "random", "el2n", "grand", "ccs", "auth_only", "herding", "kcenter", "coreset", "semdedup",
        "density", "quadmix_pub", "influence_only", "mmdataselect", "dmf_pub",
    ),
    "text": (
        "random", "influence_only", "coverage_text", "herding_text", "density_text", "quadmix_pub",
        "dmf_pub",
    ),
}
CANONICAL_CHALLENGERS: dict[str, tuple[str, ...]] = {"text": ("fixed_fusion",)}

# Earlier portfolio generations behind some canonical 2026-07 records. The 2026-07-16 paired
# batch (CIFAR-100, TEP21, Electricity, ETTh1 seeds 0 and 1) passed the QuaDMix-style proxy
# before quadmix_pub. ETTm1 (batch 2026-07-16T1532) had the proxy and neither published transfer.
# The proxy of that time produced duplicate ids. The registered 'quadmix' is the later
# uniqueness-hardened version, so these modes replay the portfolio shape, not its exact output.
HISTORICAL_REFERENCES: dict[str, dict[str, tuple[str, ...]]] = {
    "canonical_20260716": {
        track: tuple(
            x for name in names for x in (("quadmix", name) if name == "quadmix_pub" else (name,))
        )
        for track, names in CANONICAL_REFERENCES.items()
        if track in ("vision", "timeseries", "process", "tabular")
    },
    "canonical_ettm1_20260716": {
        "timeseries": (
            "mmdataselect", "auth2_only", "herding", "kcenter", "semdedup", "density", "quadmix", "dmf",
            "d4", "dsdm", "random", "auth_bottom",
        ),
    },
}

UNIFIED_ORDER: tuple[str, ...] = (
    "random", "auth_only", "auth2_only", "auth3_only", "influence_only", "coreset", "kcenter",
    "mmdataselect", "herding", "density", "semdedup", "dsdm", "d4", "dmf_pub", "quadmix_pub", "el2n",
    "grand", "ccs", "tabpfn_coreset", "tabpfn_margin", "tabpfn_hybrid", "glister", "gradmatch",
    "infomax", "auth_bottom",
)
UNIFIED_TEXT_ORDER: tuple[str, ...] = (
    "random", "auth_only", "influence_only", "coverage_text", "herding_text", "density_text", "dsir",
    "zip", "quadmix_pub", "dmf_pub", "fixed_fusion", "less", "auth_bottom",
)
# Unified members that enter as challenger-only candidates (they must pass the gate, never the reference).
UNIFIED_CHALLENGERS: dict[str, tuple[str, ...]] = {"text": ("fixed_fusion",)}
# unified_v22 members that read V_con (the learned cleanliness score), so they enter as challengers.
V22_CHALLENGERS: tuple[str, ...] = ("clean_top",)

_CLASSIFICATION_ONLY = (
    "classification-only definition (error norm or gradient norm of a labelled class probability); "
    "undefined for regression windows and autoregressive text"
)
WITHDRAWN: dict[str, str] = {
    "quadmix": "withdrawn: QuaDMix-style proxy produced duplicate ids (PROTOCOL_INVALID_DUPLICATE_IDS); "
               "quadmix_pub is the QuaDMix row",
    "dmf": "removed: non-published dynamic-fusion proxy; dmf_pub is the DMF row",
}

EXCLUSIONS: dict[tuple[str, str], str] = {}
for _track in ("timeseries", "text"):
    for _name in ("el2n", "grand", "ccs"):
        EXCLUSIONS[(_track, _name)] = _CLASSIFICATION_ONLY
for _track in TRACKS:
    if _track != "tabular":
        for _name in ("tabpfn_coreset", "tabpfn_margin", "tabpfn_hybrid"):
            EXCLUSIONS[(_track, _name)] = (
                "Tab-AICL acquisition rules read TabPFN in-context probabilities, which only the tabular track has"
            )
    if _track != "text":
        EXCLUSIONS[(_track, "less")] = "LoRA gradient features of a language model; defined on the text track only"
for _name in ("glister", "gradmatch"):
    EXCLUSIONS[("tabular", _name)] = (
        "TabPFN conditions on the context set in one forward pass and has no training gradient"
    )
    EXCLUSIONS[("text", _name)] = (
        "per-record gradients of the fine-tuned language model are the setting of LESS, which is the text member"
    )
    EXCLUSIONS[("native", _name)] = (
        "needs per-record gradients of the ResNet-18 during its training run; not run in the native protocol"
    )
EXCLUSIONS[("text", "infomax")] = "not in the unified text portfolio (LESS is the gradient-based text member)"
for _track in ("vision", "native", "text"):
    EXCLUSIONS[(_track, "auth3_only")] = (
        "the corruption (inlier) arm targets feature or sensor corruption, which this track does not inject"
    )
EXCLUSIONS[("timeseries", "auth3_only")] = (
    "the forecasting auth2_only already conjoins temporal structure and inlierness"
)
EXCLUSIONS[("text", "auth2_only")] = "text records carry no class label, so the label arm is undefined"
EXCLUSIONS[("native", "auth2_only")] = (
    "not instantiated in the native ResNet-18 protocol (penultimate kNN agreement is auth_only)"
)
EXCLUSIONS[("text", "dsdm")] = "datamodel regression needs one LM fine-tune per random subset; not run on text"
EXCLUSIONS[("native", "dsdm")] = "datamodel regression needs one ResNet-18 training per random subset; not run natively"
EXCLUSIONS[("native", "d4")] = "not instantiated in the native ResNet-18 protocol"
EXCLUSIONS[("text", "d4")] = "not instantiated in the text adapter"

CHALLENGER_ONLY = ("grid finalists", "cooperative finalists", "vote_ensemble(top3)", "fuse learned", "grpo_policy")
DIAGNOSTIC_ONLY = {"clean_oracle": "reads the corruption tags; diagnostic row, never a portfolio member"}


@dataclass
class MembershipRow:
    """One strategy on one track."""

    name: str
    role: str          # reference | challenger | diagnostic
    included: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return asdict(self)


def membership(track: str, mode: str) -> list[MembershipRow]:
    """Membership table of ``track`` under ``mode`` (canonical or unified)."""
    if track not in TRACKS:
        raise KeyError(f"unknown track {track!r}; expected one of {TRACKS}")
    rows: list[MembershipRow] = []
    if mode in HISTORICAL_REFERENCES:
        names = HISTORICAL_REFERENCES[mode].get(track)
        if names is None:
            raise KeyError(f"membership mode {mode!r} has no table for track {track!r}")
        return [MembershipRow(n, "reference", True, f"historical portfolio {mode}") for n in names]
    if mode == "canonical":
        for name in CANONICAL_REFERENCES[track]:
            rows.append(MembershipRow(name, "reference", True, "canonical 2026-07 runner portfolio"))
        for name in CANONICAL_CHALLENGERS.get(track, ()):
            rows.append(MembershipRow(name, "challenger", True, "canonical 2026-07 runner portfolio"))
        return rows
    if mode not in ("unified", "unified_v21", "unified_v22"):
        raise ValueError("mode must be canonical, unified, unified_v21, unified_v22 or a historical mode")
    order = UNIFIED_TEXT_ORDER if track == "text" else UNIFIED_ORDER
    if mode == "unified_v21":            # the single-signal rule of the alignment channel
        order = order[:1] + ("alignment_only",) + order[1:]
    if mode == "unified_v22":            # coop_herding (reference) and clean_top (challenger)
        order = order[:-1] + ("coop_herding", "clean_top") + order[-1:]
    challengers = UNIFIED_CHALLENGERS.get(track, ()) + (V22_CHALLENGERS if mode == "unified_v22" else ())
    for name in order:
        reason = EXCLUSIONS.get((track, name))
        role = "challenger" if name in challengers else "reference"
        rows.append(MembershipRow(name, role, reason is None, reason or "unified portfolio member"))
    for name, reason in WITHDRAWN.items():
        rows.append(MembershipRow(name, "reference", False, reason))
    for name, reason in DIAGNOSTIC_ONLY.items():
        rows.append(MembershipRow(name, "diagnostic", False, reason))
    return rows


def reference_members(track: str, mode: str) -> list[str]:
    """Included reference strategies of ``track`` in portfolio order."""
    return [r.name for r in membership(track, mode) if r.included and r.role == "reference"]


def challenger_members(track: str, mode: str) -> list[str]:
    """Included challenger-only strategies computed by the registry (grid cells excluded)."""
    return [r.name for r in membership(track, mode) if r.included and r.role == "challenger"]
