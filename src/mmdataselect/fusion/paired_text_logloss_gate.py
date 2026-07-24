"""Fail-closed paired clipped-logloss gate for the method-v3 text arm.

The observational unit is one held-out record, never one token.  Records are
weighted so each domain contributes exactly ``1 / D`` and tokens only determine
the within-domain average.  For the V1-frozen ordered pair we test the bounded
paired improvement

    z_i = clip(NLL_reference(i) - NLL_challenger(i), -C, C)

with the conditional Azuma/Hoeffding radius

    C * sqrt(2 log(K / delta) sum_i w_i**2).

The challenger is selected only when the strict LCB is positive.  Invalid
evidence never invokes an empirical-margin fallback and always abstains.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional, Sequence

import numpy as np


GATE_TYPE = "domain_balanced_paired_clipped_logloss_conditional_azuma_v1"
TARGET = "domain_balanced_clipped_logloss_improvement"
NULL_ID = "conditional_mean_nonpositive_per_ordered_record_v1"
DEPENDENCE_ASSUMPTION = (
    "for the V1-frozen pair, ordered V2 clipped record-level differences form "
    "an adapted bounded sequence with conditional mean at most zero under H0"
)


@dataclass(frozen=True)
class FrozenTextPair:
    reference_arm: str
    challenger_arm: str
    pair_manifest_sha256: str
    source_split: str = "V1"
    family_size: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.reference_arm, str) or not self.reference_arm:
            raise ValueError("reference_arm must be a non-empty string")
        if not isinstance(self.challenger_arm, str) or not self.challenger_arm:
            raise ValueError("challenger_arm must be a non-empty string")
        if self.reference_arm == self.challenger_arm:
            raise ValueError("reference and challenger must be distinct")
        if self.source_split != "V1" or self.family_size != 1:
            raise ValueError("text gate accepts one pair frozen on V1")
        if (
            not isinstance(self.pair_manifest_sha256, str)
            or len(self.pair_manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.pair_manifest_sha256)
        ):
            raise ValueError("pair_manifest_sha256 must be a full lowercase SHA256")


@dataclass(frozen=True)
class PairedTextLoglossGateResult:
    target: str
    gate_type: str
    null_id: str
    delta: Optional[float]
    family_size: int
    clip: Optional[float]
    n_records: int
    n_domains: int
    n_tokens: int
    sum_weight_squared: Optional[float]
    estimate: Optional[float]
    radius: Optional[float]
    lcb: Optional[float]
    switched: bool
    decision: str
    no_switch_reason: Optional[str]
    reference_arm: str
    challenger_arm: str
    selected_arm: str
    source_split: Optional[str]
    pair_manifest_sha256: Optional[str]
    dependence_assumption: str
    error_type: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def freeze_text_pair(
    reference_arm: str,
    challenger_arm: str,
    pair_manifest_sha256: str,
) -> FrozenTextPair:
    return FrozenTextPair(reference_arm, challenger_arm, pair_manifest_sha256)


def _invalid(pair: object, reason: str, error: BaseException) -> PairedTextLoglossGateResult:
    reference = getattr(pair, "reference_arm", "reference")
    challenger = getattr(pair, "challenger_arm", "challenger")
    if not isinstance(reference, str) or not reference:
        reference = "reference"
    if not isinstance(challenger, str) or not challenger:
        challenger = "challenger"
    return PairedTextLoglossGateResult(
        target=TARGET,
        gate_type=GATE_TYPE,
        null_id=NULL_ID,
        delta=None,
        family_size=1,
        clip=None,
        n_records=0,
        n_domains=0,
        n_tokens=0,
        sum_weight_squared=None,
        estimate=None,
        radius=None,
        lcb=None,
        switched=False,
        decision="ABSTAIN_UNCERTIFIED",
        no_switch_reason=reason,
        reference_arm=reference,
        challenger_arm=challenger,
        selected_arm=reference,
        source_split=getattr(pair, "source_split", None),
        pair_manifest_sha256=getattr(pair, "pair_manifest_sha256", None),
        dependence_assumption=DEPENDENCE_ASSUMPTION,
        error_type=type(error).__name__,
    )


def run_paired_text_logloss_gate(
    pair: FrozenTextPair,
    reference_mean_nll: Sequence[float],
    challenger_mean_nll: Sequence[float],
    token_counts: Sequence[int],
    domains: Sequence[str],
    *,
    delta: float = 0.05,
    clip: float = 1.0,
) -> PairedTextLoglossGateResult:
    """Certify one V1-frozen pair from ordered, paired V2 record evidence."""

    try:
        if not isinstance(pair, FrozenTextPair):
            raise TypeError("pair must be FrozenTextPair")
        pair.__post_init__()
        delta = float(delta)
        clip = float(clip)
        if not math.isfinite(delta) or not 0.0 < delta < 1.0:
            raise ValueError("delta must be finite in (0, 1)")
        if not math.isfinite(clip) or clip <= 0.0:
            raise ValueError("clip must be finite and positive")
        ref = np.asarray(reference_mean_nll, dtype=np.float64)
        chal = np.asarray(challenger_mean_nll, dtype=np.float64)
        tokens = np.asarray(token_counts)
        if ref.ndim != 1 or len(ref) == 0 or chal.shape != ref.shape:
            raise ValueError("paired NLL vectors must be non-empty and aligned")
        if tokens.ndim != 1 or tokens.shape != ref.shape:
            raise ValueError("token_counts must align with paired NLL vectors")
        if not np.isfinite(ref).all() or not np.isfinite(chal).all():
            raise ValueError("NLL values must be finite")
        if not np.issubdtype(tokens.dtype, np.integer) or (tokens <= 0).any():
            raise ValueError("token_counts must be positive integers")
        if len(domains) != len(ref) or any(not isinstance(d, str) or not d for d in domains):
            raise ValueError("domains must be aligned non-empty strings")
        domain_names = tuple(sorted(set(domains)))
        if not domain_names:
            raise ValueError("at least one domain is required")

        weights = np.zeros(len(ref), dtype=np.float64)
        for domain in domain_names:
            mask = np.asarray([value == domain for value in domains], dtype=bool)
            domain_tokens = int(tokens[mask].sum())
            if domain_tokens <= 0:
                raise ValueError("every domain must contain positive prediction tokens")
            weights[mask] = tokens[mask].astype(np.float64) / (
                len(domain_names) * domain_tokens
            )
        if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("domain-balanced weights do not sum to one")

        clipped = np.clip(ref - chal, -clip, clip)
        estimate = float(np.dot(weights, clipped))
        sum_weight_squared = float(np.dot(weights, weights))
        radius = float(
            clip
            * math.sqrt(
                2.0 * math.log(pair.family_size / delta) * sum_weight_squared
            )
        )
        lcb = estimate - radius
        switched = lcb > 0.0
        return PairedTextLoglossGateResult(
            target=TARGET,
            gate_type=GATE_TYPE,
            null_id=NULL_ID,
            delta=delta,
            family_size=pair.family_size,
            clip=clip,
            n_records=len(ref),
            n_domains=len(domain_names),
            n_tokens=int(tokens.sum()),
            sum_weight_squared=sum_weight_squared,
            estimate=estimate,
            radius=radius,
            lcb=lcb,
            switched=switched,
            decision="SWITCH_CERTIFIED" if switched else "KEEP_REFERENCE",
            no_switch_reason=None if switched else "LCB_NOT_STRICTLY_POSITIVE",
            reference_arm=pair.reference_arm,
            challenger_arm=pair.challenger_arm,
            selected_arm=pair.challenger_arm if switched else pair.reference_arm,
            source_split=pair.source_split,
            pair_manifest_sha256=pair.pair_manifest_sha256,
            dependence_assumption=DEPENDENCE_ASSUMPTION,
            error_type=None,
        )
    except (TypeError, ValueError, OverflowError, FloatingPointError) as error:
        return _invalid(pair, "UNCERTIFIED_GATE_INPUT", error)
