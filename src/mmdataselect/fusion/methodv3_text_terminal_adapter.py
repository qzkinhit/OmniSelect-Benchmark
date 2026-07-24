"""Terminal adapter for the method-v3 text clipped-logloss certificate."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from mmdataselect.fusion.paired_text_logloss_gate import (
    freeze_text_pair,
    run_paired_text_logloss_gate,
)


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def freeze_text_pair_manifest(
    *,
    reference_arm: str,
    challenger_arm: str,
    seed: int,
    pool_sha256: str,
    v1_split_sha256: str,
    reference_selector_sha256: str,
    challenger_selector_sha256: str,
    reference_selection_sha256: str,
    challenger_selection_sha256: str,
    reference_v1_evidence_sha256: str,
    challenger_v1_evidence_sha256: str,
    effective_token_cap: int,
) -> Dict[str, Any]:
    """Create the immutable pair record before any V2 outcome is opened."""
    body = {
        "schema_version": "omniselect.methodv3-text-pair.v1",
        "source_split": "V1",
        "family_size": 1,
        "reference_arm": reference_arm,
        "challenger_arm": challenger_arm,
        "seed": int(seed),
        "pool_sha256": pool_sha256,
        "v1_split_sha256": v1_split_sha256,
        "reference_selector_sha256": reference_selector_sha256,
        "challenger_selector_sha256": challenger_selector_sha256,
        "reference_selection_sha256": reference_selection_sha256,
        "challenger_selection_sha256": challenger_selection_sha256,
        "reference_v1_evidence_sha256": reference_v1_evidence_sha256,
        "challenger_v1_evidence_sha256": challenger_v1_evidence_sha256,
        "effective_token_cap": int(effective_token_cap),
    }
    for key in (
        "pool_sha256",
        "v1_split_sha256",
        "reference_selector_sha256",
        "challenger_selector_sha256",
        "reference_selection_sha256",
        "challenger_selection_sha256",
        "reference_v1_evidence_sha256",
        "challenger_v1_evidence_sha256",
    ):
        value = body[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"{key} must be a full lowercase SHA256")
    if not reference_arm or not challenger_arm or reference_arm == challenger_arm:
        raise ValueError("pair arms must be distinct non-empty strings")
    if body["effective_token_cap"] <= 0:
        raise ValueError("effective_token_cap must be positive")
    body["pair_manifest_sha256"] = canonical_sha256(body)
    return body


def run_methodv3_text_terminal_adapter(
    pair_manifest: Mapping[str, Any],
    *,
    ordered_record_ids: Sequence[str],
    domains: Sequence[str],
    token_counts: Sequence[int],
    reference_mean_nll: Sequence[float],
    challenger_mean_nll: Sequence[float],
    delta: float = 0.05,
    clip: float = 1.0,
) -> Dict[str, Any]:
    """Validate the frozen manifest and return one fail-closed terminal record."""
    try:
        manifest = dict(pair_manifest)
        claimed = manifest.pop("pair_manifest_sha256")
        if claimed != canonical_sha256(manifest):
            raise ValueError("pair manifest SHA mismatch")
        if manifest.get("schema_version") != "omniselect.methodv3-text-pair.v1":
            raise ValueError("unsupported pair manifest schema")
        if manifest.get("source_split") != "V1" or manifest.get("family_size") != 1:
            raise ValueError("pair must be frozen on V1 with family_size=1")
        ids = list(ordered_record_ids)
        if len(ids) == 0 or len(set(ids)) != len(ids):
            raise ValueError("ordered V2 record IDs must be non-empty and unique")
        expected_n = len(ids)
        if not all(
            len(values) == expected_n
            for values in (
                domains,
                token_counts,
                reference_mean_nll,
                challenger_mean_nll,
            )
        ):
            raise ValueError("all V2 arrays must align to ordered_record_ids")
        pair = freeze_text_pair(
            manifest["reference_arm"], manifest["challenger_arm"], claimed
        )
        result = run_paired_text_logloss_gate(
            pair,
            reference_mean_nll,
            challenger_mean_nll,
            token_counts,
            domains,
            delta=delta,
            clip=clip,
        )
        if result.decision == "ABSTAIN_UNCERTIFIED":
            record = result.to_dict()
            record.update(
                {
                    "mode": "method_v3",
                    "metric": "mean_token_nll",
                    "ordered_record_ids_sha256": canonical_sha256(ids),
                    "decision_record_sha256": None,
                }
            )
            record["decision_record_sha256"] = canonical_sha256(record)
            return record
        token_array = np.asarray(token_counts, dtype=np.float64)
        weights = np.zeros(expected_n, dtype=np.float64)
        domain_names = tuple(sorted(set(domains)))
        for domain in domain_names:
            mask = np.asarray([value == domain for value in domains], dtype=bool)
            weights[mask] = token_array[mask] / (
                len(domain_names) * float(token_array[mask].sum())
            )
        record = result.to_dict()
        record.update(
            {
                "mode": "method_v3",
                "metric": "mean_token_nll",
                "ordered_record_ids_sha256": canonical_sha256(ids),
                "domains_sha256": canonical_sha256(list(domains)),
                "token_counts_sha256": canonical_sha256([int(v) for v in token_counts]),
                "domain_balanced_weights_sha256": canonical_sha256(
                    [float(v) for v in weights]
                ),
                "reference_mean_nll_sha256": canonical_sha256(
                    [float(v) for v in reference_mean_nll]
                ),
                "challenger_mean_nll_sha256": canonical_sha256(
                    [float(v) for v in challenger_mean_nll]
                ),
            }
        )
        record["decision_record_sha256"] = canonical_sha256(record)
        return record
    except Exception as error:
        reference = (
            pair_manifest.get("reference_arm", "reference")
            if isinstance(pair_manifest, Mapping)
            else "reference"
        )
        return {
            "mode": "method_v3",
            "metric": "mean_token_nll",
            "decision": "ABSTAIN_UNCERTIFIED",
            "switched": False,
            "selected_arm": reference,
            "no_switch_reason": "INVALID_OR_MISALIGNED_TEXT_EVIDENCE",
            "error_type": type(error).__name__,
        }
