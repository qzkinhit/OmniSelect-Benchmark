"""Protocol presets, dotted overrides, legacy config keys and the membership tables."""
from __future__ import annotations

import pytest

from omniselect.config.config import OmniSelectConfig, parse_overrides
from omniselect.core.portfolio.membership import TRACKS, membership


def test_canonical_preset_values():
    cfg = OmniSelectConfig.preset("canonical")
    assert cfg.splits.mode == "two_way" and cfg.gate.kind == "margin" and cfg.gate.margin_frac == 0.015
    assert not cfg.cache.by_subset_hash and not cfg.fidelity.scoring_equals_reported
    assert cfg.influence.reference == "pool_clean_tag" and cfg.portfolio.membership == "canonical"
    assert not cfg.precheck.enabled and cfg.screening.sh_keep == 4
    assert cfg.fixes.fixed_fusion_gate == "sentinel"


def test_v2_preset_values():
    cfg = OmniSelectConfig.preset("v2")
    assert cfg.splits.mode == "three_way" and cfg.splits.fractions == (0.3, 0.5, 0.2)
    assert cfg.cache.by_subset_hash and cfg.fidelity.scoring_equals_reported
    assert cfg.influence.reference == "v_con" and cfg.portfolio.membership == "unified"
    assert cfg.precheck.enabled and not cfg.precheck.decide and cfg.gate.k_challengers == 3
    # margin on V_rank elects, V_conf is audited (docs/DESIGN.md)
    assert cfg.gate.kind == "margin" and cfg.gate.split == "rank" and cfg.gate.pair_split == "rank"
    assert cfg.gate.audit and cfg.gate.audit_split == "conf" and cfg.gate.reading_order == "stratified"
    assert cfg.gate.delta == 0.05 and cfg.gate.eps == 0.0 and cfg.gate.n_boot == 1000
    assert not OmniSelectConfig.preset("canonical").gate.audit


def test_overrides_cast_and_validate():
    cfg = OmniSelectConfig.preset("v2", **parse_overrides(["gate.kind=lcb", "precheck.enabled=false",
                                                           "splits.fractions=0.2,0.6,0.2"]))
    assert cfg.gate.kind == "lcb" and cfg.precheck.enabled is False and cfg.splits.fractions == (0.2, 0.6, 0.2)
    with pytest.raises(ValueError):
        OmniSelectConfig.preset("v2", **{"gate.kind": "nope"})
    with pytest.raises(KeyError):
        OmniSelectConfig.preset("v2", **{"gate.nope": 1})


def test_round_trip_through_dict():
    cfg = OmniSelectConfig.preset("v2", **{"gate.delta": 0.1})
    assert OmniSelectConfig.from_dict(cfg.to_dict()) == cfg


def test_every_excluded_member_has_a_reason():
    for track in TRACKS:
        for mode in ("canonical", "unified"):
            for row in membership(track, mode):
                assert row.reason, (track, mode, row.name)
    unified = {r.name: r for r in membership("timeseries", "unified")}
    assert not unified["el2n"].included and "classification-only" in unified["el2n"].reason
    assert not unified["quadmix"].included and "withdrawn" in unified["quadmix"].reason
    assert unified["coreset"].included and unified["auth_only"].included and unified["influence_only"].included


def test_unwired_robustness_settings_raise_before_the_cell_starts():
    from tracks.common.experiment import check_robustness

    preset = OmniSelectConfig.preset
    for track, overrides in [
        ("vision", {"robustness.val_noise_kind": "symmetric", "robustness.val_noise_rate": 0.2}),
        ("vision", {"robustness.mechanism_set": "unseen"}),
        ("vision", {"robustness.val_noise_kind": "natural"}),
        ("process", {"robustness.val_noise_kind": "prior_shift", "robustness.val_noise_rate": 0.5}),
        ("tabular", {"robustness.injection_ratio": 0.2}),
        ("timeseries", {"robustness.mechanism_set": "unseen"}),
    ]:
        check_robustness(track, preset("v2", **overrides))
    for track, overrides in [
        ("text", {"robustness.val_noise_kind": "symmetric", "robustness.val_noise_rate": 0.2}),
        ("timeseries", {"robustness.val_noise_kind": "symmetric", "robustness.val_noise_rate": 0.2}),
        ("process", {"robustness.val_noise_kind": "natural"}),
        ("native", {"robustness.injection_ratio": 0.2}),
        ("text", {"robustness.mechanism_set": "unseen"}),
    ]:
        with pytest.raises(NotImplementedError):
            check_robustness(track, preset("v2", **overrides))
    assert preset("v2", **{"robustness.val_noise_kind": "class_prior_shift"}).robustness.val_noise_kind == \
        "prior_shift"
    with pytest.raises(ValueError):
        preset("v2", **{"robustness.mechanism_set": "other"})


def test_legacy_config_keys_load():
    data = OmniSelectConfig.preset("v2").to_dict()
    data["robustness"] = {"injection": "unseen", "val_noise_kind": "none", "val_noise_rate": 0.0,
                          "injection_ratio": -1.0}
    data["gate"]["eprocess_weighting"] = "scaled"
    assert OmniSelectConfig.from_dict(data).robustness.mechanism_set == "unseen"
