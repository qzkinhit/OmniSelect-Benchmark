"""UnifiedRecord serialization round-trips + Modality semantics."""
from __future__ import annotations

from mmdataselect.datatypes import (
    DOMAIN_CODE,
    DOMAIN_GENERAL,
    DOMAIN_MATH,
    Modality,
    UnifiedRecord,
)


def test_modality_is_str_enum():
    # str mix-in keeps JSON round-trips trivial.
    assert Modality.TEXT == "text"
    assert Modality.IMAGE_TEXT.value == "image_text"
    assert Modality("audio_text") is Modality.AUDIO_TEXT


def test_domain_constants():
    assert DOMAIN_GENERAL == "general"
    assert DOMAIN_MATH == "math"
    assert DOMAIN_CODE == "code"


def test_to_dict_shape_and_modality_value():
    r = UnifiedRecord(
        id="x1",
        modality=Modality.IMAGE_TEXT,
        domain=DOMAIN_MATH,
        text="hello",
        meta={"k": 1},
    )
    d = r.to_dict()
    assert set(d.keys()) == {"id", "modality", "domain", "text", "meta"}
    # modality is serialized to its string value, not the enum object.
    assert d["modality"] == "image_text"
    assert isinstance(d["modality"], str)
    assert d == {
        "id": "x1",
        "modality": "image_text",
        "domain": "math",
        "text": "hello",
        "meta": {"k": 1},
    }


def test_round_trip_to_from_dict():
    r = UnifiedRecord(
        id="x2",
        modality=Modality.AUDIO_TEXT,
        domain=DOMAIN_CODE,
        text="some code text",
        meta={"lang": "py", "tokens": 3},
    )
    r2 = UnifiedRecord.from_dict(r.to_dict())
    assert r2 == r
    # modality is rehydrated back into an enum.
    assert isinstance(r2.modality, Modality)
    assert r2.modality is Modality.AUDIO_TEXT


def test_from_dict_defaults_and_coercion():
    # Missing optional fields fall back to sane defaults; id is coerced to str.
    r = UnifiedRecord.from_dict({"id": 123})
    assert r.id == "123"
    assert isinstance(r.id, str)
    assert r.modality is Modality.TEXT
    assert r.domain == DOMAIN_GENERAL
    assert r.text == ""
    assert r.meta == {}


def test_from_dict_none_text_and_meta_normalized():
    # Explicit None for text/meta is normalized to empty values, not propagated.
    r = UnifiedRecord.from_dict(
        {"id": "z", "modality": "text", "domain": "general", "text": None, "meta": None}
    )
    assert r.text == ""
    assert r.meta == {}


def test_from_dict_accepts_enum_modality():
    # from_dict tolerates an already-parsed Modality enum.
    r = UnifiedRecord.from_dict({"id": "e", "modality": Modality.IMAGE_TEXT, "domain": "general"})
    assert r.modality is Modality.IMAGE_TEXT
