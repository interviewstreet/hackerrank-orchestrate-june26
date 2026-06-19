"""Tests for boolean coercion in VLM output parsing (P0.3)."""
import pytest
from code.agent.vision_client import _coerce_bool, _coerce_model_output


# --- _coerce_bool ---

@pytest.mark.parametrize("value,expected", [
    (True, True),
    (False, False),
    ("true", True),
    ("True", True),
    ("TRUE", True),
    ("false", False),
    ("False", False),
    ("FALSE", False),
    ("1", True),
    ("0", False),
    (1, True),
    (0, False),
    (1.0, True),
    (0.0, False),
])
def test_coerce_bool_recognized(value, expected):
    assert _coerce_bool(value) == expected


@pytest.mark.parametrize("value", [
    "yes",   # not a recognized value — falls back to default
    "no",    # not a recognized value — falls back to default
    "maybe",
    None,
    [],
    {},
    "truee",
])
def test_coerce_bool_unknown_returns_default(value):
    # Default is False
    assert _coerce_bool(value, default=False) is False
    # Can override default
    assert _coerce_bool(value, default=True) is True


def test_coerce_bool_string_false_is_not_truthy():
    """The critical regression: bool('false') is True; _coerce_bool must return False."""
    assert _coerce_bool("false") is False
    assert _coerce_bool("False") is False
    # Prove the standard Python trap
    assert bool("false") is True  # the trap
    assert _coerce_bool("false") is False  # our fix


# --- _coerce_model_output with bool fields ---

def _raw_dict(**kwargs) -> dict:
    base = {
        "evidence_standard_met": True,
        "evidence_standard_met_reason": "Dent visible.",
        "risk_flags": ["none"],
        "issue_type": "dent",
        "object_part": "door",
        "claim_status": "supported",
        "claim_status_justification": "Image shows dent.",
        "supporting_image_ids": ["img_1"],
        "valid_image": True,
        "severity": "medium",
    }
    base.update(kwargs)
    return base


def test_model_output_true_bool():
    out = _coerce_model_output(_raw_dict(evidence_standard_met=True, valid_image=True))
    assert out.evidence_standard_met is True
    assert out.valid_image is True


def test_model_output_false_bool():
    out = _coerce_model_output(_raw_dict(evidence_standard_met=False, valid_image=False))
    assert out.evidence_standard_met is False
    assert out.valid_image is False


def test_model_output_string_true():
    out = _coerce_model_output(_raw_dict(evidence_standard_met="true", valid_image="true"))
    assert out.evidence_standard_met is True
    assert out.valid_image is True


def test_model_output_string_false():
    """Regression: 'false' string must produce False, not True."""
    out = _coerce_model_output(_raw_dict(evidence_standard_met="false", valid_image="false"))
    assert out.evidence_standard_met is False
    assert out.valid_image is False


def test_model_output_missing_bool_defaults_to_false():
    d = _raw_dict()
    d.pop("evidence_standard_met")
    d.pop("valid_image")
    out = _coerce_model_output(d)
    assert out.evidence_standard_met is False
    assert out.valid_image is False


def test_model_output_invalid_bool_defaults_to_false():
    out = _coerce_model_output(_raw_dict(evidence_standard_met="maybe", valid_image="unknown"))
    assert out.evidence_standard_met is False
    assert out.valid_image is False
