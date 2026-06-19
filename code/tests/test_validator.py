"""Tests for deterministic validator (enum enforcement, history merge, etc.)."""
import pytest

from code.agent.models import ClaimRow, HistoryRecord, ModelOutput
from code.agent.validator import validate_and_merge, zero_media_output


def _claim(claim_object: str = "car") -> ClaimRow:
    return ClaimRow(
        user_id="u1",
        image_paths="images/test/case_001/img_1.jpg",
        user_claim="My car door has a dent.",
        claim_object=claim_object,
    )


def _raw_output(**kwargs) -> ModelOutput:
    defaults = dict(
        evidence_standard_met=True,
        evidence_standard_met_reason="Dent visible.",
        risk_flags=["none"],
        issue_type="dent",
        object_part="door",
        claim_status="supported",
        claim_status_justification="Image shows dent on door.",
        supporting_image_ids=["img_1"],
        valid_image=True,
        severity="medium",
    )
    defaults.update(kwargs)
    return ModelOutput(**defaults)


def _history_with_flags(flags: str) -> HistoryRecord:
    return HistoryRecord(
        user_id="u1",
        past_claim_count=10,
        accept_claim=3,
        manual_review_claim=2,
        rejected_claim=5,
        last_90_days_claim_count=4,
        history_flags=flags,
        history_summary="High-risk user.",
    )


# --- Enum enforcement ---

def test_valid_claim_status_passthrough():
    row = validate_and_merge(_raw_output(), _claim(), None, ["img_1"])
    assert row.claim_status == "supported"


def test_invalid_claim_status_defaults_to_not_enough():
    raw = _raw_output(claim_status="maybe")
    row = validate_and_merge(raw, _claim(), None, ["img_1"])
    assert row.claim_status == "not_enough_information"


def test_invalid_severity_defaults_to_unknown():
    raw = _raw_output(severity="extreme")
    row = validate_and_merge(raw, _claim(), None, ["img_1"])
    assert row.severity == "unknown"


def test_invalid_issue_type_defaults_to_unknown():
    raw = _raw_output(issue_type="explosion")
    row = validate_and_merge(raw, _claim(), None, ["img_1"])
    assert row.issue_type == "unknown"


def test_car_object_part_valid():
    row = validate_and_merge(_raw_output(object_part="door"), _claim("car"), None, ["img_1"])
    assert row.object_part == "door"


def test_car_object_part_invalid_defaults_to_unknown():
    raw = _raw_output(object_part="screen")  # laptop part in a car claim
    row = validate_and_merge(raw, _claim("car"), None, ["img_1"])
    assert row.object_part == "unknown"


def test_laptop_object_part_valid():
    raw = _raw_output(object_part="screen")
    row = validate_and_merge(raw, _claim("laptop"), None, ["img_1"])
    assert row.object_part == "screen"


# --- supporting_image_ids subset check ---

def test_supporting_ids_filtered_to_submitted():
    raw = _raw_output(supporting_image_ids=["img_1", "img_99"])
    row = validate_and_merge(raw, _claim(), None, ["img_1"])
    assert "img_99" not in row.supporting_image_ids
    assert "img_1" in row.supporting_image_ids


def test_all_supporting_ids_invalid_becomes_none():
    raw = _raw_output(supporting_image_ids=["img_99", "img_88"])
    row = validate_and_merge(raw, _claim(), None, ["img_1"])
    assert row.supporting_image_ids == "none"


# --- History flag merge ---

def test_history_fraud_flag_adds_user_history_risk():
    hist = _history_with_flags("fraud_flag")
    row = validate_and_merge(_raw_output(), _claim(), hist, ["img_1"])
    assert "user_history_risk" in row.risk_flags


def test_history_no_flags_no_merge():
    hist = _history_with_flags("none")
    row = validate_and_merge(_raw_output(), _claim(), hist, ["img_1"])
    assert "user_history_risk" not in row.risk_flags


# --- valid_image=False propagation ---

def test_invalid_image_forces_not_enough_information():
    raw = _raw_output(valid_image=False, claim_status="supported")
    row = validate_and_merge(raw, _claim(), None, ["img_1"])
    assert row.claim_status == "not_enough_information"
    assert row.valid_image == "false"
    assert row.evidence_standard_met == "false"


# --- zero_media_output ---

def test_zero_media_no_history():
    row = zero_media_output(_claim(), None)
    assert row.valid_image == "false"
    assert row.evidence_standard_met == "false"
    assert row.claim_status == "not_enough_information"


def test_zero_media_with_risk_history():
    hist = _history_with_flags("repeated_claim")
    row = zero_media_output(_claim(), hist)
    assert "user_history_risk" in row.risk_flags
