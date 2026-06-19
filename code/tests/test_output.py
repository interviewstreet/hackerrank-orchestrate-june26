"""Tests for output CSV column structure and OutputRow serialization."""
import csv
import io
import pytest

from code.agent.models import OutputRow

OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
    "issue_type", "object_part", "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]


def _make_row(**kwargs) -> OutputRow:
    defaults = dict(
        user_id="u1",
        image_paths="images/test/case_001/img_1.jpg",
        user_claim="My car door is dented.",
        claim_object="car",
        evidence_standard_met="true",
        evidence_standard_met_reason="Dent visible.",
        risk_flags="none",
        issue_type="dent",
        object_part="door",
        claim_status="supported",
        claim_status_justification="img_1 shows dent.",
        supporting_image_ids="img_1",
        valid_image="true",
        severity="medium",
    )
    defaults.update(kwargs)
    return OutputRow(**defaults)


def test_output_row_has_14_fields():
    row = _make_row()
    d = row.model_dump()
    assert len(d) == 14


def test_output_row_column_order():
    row = _make_row()
    keys = list(row.model_dump().keys())
    assert keys == OUTPUT_COLUMNS


def test_output_row_csv_roundtrip():
    row = _make_row()
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerow(row.model_dump())

    buf.seek(0)
    reader = csv.DictReader(buf)
    read_back = next(reader)
    assert read_back["claim_status"] == "supported"
    assert read_back["valid_image"] == "true"
    assert read_back["severity"] == "medium"


def test_output_row_none_defaults():
    row = _make_row(
        evidence_standard_met="false",
        valid_image="false",
        claim_status="not_enough_information",
        supporting_image_ids="none",
        risk_flags="damage_not_visible",
    )
    assert row.supporting_image_ids == "none"
    assert row.claim_status == "not_enough_information"
