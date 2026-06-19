"""Tests for output.write."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import output

EXPECTED_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

FULL_ROW = {
    "user_id": "user_001",
    "image_paths": "images/test/case_001/img_1.jpg",
    "user_claim": "The rear bumper has a dent.",
    "claim_object": "car",
    "evidence_standard_met": "true",
    "evidence_standard_met_reason": "Dent is clearly visible in the image.",
    "risk_flags": "none",
    "issue_type": "dent",
    "object_part": "rear_bumper",
    "claim_status": "supported",
    "claim_status_justification": "Image confirms a dent on the rear bumper.",
    "supporting_image_ids": "img_1",
    "valid_image": "true",
    "severity": "medium",
}


def _read_back(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_all_14_columns_present(tmp_path):
    dest = tmp_path / "out.csv"
    output.write([FULL_ROW], dest)
    rows = _read_back(dest)
    assert len(rows) == 1
    assert list(rows[0].keys()) == EXPECTED_COLUMNS


def test_column_order_matches_sample(tmp_path):
    dest = tmp_path / "out.csv"
    output.write([FULL_ROW], dest)
    with dest.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\r\n").split(",")
    assert header == EXPECTED_COLUMNS


def test_values_round_trip(tmp_path):
    dest = tmp_path / "out.csv"
    output.write([FULL_ROW], dest)
    rows = _read_back(dest)
    for col in EXPECTED_COLUMNS:
        assert rows[0][col] == FULL_ROW[col], f"mismatch on column {col!r}"


def test_missing_key_writes_empty_string(tmp_path):
    dest = tmp_path / "out.csv"
    output.write([{"user_id": "u1"}], dest)
    rows = _read_back(dest)
    assert rows[0]["user_id"] == "u1"
    for col in EXPECTED_COLUMNS:
        if col != "user_id":
            assert rows[0][col] == "", f"expected empty string for {col!r}"


def test_extra_key_does_not_crash(tmp_path):
    dest = tmp_path / "out.csv"
    row = dict(FULL_ROW, _internal_extra="should_be_ignored")
    output.write([row], dest)
    rows = _read_back(dest)
    assert "_internal_extra" not in rows[0]


def test_utf8_no_bom(tmp_path):
    dest = tmp_path / "out.csv"
    row = dict(FULL_ROW, user_claim="Daño en el parabrisas")
    output.write([row], dest)
    raw = dest.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "BOM found"
    raw.decode("utf-8")  # raises if not valid UTF-8


def test_multiple_rows(tmp_path):
    dest = tmp_path / "out.csv"
    rows_in = [dict(FULL_ROW, user_id=f"user_{i:03d}") for i in range(5)]
    output.write(rows_in, dest)
    rows_out = _read_back(dest)
    assert len(rows_out) == 5
    for i, r in enumerate(rows_out):
        assert r["user_id"] == f"user_{i:03d}"


def test_empty_predictions_writes_header_only(tmp_path):
    dest = tmp_path / "out.csv"
    output.write([], dest)
    with dest.open(encoding="utf-8") as fh:
        lines = fh.readlines()
    assert len(lines) == 1  # header only
    assert lines[0].rstrip("\r\n").split(",") == EXPECTED_COLUMNS


def test_accepts_string_path(tmp_path):
    dest = str(tmp_path / "out.csv")
    output.write([FULL_ROW], dest)
    rows = _read_back(Path(dest))
    assert len(rows) == 1
