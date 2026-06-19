"""Tests for Pydantic models."""
import pytest
from code.agent.models import ClaimRow, HistoryRecord, MediaFile, ModelOutput, OutputRow, RowStats


def test_claim_row_image_path_list():
    row = ClaimRow(
        user_id="u1",
        image_paths="images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg",
        user_claim="My car is damaged.",
        claim_object="car",
    )
    assert row.image_path_list == [
        "images/test/case_001/img_1.jpg",
        "images/test/case_001/img_2.jpg",
    ]
    assert row.image_ids == ["img_1", "img_2"]


def test_claim_row_single_image():
    row = ClaimRow(
        user_id="u2",
        image_paths="images/test/case_003/img_1.jpg",
        user_claim="Laptop screen cracked.",
        claim_object="laptop",
    )
    assert len(row.image_path_list) == 1
    assert row.image_ids == ["img_1"]


def test_history_record_flag_set_none():
    rec = HistoryRecord(
        user_id="u1",
        past_claim_count=2,
        accept_claim=2,
        manual_review_claim=0,
        rejected_claim=0,
        last_90_days_claim_count=1,
        history_flags="none",
        history_summary="Low-risk user.",
    )
    assert rec.flag_set == set()


def test_history_record_flag_set_multi():
    rec = HistoryRecord(
        user_id="u2",
        past_claim_count=10,
        accept_claim=3,
        manual_review_claim=2,
        rejected_claim=5,
        last_90_days_claim_count=4,
        history_flags="repeated_claim;fraud_flag",
        history_summary="High risk.",
    )
    assert "repeated_claim" in rec.flag_set
    assert "fraud_flag" in rec.flag_set


def test_media_file_has_visual_content():
    mf = MediaFile(
        original_path="img_1.jpg",
        image_id="img_1",
        actual_format="JPEG",
        usable_frames=[b"\xff\xd8\xff" + b"\x00" * 10],
    )
    assert mf.has_visual_content is True


def test_media_file_no_content():
    mf = MediaFile(
        original_path="img_1.jpg",
        image_id="img_1",
        actual_format="UNKNOWN",
        usable_frames=[],
    )
    assert mf.has_visual_content is False


def test_output_row_model_dump():
    row = OutputRow(
        user_id="u1",
        image_paths="images/test/case_001/img_1.jpg",
        user_claim="Test claim.",
        claim_object="car",
        evidence_standard_met="true",
        evidence_standard_met_reason="Clear dent visible.",
        risk_flags="none",
        issue_type="dent",
        object_part="door",
        claim_status="supported",
        claim_status_justification="Image shows dent on door.",
        supporting_image_ids="img_1",
        valid_image="true",
        severity="medium",
    )
    d = row.model_dump()
    assert d["claim_status"] == "supported"
    assert d["valid_image"] == "true"


def test_row_stats_defaults():
    stats = RowStats(user_id="u1", strategy="strategy_b")
    assert stats.input_tokens == 0
    assert stats.cache_hit is False
    assert stats.error is None
    assert stats.estimated_input_cost_usd is None
