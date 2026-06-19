"""Tests for prompt builders."""
import io
import pytest
from PIL import Image

from code.agent.models import ClaimRow, HistoryRecord, MediaFile
from code.agent.prompt import (
    STRATEGY_A,
    STRATEGY_B,
    build_system_prompt,
    build_user_message,
)


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (100, 100, 100)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def claim():
    return ClaimRow(
        user_id="u1",
        image_paths="images/test/case_001/img_1.jpg",
        user_claim="My car door has a dent.",
        claim_object="car",
    )


@pytest.fixture
def media_file():
    return MediaFile(
        original_path="images/test/case_001/img_1.jpg",
        image_id="img_1",
        actual_format="JPEG",
        usable_frames=[_jpeg_bytes()],
        frame_labels=["img_1, frame 0/1, format JPEG"],
    )


def test_system_prompt_contains_injection_guard():
    sp = build_system_prompt()
    assert "UNTRUSTED" in sp
    assert "text_instruction_present" in sp


def test_system_prompt_contains_schema():
    sp = build_system_prompt()
    assert "claim_status" in sp
    assert "evidence_standard_met" in sp


def test_user_message_strategy_a_structure(claim, media_file):
    content = build_user_message(claim, [media_file], None, None, STRATEGY_A)
    # First block is text
    assert content[0]["type"] == "text"
    text = content[0]["text"]
    assert "car" in text
    assert "UNTRUSTED" in text
    # Second block is text label, third is image
    assert content[1]["type"] == "text"
    assert "img_1" in content[1]["text"]
    assert content[2]["type"] == "image_url"


def test_user_message_strategy_b_includes_evidence(claim, media_file):
    evidence_text = "- [R1] (car): Show clear damage to claimed part."
    content = build_user_message(
        claim, [media_file], None, evidence_text, STRATEGY_B
    )
    text = content[0]["text"]
    assert "minimum evidence requirements" in text.lower() or "evidence requirements" in text.lower()
    assert "R1" in text


def test_user_message_strategy_b_includes_history(claim, media_file):
    history = HistoryRecord(
        user_id="u1",
        past_claim_count=3,
        accept_claim=2,
        manual_review_claim=1,
        rejected_claim=0,
        last_90_days_claim_count=2,
        history_flags="none",
        history_summary="Mostly accepted claims.",
    )
    content = build_user_message(claim, [media_file], history, None, STRATEGY_B)
    text = content[0]["text"]
    assert "history" in text.lower()


def test_user_message_no_media_short_text(claim):
    content = build_user_message(claim, [], None, None, STRATEGY_A)
    # Only one text block, no image blocks
    assert len(content) == 1
    assert "No usable images" in content[0]["text"]


def test_user_message_strategy_a_no_evidence(claim, media_file):
    content = build_user_message(claim, [media_file], None, "rules text", STRATEGY_A)
    # Strategy A must NOT include evidence text
    text = content[0]["text"]
    assert "rules text" not in text


def test_strategy_b_includes_calibration_guidance(claim, media_file):
    """Strategy B must include the calibration guidance block."""
    content = build_user_message(claim, [media_file], None, None, STRATEGY_B)
    text = content[0]["text"]
    assert "Calibration guidance" in text
    # Core calibration rules must be present
    assert "crack" in text
    assert "glass_shatter" in text
    assert "severity" in text.lower()
    assert "evidence_standard_met=true" in text


def test_strategy_a_excludes_calibration_guidance(claim, media_file):
    """Strategy A must NOT include the Strategy B calibration block."""
    content = build_user_message(claim, [media_file], None, None, STRATEGY_A)
    text = content[0]["text"]
    assert "Calibration guidance" not in text


def test_strategy_b_calibration_version_constant_exists():
    from code.agent.prompt import STRATEGY_B_CALIBRATION_VERSION
    assert isinstance(STRATEGY_B_CALIBRATION_VERSION, str) and STRATEGY_B_CALIBRATION_VERSION
