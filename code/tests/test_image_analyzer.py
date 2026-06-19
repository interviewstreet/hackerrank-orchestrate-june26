"""Unit tests for image_analyzer.py (CALL 2).

All tests are fully mocked — no Anthropic API calls are made. The lazy client
is patched so the module never touches the network. Sections mirror the module's
helper structure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

import image_analyzer as ia


# --- helpers ------------------------------------------------------------------

def _make_response(payload: dict | str) -> SimpleNamespace:
    """Build a minimal mock Anthropic response object."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(content=[SimpleNamespace(text=text)])


def _valid_payload(**overrides) -> dict:
    base = {
        "per_image_analysis": [
            {
                "image_id": "img_1",
                "what_is_visible": "car rear bumper with a dent",
                "object_part_visible": "rear_bumper",
                "issue_visible": "dent",
                "quality_flags": [],
            }
        ],
        "overall_part_visible": "rear_bumper",
        "overall_issue_visible": "dent",
        "image_quality_flags": [],
        "valid_image": True,
        "evidence_standard_met": True,
        "evidence_standard_met_reason": "Bumper clearly visible; dent assessable.",
        "candidate_supporting_image_ids": ["img_1"],
    }
    base.update(overrides)
    return base


# --- SECTION 1: empty image list (fork 3) ------------------------------------

def test_empty_image_paths_returns_fallback_without_api_call() -> None:
    with patch.object(ia, "_get_client") as mock_client:
        result = ia.analyze_images([], "car", [], "summary", [])
    mock_client.assert_not_called()
    assert result["valid_image"] is False
    assert result["evidence_standard_met"] is False
    assert result["evidence_standard_met_reason"] == "No images submitted."
    assert result["per_image_analysis"] == []
    assert result["candidate_supporting_image_ids"] == []


# _DUMMY_PREPARED stands in for real image data in tests that exercise the
# API-response path. Patching _prepare_image avoids FileNotFoundError on
# non-existent paths and skips the AVIF detection/conversion logic.
_DUMMY_PREPARED = ("image/jpeg", "AAAA")


# --- SECTION 2: successful JSON response --------------------------------------

def test_valid_json_response_is_returned_unchanged() -> None:
    payload = _valid_payload()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_response(payload)
    with patch.object(ia, "_get_client", return_value=mock_client), \
         patch.object(ia, "_prepare_image", return_value=_DUMMY_PREPARED):
        result = ia.analyze_images(
            ["dataset/images/test/case_001/img_1.jpg"],
            "car",
            [{"object_part": "rear_bumper", "issue_type": "dent"}],
            "Rear bumper dented",
            [{"requirement_id": "REQ_CAR_BODY_PANEL", "minimum_image_evidence": "panel visible"}],
        )
    assert result["overall_issue_visible"] == "dent"
    assert result["overall_part_visible"] == "rear_bumper"
    assert result["valid_image"] is True
    assert result["evidence_standard_met"] is True
    assert result["candidate_supporting_image_ids"] == ["img_1"]


# --- SECTION 3: malformed JSON response --------------------------------------

def test_malformed_json_returns_fallback() -> None:
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_response("this is not json {{{")
    with patch.object(ia, "_get_client", return_value=mock_client), \
         patch.object(ia, "_prepare_image", return_value=_DUMMY_PREPARED):
        result = ia.analyze_images(
            ["dataset/images/test/case_001/img_1.jpg"],
            "car", [], "summary", [],
        )
    assert result["valid_image"] is False
    assert result["evidence_standard_met"] is False
    assert "parsed" in result["evidence_standard_met_reason"].lower()


def test_empty_string_response_returns_fallback() -> None:
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_response("")
    with patch.object(ia, "_get_client", return_value=mock_client), \
         patch.object(ia, "_prepare_image", return_value=_DUMMY_PREPARED):
        result = ia.analyze_images(
            ["dataset/images/test/case_001/img_1.jpg"],
            "car", [], "summary", [],
        )
    assert result["valid_image"] is False


# --- SECTION 4: missing required key -----------------------------------------

@pytest.mark.parametrize("missing_key", list(ia._REQUIRED_KEYS))
def test_missing_required_key_returns_fallback(missing_key: str) -> None:
    payload = _valid_payload()
    del payload[missing_key]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_response(payload)
    with patch.object(ia, "_get_client", return_value=mock_client), \
         patch.object(ia, "_prepare_image", return_value=_DUMMY_PREPARED):
        result = ia.analyze_images(
            ["dataset/images/test/case_001/img_1.jpg"],
            "car", [], "summary", [],
        )
    assert result["valid_image"] is False
    assert "shape" in result["evidence_standard_met_reason"].lower()


# --- SECTION 5: image_id extraction ------------------------------------------

def test_image_id_extracted_from_deep_path() -> None:
    path = "dataset/images/test/case_001/img_1.jpg"
    assert Path(path).stem == "img_1"


def test_build_content_uses_filename_stem_as_image_id(tmp_path: Path) -> None:
    img = tmp_path / "img_42.jpg"
    img.write_bytes(b"\xff\xd8\xff")  # minimal JPEG header bytes
    content = ia._build_content([str(img)], "car", [], "summary")
    id_blocks = [b for b in content if b.get("type") == "text" and "image_id:" in b.get("text", "")]
    assert len(id_blocks) == 1
    assert id_blocks[0]["text"] == "image_id: img_42"


# --- SECTION 6: media type detection -----------------------------------------

@pytest.mark.parametrize("ext,expected", [
    (".jpg",  "image/jpeg"),
    (".jpeg", "image/jpeg"),
    (".png",  "image/png"),
    (".PNG",  "image/png"),   # uppercase extension
    (".webp", "image/jpeg"),  # unknown → default JPEG
])
def test_media_type(ext: str, expected: str) -> None:
    assert ia._media_type(f"/some/path/img{ext}") == expected
