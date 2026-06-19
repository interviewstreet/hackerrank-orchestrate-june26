"""Tests for cache key generation and CacheStore round-trip."""
import io
import pytest
from PIL import Image

from code.agent.cache import CacheStore, make_cache_key
from code.agent.models import ClaimRow, MediaFile, ModelOutput


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (0, 128, 255)).save(buf, "JPEG")
    return buf.getvalue()


def _claim(**kwargs) -> ClaimRow:
    defaults = dict(
        user_id="u1",
        image_paths="images/test/case_001/img_1.jpg",
        user_claim="My car door is dented.",
        claim_object="car",
    )
    defaults.update(kwargs)
    return ClaimRow(**defaults)


def _fake_media(image_id: str = "img_1") -> MediaFile:
    return MediaFile(
        original_path=f"{image_id}.jpg",
        image_id=image_id,
        actual_format="JPEG",
        usable_frames=[_jpeg_bytes()],
        frame_labels=[f"{image_id}, frame 0/1, format JPEG"],
    )


def _sample_output() -> ModelOutput:
    return ModelOutput(
        evidence_standard_met=True,
        evidence_standard_met_reason="Dent visible on door.",
        risk_flags=["none"],
        issue_type="dent",
        object_part="door",
        claim_status="supported",
        claim_status_justification="Image shows dent.",
        supporting_image_ids=["img_1"],
        valid_image=True,
        severity="medium",
    )


# --- Key is a valid hex string ---

def test_cache_key_is_hex_string():
    key = make_cache_key("qwen", "qwen3.5-plus", "strategy_b", _claim(), None, None, [])
    assert len(key) == 64
    int(key, 16)  # must be valid hex


# --- Key changes with each input category ---

def test_cache_key_changes_with_model():
    k1 = make_cache_key("qwen", "qwen3.5-plus", "strategy_b", _claim(), None, None, [])
    k2 = make_cache_key("qwen", "qwen3.5-flash", "strategy_b", _claim(), None, None, [])
    assert k1 != k2


def test_cache_key_changes_with_strategy():
    k1 = make_cache_key("qwen", "qwen3.5-plus", "strategy_a", _claim(), None, None, [])
    k2 = make_cache_key("qwen", "qwen3.5-plus", "strategy_b", _claim(), None, None, [])
    assert k1 != k2


def test_cache_key_changes_with_claim_object():
    c1 = _claim(claim_object="car")
    c2 = _claim(claim_object="laptop")
    k1 = make_cache_key("qwen", "qwen3.5-plus", "strategy_b", c1, None, None, [])
    k2 = make_cache_key("qwen", "qwen3.5-plus", "strategy_b", c2, None, None, [])
    assert k1 != k2


def test_cache_key_changes_with_claim_text():
    c1 = _claim(user_claim="Door dent.")
    c2 = _claim(user_claim="Screen crack.")
    k1 = make_cache_key("qwen", "qwen3.5-plus", "strategy_b", c1, None, None, [])
    k2 = make_cache_key("qwen", "qwen3.5-plus", "strategy_b", c2, None, None, [])
    assert k1 != k2


def test_cache_key_changes_with_image_id():
    """Same pixel bytes, different image_id → different key (supporting IDs differ)."""
    same_bytes = _jpeg_bytes()
    mf1 = MediaFile(
        original_path="img_1.jpg", image_id="img_1",
        actual_format="JPEG", usable_frames=[same_bytes],
        frame_labels=["img_1, frame 0/1, format JPEG"]
    )
    mf2 = MediaFile(
        original_path="img_2.jpg", image_id="img_2",
        actual_format="JPEG", usable_frames=[same_bytes],
        frame_labels=["img_2, frame 0/1, format JPEG"]
    )
    k1 = make_cache_key("qwen", "qwen3.5-plus", "strategy_a", _claim(), None, None, [mf1])
    k2 = make_cache_key("qwen", "qwen3.5-plus", "strategy_a", _claim(), None, None, [mf2])
    assert k1 != k2, "Same pixels under different image IDs must not share cache key"


def test_cache_key_changes_with_frames():
    mf1 = _fake_media("img_1")
    mf_empty = MediaFile(
        original_path="img_1.jpg", image_id="img_1",
        actual_format="JPEG", usable_frames=[],
        frame_labels=[],
    )
    k1 = make_cache_key("qwen", "qwen3.5-plus", "strategy_a", _claim(), None, None, [mf1])
    k2 = make_cache_key("qwen", "qwen3.5-plus", "strategy_a", _claim(), None, None, [mf_empty])
    assert k1 != k2


def test_cache_key_changes_with_evidence_text():
    k1 = make_cache_key("qwen", "qwen3.5-plus", "strategy_b", _claim(), "evidence A", None, [])
    k2 = make_cache_key("qwen", "qwen3.5-plus", "strategy_b", _claim(), "evidence B", None, [])
    assert k1 != k2


# --- CacheStore ---

def test_cache_store_miss(tmp_path):
    store = CacheStore(tmp_path)
    assert store.get("nonexistentkey") is None


def test_cache_store_round_trip(tmp_path):
    store = CacheStore(tmp_path)
    output = _sample_output()
    key = make_cache_key("qwen", "qwen3.5-plus", "strategy_b", _claim(), "ev", "hist", [])
    store.set(key, output)
    retrieved = store.get(key)
    assert retrieved is not None
    assert retrieved.claim_status == "supported"
    assert retrieved.issue_type == "dent"


def test_cache_store_corrupted_file_returns_none(tmp_path):
    store = CacheStore(tmp_path)
    p = store._path("badkey")
    p.write_text("not valid json", encoding="utf-8")
    assert store.get("badkey") is None
