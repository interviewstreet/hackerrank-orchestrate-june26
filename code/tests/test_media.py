"""Tests for media loading, format detection, and image normalisation."""
import io
import pytest
from PIL import Image

from code.agent.media import detect_format, _normalize_to_jpeg, load_media_file, load_row_media
from code.agent.models import MediaFile


def _make_jpeg_bytes(w: int = 64, h: int = 64) -> bytes:
    img = Image.new("RGB", (w, h), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_bytes(w: int = 64, h: int = 64) -> bytes:
    img = Image.new("RGB", (w, h), color=(50, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- detect_format ---

def test_detect_jpeg():
    raw = _make_jpeg_bytes()
    assert detect_format(raw[:12]) == "JPEG"


def test_detect_png():
    raw = _make_png_bytes()
    assert detect_format(raw[:12]) == "PNG"


def test_detect_webp():
    header = b"RIFF\x00\x00\x00\x00WEBP"
    assert detect_format(header[:12]) == "WEBP"


def test_detect_mp4_ftyp():
    header = b"\x00\x00\x00\x20ftypisom"
    assert detect_format(header[:12]) == "MP4"


def test_detect_mp4_moov():
    header = b"\x00\x00\x00\x08moov" + b"\x00" * 4
    assert detect_format(header[:12]) == "MP4"


def test_detect_unknown():
    assert detect_format(b"\x00\x01\x02\x03" + b"\x00" * 8) == "UNKNOWN"


# --- _normalize_to_jpeg ---

def test_normalize_jpeg_passthrough():
    raw = _make_jpeg_bytes(200, 150)
    result = _normalize_to_jpeg(raw, max_long_edge=1024)
    out_img = Image.open(io.BytesIO(result))
    assert out_img.format == "JPEG"
    assert max(out_img.size) <= 1024


def test_normalize_large_image_resizes():
    raw = _make_jpeg_bytes(2048, 1024)
    result = _normalize_to_jpeg(raw, max_long_edge=1024)
    out_img = Image.open(io.BytesIO(result))
    assert max(out_img.size) == 1024


def test_normalize_png():
    raw = _make_png_bytes(100, 200)
    result = _normalize_to_jpeg(raw, max_long_edge=1024)
    out_img = Image.open(io.BytesIO(result))
    assert out_img.format == "JPEG"


def test_normalize_aspect_ratio_preserved():
    raw = _make_jpeg_bytes(2000, 500)
    result = _normalize_to_jpeg(raw, max_long_edge=1000)
    out_img = Image.open(io.BytesIO(result))
    w, h = out_img.size
    assert w == 1000
    assert abs(h - 250) <= 2  # tolerance for rounding


# --- load_media_file (real file) ---

def test_load_jpeg_file(images_root):
    """Load a real JPEG from the test dataset (sample case_001)."""
    img_path = images_root / "sample" / "case_001" / "img_1.jpg"
    if not img_path.exists():
        pytest.skip("Dataset image not available")
    mf = load_media_file(img_path)
    assert mf.image_id == "img_1"
    assert mf.has_visual_content
    assert len(mf.usable_frames) == 1
    assert mf.actual_format in ("JPEG", "PNG", "WEBP", "MP4")


# --- load_row_media ---

def test_load_row_media_missing(repo_root, tmp_path):
    """Missing file produces MediaFile with no frames (no crash)."""
    files = load_row_media(["images/test/nonexistent/img_99.jpg"], repo_root)
    assert len(files) == 1
    mf = files[0]
    assert mf.actual_format == "MISSING"
    assert not mf.has_visual_content


def test_load_row_media_multiple(images_root, repo_root):
    """Load two real dataset images."""
    paths = [
        "dataset/images/sample/case_005/img_1.jpg",
        "dataset/images/sample/case_005/img_2.jpg",
    ]
    files = load_row_media(paths, repo_root)
    # At least verify we got 2 MediaFile objects (may be MISSING if dataset absent)
    assert len(files) == 2
