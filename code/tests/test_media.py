"""Tests for media loading, format detection, image normalisation, and FFmpeg."""
import io
import pytest
from PIL import Image
from unittest.mock import patch

from code.agent.media import (
    detect_format,
    _normalize_to_jpeg,
    _ffmpeg_available,
    extract_video_frames,
    load_media_file,
    load_row_media,
)
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
    assert detect_format(_make_jpeg_bytes()[:12]) == "JPEG"


def test_detect_png():
    assert detect_format(_make_png_bytes()[:12]) == "PNG"


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
    assert abs(h - 250) <= 2


# --- FFmpeg integration (real video files in dataset) ---

def _find_mp4_paths(images_root):
    """Return dataset-relative paths (images/test/...) for MP4 files."""
    paths = []
    for f in images_root.rglob("*.jpg"):
        with open(f, "rb") as fh:
            h = fh.read(12)
        if h[4:8] in (b"ftyp", b"moov", b"mdat", b"free", b"wide"):
            # Return relative to dataset_root (parent of images_root)
            paths.append(f)
    return paths


def test_ffmpeg_extracts_real_video_frames(images_root):
    if not _ffmpeg_available():
        pytest.skip("FFmpeg not available")
    mp4_paths = _find_mp4_paths(images_root)
    if not mp4_paths:
        pytest.skip("No MP4 files found in dataset")

    path = mp4_paths[0]
    frames = extract_video_frames(path, n_frames=3)
    assert len(frames) >= 1, "Expected at least one frame from real MP4"
    for frame_bytes in frames:
        img = Image.open(io.BytesIO(frame_bytes))
        assert img.format == "JPEG"
        assert max(img.size) <= 1024


def test_ffmpeg_all_videos_produce_frames(images_root):
    if not _ffmpeg_available():
        pytest.skip("FFmpeg not available")
    mp4_paths = _find_mp4_paths(images_root)
    if not mp4_paths:
        pytest.skip("No MP4 files found in dataset")

    for path in mp4_paths:
        frames = extract_video_frames(path, n_frames=3)
        assert len(frames) >= 1, f"No frames from {path}"
        for fb in frames:
            img = Image.open(io.BytesIO(fb))
            assert max(img.size) <= 1024


def test_ffmpeg_unavailable_returns_empty(images_root):
    """When FFmpeg is mocked as absent, extract_video_frames returns []."""
    mp4_paths = _find_mp4_paths(images_root)
    if not mp4_paths:
        pytest.skip("No MP4 files found in dataset")
    with patch("code.agent.media._ffmpeg_available", return_value=False):
        frames = extract_video_frames(mp4_paths[0])
    assert frames == []


# --- load_media_file (real file) ---

def test_load_jpeg_file(images_root):
    img_path = images_root / "sample" / "case_001" / "img_1.jpg"
    if not img_path.exists():
        pytest.skip("Dataset image not available")
    mf = load_media_file(img_path)
    assert mf.image_id == "img_1"
    assert mf.has_visual_content
    assert len(mf.usable_frames) == 1
    assert mf.actual_format in ("JPEG", "PNG", "WEBP", "MP4")


def test_load_mp4_file_extracts_frames(images_root):
    if not _ffmpeg_available():
        pytest.skip("FFmpeg not available")
    mp4_paths = _find_mp4_paths(images_root)
    if not mp4_paths:
        pytest.skip("No MP4 files in dataset")
    mf = load_media_file(mp4_paths[0])
    assert mf.actual_format == "MP4"
    assert mf.has_visual_content
    assert len(mf.usable_frames) >= 1


# --- load_row_media with correct dataset_root ---

def test_load_row_media_missing(dataset_root):
    """Missing file produces MediaFile with no frames (no crash)."""
    files = load_row_media(["images/test/nonexistent/img_99.jpg"], dataset_root)
    assert len(files) == 1
    assert files[0].actual_format == "MISSING"
    assert not files[0].has_visual_content


def test_load_row_media_path_traversal_rejected(dataset_root):
    """Path traversal outside dataset_root must be rejected."""
    files = load_row_media(["../../some/secret.jpg"], dataset_root)
    assert len(files) == 1
    assert files[0].actual_format == "TRAVERSAL"
    assert not files[0].has_visual_content


def test_load_row_media_real_sample(dataset_root):
    """Load two real dataset images using exact CSV path form."""
    paths = [
        "images/sample/case_005/img_1.jpg",
        "images/sample/case_005/img_2.jpg",
    ]
    files = load_row_media(paths, dataset_root)
    assert len(files) == 2
    # Both files exist in the dataset — verify at least one has content
    has_content = [f for f in files if f.has_visual_content or f.actual_format == "MISSING"]
    assert len(has_content) == 2


def test_load_row_media_resolves_against_dataset_not_repo_root(dataset_root, repo_root):
    """CSV path 'images/sample/case_001/img_1.jpg' resolves via dataset_root, not repo_root."""
    # Using repo_root would give challenge/images/... which doesn't exist
    path = "images/sample/case_001/img_1.jpg"
    files_correct = load_row_media([path], dataset_root)
    files_wrong = load_row_media([path], repo_root)
    # Correct: real file found
    assert files_correct[0].actual_format != "MISSING"
    # Wrong root: MISSING
    assert files_wrong[0].actual_format == "MISSING"
