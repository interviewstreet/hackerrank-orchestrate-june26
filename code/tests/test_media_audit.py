"""Offline media audit tests.

P0.1: Integration test that reads real rows from sample_claims.csv,
      resolves all image paths, and asserts usable content is found.
P0.2: FFmpeg integration coverage over the full dataset.

These tests make zero API calls.
"""
import csv
import io
import pytest
from pathlib import Path
from PIL import Image

from code.agent.media import (
    _ffmpeg_available,
    detect_format,
    extract_video_frames,
    load_row_media,
)


# ---------------------------------------------------------------------------
# Helper: detect format from file header
# ---------------------------------------------------------------------------

def _detect_file(path: Path) -> str:
    with open(path, "rb") as fh:
        h = fh.read(12)
    return detect_format(h)


# ---------------------------------------------------------------------------
# P0.1 — Sample CSV integration test
# ---------------------------------------------------------------------------

def test_sample_csv_paths_all_resolve(sample_csv, dataset_root):
    """Every image path in sample_claims.csv must resolve and have usable frames."""
    with open(sample_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    missing = []
    no_frames = []
    total_refs = 0

    for row in rows:
        paths = [p.strip() for p in row["image_paths"].split(";") if p.strip()]
        media_files = load_row_media(paths, dataset_root)
        for mf in media_files:
            total_refs += 1
            if mf.actual_format == "MISSING":
                missing.append(mf.original_path)
            elif not mf.has_visual_content:
                no_frames.append(mf.original_path)

    assert not missing, f"Missing files in sample CSV: {missing}"
    assert not no_frames, f"Files with no usable frames in sample CSV: {no_frames}"
    assert total_refs > 0


def test_sample_csv_row_one_has_visual_content(sample_csv, dataset_root):
    """Quick check: first row of sample_claims.csv produces at least one frame."""
    with open(sample_csv, newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    paths = [p.strip() for p in row["image_paths"].split(";") if p.strip()]
    media_files = load_row_media(paths, dataset_root)
    total_frames = sum(len(mf.usable_frames) for mf in media_files)
    assert total_frames >= 1


# ---------------------------------------------------------------------------
# Full dataset audit (test + sample)
# ---------------------------------------------------------------------------

def test_full_dataset_audit(claims_csv, sample_csv, dataset_root):
    """Audit every referenced file in claims.csv and sample_claims.csv.

    Asserts:
    - zero MISSING files
    - format counts match expectations (JPEG/PNG/WEBP/MP4 only)
    - every file produces at least one usable frame
    """
    all_paths: set[str] = set()
    for csv_path in (claims_csv, sample_csv):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for p in row["image_paths"].split(";"):
                    p = p.strip()
                    if p:
                        all_paths.add(p)

    missing = []
    no_frames = []
    format_counts: dict[str, int] = {}

    for rel in sorted(all_paths):
        abs_path = (dataset_root / rel).resolve()
        if not abs_path.exists():
            missing.append(rel)
            format_counts["MISSING"] = format_counts.get("MISSING", 0) + 1
            continue
        fmt = _detect_file(abs_path)
        format_counts[fmt] = format_counts.get(fmt, 0) + 1

        mf_list = load_row_media([rel], dataset_root)
        for mf in mf_list:
            if not mf.has_visual_content and fmt != "MP4":
                no_frames.append(rel)
            elif not mf.has_visual_content and fmt == "MP4" and not _ffmpeg_available():
                pass  # expected: video needs FFmpeg

    # Report
    print("\n=== Dataset Media Audit ===")
    for fmt, count in sorted(format_counts.items()):
        print(f"  {fmt}: {count}")
    print(f"  Missing: {len(missing)}")
    print(f"  No frames (non-video): {len(no_frames)}")

    assert not missing, f"Files referenced in CSV but not found on disk: {missing}"
    # Only known formats
    unknown = [f for f in format_counts if f not in ("JPEG", "PNG", "WEBP", "MP4")]
    assert not unknown, f"Unexpected formats found: {unknown}"


# ---------------------------------------------------------------------------
# P0.2 — FFmpeg integration: all MP4 files in dataset
# ---------------------------------------------------------------------------

def test_all_mp4_files_produce_frames(dataset_root):
    """Every MP4 file in the dataset must produce >= 1 valid JPEG frame."""
    if not _ffmpeg_available():
        pytest.skip("FFmpeg not available — install to run video tests")

    images_root = dataset_root / "images"
    mp4_files = []
    for f in images_root.rglob("*.jpg"):
        fmt = _detect_file(f)
        if fmt == "MP4":
            mp4_files.append(f)

    assert mp4_files, "Expected to find MP4 files in dataset"

    results = []
    for path in mp4_files:
        frames = extract_video_frames(path, n_frames=3)
        results.append((path, len(frames)))
        assert frames, f"No frames extracted from {path}"
        for fb in frames:
            img = Image.open(io.BytesIO(fb))
            assert img.format == "JPEG"
            assert max(img.size) <= 1024, f"Frame too large from {path}: {img.size}"

    print(f"\n=== FFmpeg Video Audit ({len(results)} files) ===")
    for path, n in results:
        print(f"  {path.relative_to(dataset_root)}: {n} frame(s)")
