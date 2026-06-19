"""Media loading, format detection, normalisation, and FFmpeg frame extraction.

All images and video frames are normalised to JPEG with the long edge capped
at MAX_LONG_EDGE pixels before being passed to the VLM.  This keeps token
costs predictable and consistent across static images and video frames.

Magic-byte detection is used instead of the file extension, because the
dataset contains MP4/WEBP/PNG files with .jpg extensions.
"""
from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from code.agent.models import MediaFile

MAX_LONG_EDGE = 1024
FFMPEG_N_FRAMES = 3  # representative frames extracted per video file


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(header: bytes) -> str:
    """Identify actual file format from first 12 bytes."""
    if header[:2] == b"\xff\xd8":
        return "JPEG"
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "WEBP"
    # MP4/MOV: 4-byte box size + 'ftyp' or common QuickTime atoms
    if header[4:8] == b"ftyp":
        return "MP4"
    if header[4:8] in (b"moov", b"mdat", b"free", b"wide"):
        return "MP4"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Image normalisation (Pillow)
# ---------------------------------------------------------------------------

def _normalize_to_jpeg(raw: bytes, max_long_edge: int = MAX_LONG_EDGE) -> bytes:
    """Open any Pillow-supported format, resize to cap long edge, return JPEG bytes."""
    img = Image.open(io.BytesIO(raw))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w, h = img.size
    long_edge = max(w, h)
    if long_edge > max_long_edge:
        scale = max_long_edge / long_edge
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# FFmpeg integration
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _get_video_duration(path: Path) -> float | None:
    """Return video duration in seconds, or None on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return float(json.loads(result.stdout)["format"]["duration"])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            KeyError, ValueError, json.JSONDecodeError, OSError):
        return None


def _extract_single_frame(path: Path, max_long_edge: int) -> list[bytes]:
    """Extract the first/only frame from a durationless container (AVIF, single-image MP4).

    Some files use ISOBMFF (ftyp/moov atoms) as an image container rather than
    a video stream.  FFprobe reports no duration for these; seeking is not
    possible, but a simple first-frame extraction works.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", str(path),
                "-frames:v", "1",
                "-f", "image2pipe",
                "-vcodec", "mjpeg",
                "pipe:1",
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout:
            return [_normalize_to_jpeg(result.stdout, max_long_edge)]
    except (subprocess.TimeoutExpired, OSError):
        pass
    return []


def extract_video_frames(
    path: Path,
    n_frames: int = FFMPEG_N_FRAMES,
    max_long_edge: int = MAX_LONG_EDGE,
) -> list[bytes]:
    """Extract up to *n_frames* frames from a video/image container using FFmpeg.

    For actual video streams: positions t_i = duration*(i+1)/(n+1).
    For durationless containers (AVIF, single-frame MP4): extracts one frame.
    All frames are Pillow-normalised to JPEG with the long-edge capped.
    Returns [] on any failure.
    """
    if not _ffmpeg_available():
        return []

    duration = _get_video_duration(path)
    if duration is None or duration <= 0:
        # Durationless image container (e.g. AVIF stored with .jpg extension)
        return _extract_single_frame(path, max_long_edge)

    frames: list[bytes] = []
    for i in range(n_frames):
        t = duration * (i + 1) / (n_frames + 1)
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-ss", f"{t:.3f}",
                    "-i", str(path),
                    "-frames:v", "1",
                    "-f", "image2pipe",
                    "-vcodec", "mjpeg",
                    "pipe:1",
                ],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout:
                try:
                    normalised = _normalize_to_jpeg(result.stdout, max_long_edge)
                    frames.append(normalised)
                except Exception:
                    pass  # skip undecodable frames
        except (subprocess.TimeoutExpired, OSError):
            pass

    return frames


# ---------------------------------------------------------------------------
# Single-file loader
# ---------------------------------------------------------------------------

def load_media_file(path: Path) -> MediaFile:
    """Load one file (any format) into a MediaFile with normalised JPEG frames."""
    image_id = path.stem
    raw = path.read_bytes()
    fmt = detect_format(raw[:12])

    if fmt in ("JPEG", "PNG", "WEBP"):
        try:
            normalised = _normalize_to_jpeg(raw)
            label = f"{image_id}, frame 0/1, format {fmt}"
            return MediaFile(
                original_path=str(path),
                image_id=image_id,
                actual_format=fmt,
                usable_frames=[normalised],
                frame_labels=[label],
            )
        except Exception:
            return MediaFile(
                original_path=str(path),
                image_id=image_id,
                actual_format=fmt,
                usable_frames=[],
                frame_labels=[],
            )

    elif fmt == "MP4":
        frames = extract_video_frames(path)
        n = len(frames)
        labels = [
            f"{image_id}, frame {i}/{n}, format MP4"
            for i in range(n)
        ]
        return MediaFile(
            original_path=str(path),
            image_id=image_id,
            actual_format=fmt,
            usable_frames=frames,
            frame_labels=labels,
        )

    else:  # UNKNOWN or unreadable
        return MediaFile(
            original_path=str(path),
            image_id=image_id,
            actual_format=fmt,
            usable_frames=[],
            frame_labels=[],
        )


def load_row_media(image_path_strings: list[str], dataset_root: Path) -> list[MediaFile]:
    """Load all images for one CSV row.

    *image_path_strings* contains paths as they appear in claims.csv — i.e.
    relative to the *dataset/* directory (e.g. ``images/test/case_001/img_1.jpg``).
    *dataset_root* must point at the ``challenge/dataset/`` folder.

    Path-traversal safety: any path that resolves outside *dataset_root* is
    rejected and treated as MISSING.
    """
    resolved_root = dataset_root.resolve()
    result: list[MediaFile] = []
    for rel in image_path_strings:
        abs_path = (dataset_root / rel).resolve()
        # Reject traversal outside dataset_root
        try:
            abs_path.relative_to(resolved_root)
        except ValueError:
            result.append(
                MediaFile(
                    original_path=rel,
                    image_id=Path(rel).stem,
                    actual_format="TRAVERSAL",
                    usable_frames=[],
                    frame_labels=[],
                )
            )
            continue
        if not abs_path.exists():
            result.append(
                MediaFile(
                    original_path=rel,
                    image_id=Path(rel).stem,
                    actual_format="MISSING",
                    usable_frames=[],
                    frame_labels=[],
                )
            )
        else:
            result.append(load_media_file(abs_path))
    return result
