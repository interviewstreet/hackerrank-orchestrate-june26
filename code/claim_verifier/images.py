"""Image path resolution and lightweight image quality checks."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import DATASET_ROOT
from .models import ImageAsset


class ImageLoadError(RuntimeError):
    """Raised when image preprocessing fails for a readable file."""


def split_image_paths(image_paths: str) -> list[str]:
    return [item.strip() for item in image_paths.split(";") if item.strip()]


def resolve_image_path(source_path: str) -> Path:
    return DATASET_ROOT / source_path


def _blur_score(image: Image.Image) -> float:
    array = np.asarray(image.convert("L"), dtype=np.float32)
    dx = np.diff(array, axis=1)
    dy = np.diff(array, axis=0)
    return float(np.var(dx) + np.var(dy))


def _brightness_score(image: Image.Image) -> float:
    array = np.asarray(image.convert("L"), dtype=np.float32)
    return float(array.mean())


def _encode_resized_image(path: Path, max_edge: int) -> bytes:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((max_edge, max_edge))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90)
            return buffer.getvalue()
    except Exception as exc:  # pragma: no cover - Pillow boundary
        raise ImageLoadError(f"Failed to preprocess image: {path}") from exc


def load_images(image_paths: str, max_edge: int) -> tuple[list[ImageAsset], list[str]]:
    assets: list[ImageAsset] = []
    missing: list[str] = []

    for source_path in split_image_paths(image_paths):
        resolved_path = resolve_image_path(source_path)
        image_id = resolved_path.stem
        if not resolved_path.exists():
            missing.append(source_path)
            continue

        encoded_bytes = _encode_resized_image(resolved_path, max_edge)
        with Image.open(io.BytesIO(encoded_bytes)) as image:
            blurry = _blur_score(image) < 40.0
            brightness = _brightness_score(image)
            dark_or_glary = brightness < 45.0 or brightness > 225.0

        assets.append(
            ImageAsset(
                image_id=image_id,
                source_path=source_path,
                resolved_path=resolved_path,
                mime_type="image/jpeg",
                encoded_bytes=encoded_bytes,
                blurry=blurry,
                dark_or_glary=dark_or_glary,
            )
        )

    return assets, missing


def to_gemini_inline_part(asset: ImageAsset) -> dict[str, dict[str, str]]:
    return {
        "inline_data": {
            "mime_type": asset.mime_type,
            "data": base64.b64encode(asset.encoded_bytes).decode("ascii"),
        }
    }
