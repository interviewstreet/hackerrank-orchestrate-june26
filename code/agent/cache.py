"""Content-addressed disk cache for VLM responses.

Cache key = SHA256 of:
  endpoint, provider, model, strategy,
  prompt/schema version, media-normalization tag, decode settings,
  claim.user_id, claim.claim_object, claim.user_claim, claim.image_paths,
  evidence_text, history_text,
  per-frame: image_id + label + bytes (with fallback label for unlabeled frames).

Stored as JSON files under code/.cache/ — excluded from git via .gitignore.
Bump _PROMPT_SCHEMA_VERSION whenever the system prompt template or output
schema changes.  Bump _MEDIA_NORM_TAG when normalization parameters change.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from code.agent.models import ClaimRow, MediaFile, ModelOutput

# Version constants — bump these when the corresponding component changes.
_PROMPT_SCHEMA_VERSION = "v1"         # system prompt template + JSON schema version
_MEDIA_NORM_TAG = (                   # normalization pipeline parameters
    "max_long_edge=1024;jpeg_quality=85;"
    "avif=single_frame_ffmpeg;video=3_frame_evenly_spaced"
)
_DECODE_TAG = "temperature=0;max_tokens=1024;thinking=false"
# Calibration guidance versions — only enter the respective strategy's cache keys.
# Bump when _CALIBRATION_BLOCK in prompt.py changes; Strategy A keys are unaffected.
_STRATEGY_B_CALIBRATION_VERSION = "v1"
_STRATEGY_C_CALIBRATION_VERSION = "v1"


def make_cache_key(
    provider: str,
    model: str,
    strategy: str,
    claim: ClaimRow,
    evidence_text: str | None,
    history_text: str | None,
    media_files: list[MediaFile],
    endpoint: str = "",
) -> str:
    """Return a hex SHA256 string that uniquely identifies one model call."""
    h = hashlib.sha256()
    # Provider / model / endpoint identity
    h.update(endpoint.encode())
    h.update(provider.encode())
    h.update(model.encode())
    # Version tags
    h.update(strategy.encode())
    h.update(_PROMPT_SCHEMA_VERSION.encode())
    h.update(_MEDIA_NORM_TAG.encode())
    h.update(_DECODE_TAG.encode())
    if strategy == "strategy_b":
        h.update(_STRATEGY_B_CALIBRATION_VERSION.encode())
    elif strategy == "strategy_c":
        h.update(_STRATEGY_C_CALIBRATION_VERSION.encode())
    # Claim identity
    h.update(claim.user_id.encode())
    h.update(claim.claim_object.encode())
    h.update(claim.user_claim.encode())
    h.update(claim.image_paths.encode())   # original semicolon-separated paths
    # Context text
    h.update((evidence_text or "").encode())
    h.update((history_text or "").encode())
    # Media: image_id + per-frame (label + bytes).
    # Uses enumerate + fallback label so unlabeled frames are never silently
    # omitted from the key (zip would stop at the shorter list).
    for mf in media_files:
        h.update(mf.image_id.encode())
        for i, frame in enumerate(mf.usable_frames):
            label = (
                mf.frame_labels[i]
                if i < len(mf.frame_labels)
                else f"{mf.image_id},frame{i},unlabeled"
            )
            h.update(label.encode())
            h.update(frame)
    return h.hexdigest()


class CacheStore:
    """Simple file-system cache backed by a directory of JSON blobs."""

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, key: str) -> ModelOutput | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return ModelOutput(**data)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def set(self, key: str, output: ModelOutput) -> None:
        self._path(key).write_text(
            output.model_dump_json(indent=2), encoding="utf-8"
        )
