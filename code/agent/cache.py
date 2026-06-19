"""Content-addressed disk cache for VLM responses.

Cache key = SHA256 of:
  provider, model, strategy, claim.user_id, claim.claim_object,
  claim.user_claim, claim.image_paths (original), evidence_text,
  history_text, image_id + frame_label + frame_bytes per frame,
  decoding settings (temperature=0, max_tokens=1024, thinking=False).

This ensures that identical pixels presented under different image IDs
(which would produce different supporting_image_ids in the output) never
share a cached raw model result.

Stored as JSON files under code/.cache/ — excluded from git via .gitignore.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from code.agent.models import ClaimRow, MediaFile, ModelOutput

# Decoding settings that affect the VLM response — must be part of the key.
_DECODE_TAG = "temperature=0;max_tokens=1024;thinking=false"


def make_cache_key(
    provider: str,
    model: str,
    strategy: str,
    claim: ClaimRow,
    evidence_text: str | None,
    history_text: str | None,
    media_files: list[MediaFile],
) -> str:
    """Return a hex SHA256 string that uniquely identifies one model call."""
    h = hashlib.sha256()
    h.update(provider.encode())
    h.update(model.encode())
    h.update(strategy.encode())
    h.update(_DECODE_TAG.encode())
    # Claim identity
    h.update(claim.user_id.encode())
    h.update(claim.claim_object.encode())
    h.update(claim.user_claim.encode())
    h.update(claim.image_paths.encode())  # original semicolon-separated paths
    # Context text
    h.update((evidence_text or "").encode())
    h.update((history_text or "").encode())
    # Media: image_id + frame_label + frame_bytes so same pixels under
    # different IDs get different cache keys.
    for mf in media_files:
        h.update(mf.image_id.encode())
        for label, frame in zip(mf.frame_labels, mf.usable_frames):
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
