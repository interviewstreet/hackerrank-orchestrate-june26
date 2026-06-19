"""Content-addressed disk cache for VLM responses.

Cache key = SHA256(provider + model + strategy + claim_text + evidence_text
                   + history_text + sorted(frame_bytes per media file))

Stored as JSON files under code/.cache/ — excluded from git via .gitignore.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from code.agent.models import MediaFile, ModelOutput


def make_cache_key(
    provider: str,
    model: str,
    strategy: str,
    claim_text: str,
    evidence_text: str | None,
    history_text: str | None,
    media_files: list[MediaFile],
) -> str:
    """Return a hex SHA256 string that uniquely identifies one model call."""
    h = hashlib.sha256()
    h.update(provider.encode())
    h.update(model.encode())
    h.update(strategy.encode())
    h.update(claim_text.encode())
    h.update((evidence_text or "").encode())
    h.update((history_text or "").encode())
    for mf in media_files:
        for frame in mf.usable_frames:
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
