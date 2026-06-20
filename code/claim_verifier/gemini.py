"""Gemini client wrapper with caching and structured traces."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from google import genai
from pydantic import ValidationError

from .config import ARTIFACTS_ROOT, Settings
from .images import to_gemini_inline_part
from .logging_utils import log_llm_call, trace_event
from .models import ClaimAnalysis, ClaimRow, ImageAsset


class GeminiClientError(RuntimeError):
    """Raised when Gemini fails to return valid structured output."""


class GeminiClient:
    """Thin wrapper around Gemini with file-based caching."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._cache_dir = ARTIFACTS_ROOT / "cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def analyze_claim(
        self,
        claim: ClaimRow,
        model: str,
        prompt: str,
        images: list[ImageAsset],
    ) -> ClaimAnalysis:
        payload = self._generate_json(
            model=model,
            prompt=prompt,
            images=images,
            cache_namespace="analysis",
        )
        try:
            return ClaimAnalysis.model_validate(payload)
        except ValidationError as exc:  # pragma: no cover - boundary with remote output
            trace_event(
                "schema_retry",
                {
                    "user_id": claim.user_id,
                    "model": model,
                    "reason": "claim_analysis_validation_failed",
                },
            )
            retry_payload = self._generate_json(
                model=model,
                prompt=f"{prompt}\n\n{_analysis_retry_instruction()}",
                images=images,
                cache_namespace="analysis_retry",
            )
            try:
                return ClaimAnalysis.model_validate(retry_payload)
            except ValidationError as retry_exc:  # pragma: no cover - boundary with remote output
                raise GeminiClientError(
                    f"Invalid structured analysis for user {claim.user_id} after one schema retry"
                ) from retry_exc

    def judge_claim(self, model: str, prompt: str, images: list[ImageAsset]) -> dict[str, Any]:
        return self._generate_json(
            model=model,
            prompt=prompt,
            images=images,
            cache_namespace="judge",
        )

    def _generate_json(
        self,
        model: str,
        prompt: str,
        images: list[ImageAsset],
        cache_namespace: str,
    ) -> dict[str, Any]:
        cache_key = self._cache_key(model=model, prompt=prompt, images=images)
        cache_file = self._cache_dir / cache_namespace / f"{cache_key}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        if cache_file.exists():
            trace_event(
                "model_call",
                {
                    "model": model,
                    "cache_namespace": cache_namespace,
                    "cache_key": cache_key,
                    "image_count": len(images),
                    "cache_hit": True,
                },
            )
            return _normalize_payload_root(json.loads(cache_file.read_text(encoding="utf-8")))

        contents: list[Any] = [prompt]
        contents.extend(to_gemini_inline_part(asset) for asset in images)

        started = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=model,
                contents=contents,
                config={
                    "temperature": 0,
                    "response_mime_type": "application/json",
                },
            )
        except Exception as exc:  # pragma: no cover - remote boundary
            raise GeminiClientError(f"Gemini call failed for model {model}") from exc

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        raw_text = getattr(response, "text", None)
        if not raw_text:
            raise GeminiClientError("Gemini returned an empty response.")

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:  # pragma: no cover - remote boundary
            raise GeminiClientError("Gemini did not return valid JSON.") from exc

        payload = _normalize_payload_root(payload)

        cache_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        log_llm_call(
            {
                "kind": "llm_call",
                "model": model,
                "cache_namespace": cache_namespace,
                "cache_key": cache_key,
                "image_count": len(images),
                "image_refs": [
                    {
                        "image_id": asset.image_id,
                        "source_path": asset.source_path,
                    }
                    for asset in images
                ],
                "prompt": prompt,
                "raw_output_text": raw_text,
                "elapsed_ms": elapsed_ms,
                "usage_metadata": _serialize_usage_metadata(getattr(response, "usage_metadata", None)),
            }
        )
        trace_event(
            "model_call",
            {
                "model": model,
                "cache_namespace": cache_namespace,
                "cache_key": cache_key,
                "image_count": len(images),
                "elapsed_ms": elapsed_ms,
                "cache_hit": False,
                "usage_metadata": _serialize_usage_metadata(getattr(response, "usage_metadata", None)),
            },
        )
        return payload

    @staticmethod
    def _cache_key(model: str, prompt: str, images: list[ImageAsset]) -> str:
        digest = hashlib.sha256()
        digest.update(model.encode("utf-8"))
        digest.update(prompt.encode("utf-8"))
        for asset in images:
            digest.update(asset.image_id.encode("utf-8"))
            digest.update(asset.encoded_bytes)
        return digest.hexdigest()


def _serialize_usage_metadata(value: object) -> object:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _normalize_payload_root(payload: object) -> dict[str, Any]:
    """Accept the common singleton-list mistake and normalize it to one object."""

    if isinstance(payload, dict):
        return payload

    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        trace_event(
            "payload_normalized",
            {
                "normalization": "singleton_list_unwrapped",
            },
        )
        return payload[0]

    raise GeminiClientError(
        "Model returned an unexpected JSON root type. Expected one object, got "
        f"{type(payload).__name__}."
    )


def _analysis_retry_instruction() -> str:
    return (
        "SCHEMA CORRECTION: Your previous response did not match the required schema. "
        "Return exactly one top-level JSON object, not a list. "
        "Do not add markdown, explanations, or extra keys. "
        "The object must contain extracted_claim, image_observations, and decision."
    )
