"""VisionClient abstraction — OpenAI-compatible adapter (primary: Qwen/dashscope-intl).

Provider configuration is entirely through environment variables:
    MODEL_PROVIDER   qwen | openai | anthropic  (default: qwen)
    OPENAI_BASE_URL  https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    DASHSCOPE_API_KEY / OPENAI_API_KEY   (never logged or printed)
    VISION_MODEL     qwen3.5-plus | qwen3.5-flash | gpt-4o | ...
"""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod

from code.agent.models import ModelOutput, RowStats

# ---------------------------------------------------------------------------
# Pricing snapshot (estimated list prices; free-quota usage is still reported
# as estimated list cost and labelled as such)
# ---------------------------------------------------------------------------
PRICING_SNAPSHOT: dict[str, dict] = {
    "qwen3.5-plus": {
        "date": "2026-06-19",
        "input_per_1m_usd": 0.40,
        "output_per_1m_usd": 2.40,
    },
    "qwen3.5-flash": {
        "date": "2026-06-19",
        "input_per_1m_usd": 0.07,
        "output_per_1m_usd": 0.30,
    },
    "gpt-4o": {
        "date": "2026-06-19",
        "input_per_1m_usd": 2.50,
        "output_per_1m_usd": 10.00,
    },
}


def _estimated_cost(
    model: str, input_tokens: int, output_tokens: int
) -> tuple[float | None, float | None]:
    p = PRICING_SNAPSHOT.get(model)
    if p is None:
        return None, None
    return (
        input_tokens / 1_000_000 * p["input_per_1m_usd"],
        output_tokens / 1_000_000 * p["output_per_1m_usd"],
    )


# ---------------------------------------------------------------------------
# Fallback model output (used when all retries fail)
# ---------------------------------------------------------------------------

def _fallback_output(reason: str) -> ModelOutput:
    return ModelOutput(
        evidence_standard_met=False,
        evidence_standard_met_reason=f"Model call failed: {reason[:120]}",
        risk_flags=["manual_review_required"],
        issue_type="unknown",
        object_part="unknown",
        claim_status="not_enough_information",
        claim_status_justification="Model call failed; cannot evaluate.",
        supporting_image_ids=["none"],
        valid_image=False,
        severity="unknown",
    )


# ---------------------------------------------------------------------------
# JSON parsing with controlled repair
# ---------------------------------------------------------------------------

def _parse_raw(raw_text: str) -> dict:
    """Parse JSON from model text; strip markdown fences if present."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.startswith("```")
        ).strip()
    return json.loads(text)


def _coerce_model_output(raw: dict) -> ModelOutput:
    def as_list(v) -> list[str]:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            return [s.strip() for s in v.split(";") if s.strip()]
        return ["none"]

    return ModelOutput(
        evidence_standard_met=bool(raw.get("evidence_standard_met", False)),
        evidence_standard_met_reason=str(raw.get("evidence_standard_met_reason", "")).strip(),
        risk_flags=as_list(raw.get("risk_flags", ["none"])),
        issue_type=str(raw.get("issue_type", "unknown")).strip(),
        object_part=str(raw.get("object_part", "unknown")).strip(),
        claim_status=str(raw.get("claim_status", "not_enough_information")).strip(),
        claim_status_justification=str(raw.get("claim_status_justification", "")).strip(),
        supporting_image_ids=as_list(raw.get("supporting_image_ids", ["none"])),
        valid_image=bool(raw.get("valid_image", False)),
        severity=str(raw.get("severity", "unknown")).strip(),
    )


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class VisionClient(ABC):
    @abstractmethod
    def call(
        self,
        system_prompt: str,
        user_content: list[dict],
        stats: RowStats,
    ) -> ModelOutput:
        """Send one multimodal request. Mutates *stats* with usage data."""
        ...

    @property
    @abstractmethod
    def provider(self) -> str: ...

    @property
    @abstractmethod
    def model(self) -> str: ...


# ---------------------------------------------------------------------------
# OpenAI-compatible adapter (covers Qwen/dashscope-intl, standard OpenAI,
# DeepSeek-VL, and any other OAI-compatible vision endpoint)
# ---------------------------------------------------------------------------

class OpenAICompatVisionClient(VisionClient):
    """Primary adapter.  Uses the openai Python SDK against any OAI-compatible endpoint."""

    def __init__(
        self,
        model: str | None = None,
        max_retries: int = 2,
        enable_thinking: bool = False,
    ) -> None:
        from openai import OpenAI

        provider = os.environ.get("MODEL_PROVIDER", "qwen").lower()
        if provider == "qwen":
            api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "No API key found.  Set DASHSCOPE_API_KEY (for Qwen) "
                "or OPENAI_API_KEY in your .env file."
            )

        base_url = os.environ.get("OPENAI_BASE_URL") or None
        self._model = model or os.environ.get("VISION_MODEL", "qwen3.5-plus")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._max_retries = max_retries
        self._enable_thinking = enable_thinking
        self._provider_name = os.environ.get("MODEL_PROVIDER", "qwen")

    @property
    def provider(self) -> str:
        return self._provider_name

    @property
    def model(self) -> str:
        return self._model

    def call(
        self,
        system_prompt: str,
        user_content: list[dict],
        stats: RowStats,
    ) -> ModelOutput:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        # Qwen3 thinking mode — disable for predictable latency/cost
        extra_body: dict = {}
        if not self._enable_thinking:
            extra_body["enable_thinking"] = False

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            t0 = time.monotonic()
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=1024,
                    extra_body=extra_body or None,
                )
                elapsed_ms = (time.monotonic() - t0) * 1000
                usage = resp.usage
                in_tok = usage.prompt_tokens if usage else 0
                out_tok = usage.completion_tokens if usage else 0
                stats.input_tokens += in_tok
                stats.output_tokens += out_tok
                stats.latency_ms += elapsed_ms
                stats.provider = self._provider_name
                stats.model = self._model

                est_in, est_out = _estimated_cost(self._model, in_tok, out_tok)
                stats.estimated_input_cost_usd = (
                    (stats.estimated_input_cost_usd or 0.0) + est_in
                    if est_in is not None else stats.estimated_input_cost_usd
                )
                stats.estimated_output_cost_usd = (
                    (stats.estimated_output_cost_usd or 0.0) + est_out
                    if est_out is not None else stats.estimated_output_cost_usd
                )

                raw_text = resp.choices[0].message.content or ""
                raw_dict = _parse_raw(raw_text)
                return _coerce_model_output(raw_dict)

            except json.JSONDecodeError as exc:
                last_exc = exc
                stats.retries += 1
                # One controlled repair: try again with stricter instruction
                if attempt == 0:
                    messages = messages + [{
                        "role": "assistant",
                        "content": resp.choices[0].message.content,
                    }, {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON. "
                            "Return ONLY the JSON object, no markdown, no explanation."
                        ),
                    }]
                else:
                    time.sleep(2 ** attempt)
            except Exception as exc:
                last_exc = exc
                stats.retries += 1
                time.sleep(min(2 ** attempt, 8))

        stats.error = str(last_exc)
        return _fallback_output(str(last_exc))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_client() -> VisionClient:
    """Build the correct client from environment variables."""
    return OpenAICompatVisionClient()
