"""Deterministic post-processing and normalization."""

from __future__ import annotations

from typing import Iterable

from .constants import (
    CLAIM_STATUS_VALUES,
    ISSUE_TYPE_VALUES,
    JUDGE_RISK_FLAGS,
    OBJECT_PART_VALUES,
    PROMPT_INJECTION_HINTS,
    RISK_FLAG_VALUES,
    SEVERITY_VALUES,
)
from .models import ClaimAnalysis, ClaimRow, FinalPrediction, HistoryRow


def normalize_prediction(
    claim: ClaimRow,
    analysis: ClaimAnalysis,
    history: HistoryRow | None,
    missing_images: list[str],
) -> FinalPrediction:
    decision = analysis.decision
    risk_flags = set(_normalize_flags(decision.risk_flags))

    if _contains_injection_text(claim.user_claim):
        risk_flags.add("text_instruction_present")

    if history and history.history_flags != "none":
        risk_flags.update(flag.strip() for flag in history.history_flags.split(";") if flag.strip())

    if missing_images:
        risk_flags.add("manual_review_required")

    supporting_ids = [image_id for image_id in decision.supporting_image_ids if image_id]
    if decision.claim_status == "supported" and not supporting_ids:
        supporting_ids = [
            observation.image_id
            for observation in analysis.image_observations
            if observation.relevant and observation.supports_claim
        ]

    claim_status = _normalize_value(decision.claim_status, CLAIM_STATUS_VALUES, "not_enough_information")
    issue_type = _normalize_value(decision.issue_type, ISSUE_TYPE_VALUES, "unknown")
    object_part = _normalize_value(
        decision.object_part,
        OBJECT_PART_VALUES.get(claim.claim_object, {"unknown"}),
        "unknown",
    )
    severity = _normalize_value(decision.severity, SEVERITY_VALUES, "unknown")

    if missing_images and not supporting_ids:
        claim_status = "not_enough_information"

    return FinalPrediction(
        user_id=claim.user_id,
        image_paths=claim.image_paths,
        user_claim=claim.user_claim,
        claim_object=claim.claim_object,
        evidence_standard_met=str(bool(decision.evidence_standard_met)).lower(),
        evidence_standard_met_reason=_join_reason(
            decision.evidence_standard_met_reason,
            missing_images,
        ),
        risk_flags=";".join(sorted(risk_flags)) if risk_flags else "none",
        issue_type=issue_type,
        object_part=object_part,
        claim_status=claim_status,
        claim_status_justification=decision.claim_status_justification.strip(),
        supporting_image_ids=";".join(sorted(set(supporting_ids))) if supporting_ids else "none",
        valid_image=str(bool(decision.valid_image and not missing_images)).lower(),
        severity=severity,
    )


def should_run_judge(prediction: FinalPrediction, confidence: float, threshold: float) -> bool:
    if prediction.claim_status == "not_enough_information":
        return True

    flags = set(prediction.risk_flags.split(";")) if prediction.risk_flags != "none" else set()
    return confidence < threshold or bool(flags & JUDGE_RISK_FLAGS)


def apply_judge_feedback(
    prediction: FinalPrediction,
    judge_payload: dict[str, object],
    claim_object: str,
) -> FinalPrediction:
    if not judge_payload.get("needs_revision"):
        return prediction

    changes = judge_payload.get("suggested_changes") or {}
    flags = changes.get("risk_flags", prediction.risk_flags.split(";") if prediction.risk_flags != "none" else [])
    normalized_flags = _normalize_flags(flags)

    return prediction.model_copy(
        update={
            "risk_flags": ";".join(sorted(normalized_flags)) if normalized_flags else "none",
            "claim_status": _normalize_value(
                str(changes.get("claim_status", prediction.claim_status)),
                CLAIM_STATUS_VALUES,
                prediction.claim_status,
            ),
            "issue_type": _normalize_value(
                str(changes.get("issue_type", prediction.issue_type)),
                ISSUE_TYPE_VALUES,
                prediction.issue_type,
            ),
            "object_part": _normalize_value(
                str(changes.get("object_part", prediction.object_part)),
                OBJECT_PART_VALUES.get(claim_object, {"unknown"}),
                prediction.object_part,
            ),
            "supporting_image_ids": _normalize_supporting_ids(
                changes.get("supporting_image_ids"),
                prediction.supporting_image_ids,
            ),
            "claim_status_justification": str(
                changes.get("justification", prediction.claim_status_justification)
            ).strip(),
        }
    )


def _normalize_flags(flags: Iterable[str]) -> list[str]:
    normalized = []
    for flag in flags:
        value = str(flag).strip().lower()
        if value and value in RISK_FLAG_VALUES and value != "none":
            normalized.append(value)
    return sorted(set(normalized))


def _normalize_value(value: str, allowed: set[str], fallback: str) -> str:
    normalized = value.strip().lower()
    return normalized if normalized in allowed else fallback


def _contains_injection_text(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in PROMPT_INJECTION_HINTS)


def _join_reason(reason: str, missing_images: list[str]) -> str:
    reason = reason.strip()
    if not missing_images:
        return reason
    suffix = f" Missing images: {', '.join(missing_images)}."
    return f"{reason}{suffix}".strip()


def _normalize_supporting_ids(value: object, existing: str) -> str:
    if isinstance(value, list):
        cleaned = sorted({str(item).strip() for item in value if str(item).strip()})
        return ";".join(cleaned) if cleaned else "none"
    return existing
