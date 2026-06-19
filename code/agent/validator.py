"""Deterministic post-processing: enum enforcement, history flag merge,
supporting_image_ids subset validation, none-normalization.

Rules applied in order:
1. Clamp all string fields to their allowed enum (default to safest value).
2. Merge history-derived risk flags from HistoryRecord into risk_flags.
3. Strip any supporting_image_ids that are not in the submitted image ID set;
   fall back to ["none"] if all are stripped.
4. Normalize "none"-like values to the string "none" (single-element list).
5. Build the final OutputRow with all 14 string columns.
"""
from __future__ import annotations

from code.agent.models import ClaimRow, HistoryRecord, ModelOutput, OutputRow

# ---------------------------------------------------------------------------
# Allowed enumerations
# ---------------------------------------------------------------------------

_CLAIM_STATUSES = frozenset({"supported", "contradicted", "not_enough_information"})
_SEVERITIES = frozenset({"none", "low", "medium", "high", "unknown"})
_ISSUE_TYPES = frozenset({
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
})
_RISK_FLAGS = frozenset({
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
})

_OBJECT_PARTS: dict[str, frozenset[str]] = {
    "car": frozenset({
        "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
        "headlight", "taillight", "fender", "quarter_panel", "body", "unknown",
    }),
    "laptop": frozenset({
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port",
        "base", "body", "unknown",
    }),
    "package": frozenset({
        "box", "package_corner", "package_side", "seal", "label", "contents",
        "item", "unknown",
    }),
}


def _clamp(value: str, allowed: frozenset[str], default: str) -> str:
    v = value.strip().lower()
    return v if v in allowed else default


def _clamp_list(values: list[str], allowed: frozenset[str]) -> list[str]:
    valid = [v.strip().lower() for v in values if v.strip().lower() in allowed]
    return valid if valid else ["none"]


def _list_to_csv(items: list[str]) -> str:
    """Semicolon-separated; never duplicates, preserves order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return ";".join(out)


# ---------------------------------------------------------------------------
# History flag injection
# ---------------------------------------------------------------------------

_HISTORY_FLAG_TRIGGERS = frozenset({
    "repeated_claim", "high_frequency", "suspicious_pattern",
    "fraud_flag", "manual_review_required",
})

_HISTORY_MANUAL_REVIEW_TRIGGERS = frozenset({
    "manual_review_required",
})


def _merge_history_flags(
    risk_flags: list[str], history: HistoryRecord | None
) -> list[str]:
    """Deterministically merge history-derived flags into the VLM risk_flags list.

    Rules (applied in order):
    1. If history has any flag in _HISTORY_FLAG_TRIGGERS → add 'user_history_risk'.
    2. If history explicitly carries 'manual_review_required' → add it.
    3. Remove the sentinel 'none' whenever at least one real flag exists.
    """
    if history is None:
        return risk_flags

    flags = list(risk_flags)

    if history.flag_set & _HISTORY_FLAG_TRIGGERS:
        if "user_history_risk" not in flags:
            flags.append("user_history_risk")

    if history.flag_set & _HISTORY_MANUAL_REVIEW_TRIGGERS:
        if "manual_review_required" not in flags:
            flags.append("manual_review_required")

    # Remove sentinel "none" when real flags are present
    real = [f for f in flags if f != "none"]
    return real if real else ["none"]


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate_and_merge(
    raw: ModelOutput,
    claim: ClaimRow,
    history: HistoryRecord | None,
    submitted_image_ids: list[str],
) -> OutputRow:
    """Validate, clamp, merge, and normalise a ModelOutput → OutputRow."""
    # 1. Enum enforcement
    claim_status = _clamp(raw.claim_status, _CLAIM_STATUSES, "not_enough_information")
    severity = _clamp(raw.severity, _SEVERITIES, "unknown")
    issue_type = _clamp(raw.issue_type, _ISSUE_TYPES, "unknown")

    parts_allowed = _OBJECT_PARTS.get(claim.claim_object, frozenset({"unknown"}))
    object_part = _clamp(raw.object_part, parts_allowed, "unknown")

    risk_flags = _clamp_list(raw.risk_flags, _RISK_FLAGS)

    # 2. History flag merge
    risk_flags = _merge_history_flags(risk_flags, history)
    risk_flags = _clamp_list(risk_flags, _RISK_FLAGS)  # re-clamp after merge

    # 3. supporting_image_ids subset validation
    id_set = frozenset(submitted_image_ids)
    filtered_ids = [
        sid for sid in raw.supporting_image_ids
        if sid.strip().lower() not in ("none", "") and sid in id_set
    ]
    supporting_image_ids = filtered_ids if filtered_ids else ["none"]

    # 4. Propagate: if no valid images, force safe values
    valid_image = raw.valid_image
    evidence_standard_met = raw.evidence_standard_met
    if not valid_image:
        evidence_standard_met = False
        if claim_status == "supported":
            claim_status = "not_enough_information"
        supporting_image_ids = ["none"]

    # 5. Build final OutputRow
    return OutputRow(
        user_id=claim.user_id,
        image_paths=claim.image_paths,
        user_claim=claim.user_claim,
        claim_object=claim.claim_object,
        evidence_standard_met=str(evidence_standard_met).lower(),
        evidence_standard_met_reason=raw.evidence_standard_met_reason.strip() or "See justification.",
        risk_flags=_list_to_csv(risk_flags),
        issue_type=issue_type,
        object_part=object_part,
        claim_status=claim_status,
        claim_status_justification=raw.claim_status_justification.strip() or "See evidence review.",
        supporting_image_ids=_list_to_csv(supporting_image_ids),
        valid_image=str(valid_image).lower(),
        severity=severity,
    )


# ---------------------------------------------------------------------------
# Zero-media deterministic output
# ---------------------------------------------------------------------------

def zero_media_output(claim: ClaimRow, history: HistoryRecord | None) -> OutputRow:
    """Short-circuit result when no usable frames exist for a row."""
    risk_flags: list[str] = ["damage_not_visible"]
    if history is not None and (history.flag_set & _HISTORY_FLAG_TRIGGERS):
        risk_flags.append("user_history_risk")
    return OutputRow(
        user_id=claim.user_id,
        image_paths=claim.image_paths,
        user_claim=claim.user_claim,
        claim_object=claim.claim_object,
        evidence_standard_met="false",
        evidence_standard_met_reason="No usable image or video frames could be decoded from the submitted files.",
        risk_flags=_list_to_csv(risk_flags),
        issue_type="unknown",
        object_part="unknown",
        claim_status="not_enough_information",
        claim_status_justification="No visual evidence available for automated review.",
        supporting_image_ids="none",
        valid_image="false",
        severity="unknown",
    )
