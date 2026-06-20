"""Prompt builders for analysis and judge passes."""

from __future__ import annotations

import json
from pathlib import Path

from .config import PROMPTS_ROOT
from .constants import ISSUE_TYPE_VALUES, OBJECT_PART_VALUES, RISK_FLAG_VALUES
from .models import ClaimRow, HistoryRow, RequirementRow


def _allowed_object_parts(claim_object: str) -> list[str]:
    return sorted(OBJECT_PART_VALUES.get(claim_object, {"unknown"}))


def _history_context(history: HistoryRow | None) -> dict[str, object]:
    if history is None:
        return {"available": False, "note": "No prior claim history on record for this user."}

    flags = [
        flag.strip()
        for flag in history.history_flags.split(";")
        if flag.strip() and flag.strip() != "none"
    ]
    return {
        "available": True,
        "past_claim_count": history.past_claim_count,
        "accepted_claims": history.accept_claim,
        "manual_review_claims": history.manual_review_claim,
        "rejected_claims": history.rejected_claim,
        "last_90_days_claim_count": history.last_90_days_claim_count,
        "history_flags": flags,
        "history_summary": history.history_summary,
    }


def build_analysis_prompt(
    claim: ClaimRow,
    requirements: list[RequirementRow],
    prompt_version: str,
    history: HistoryRow | None = None,
) -> str:
    relevant_rules = [
        {
            "requirement_id": row.requirement_id,
            "applies_to": row.applies_to,
            "minimum_image_evidence": row.minimum_image_evidence,
        }
        for row in requirements
        if row.claim_object in {claim.claim_object, "all"}
    ]

    instruction = {
        "allowed_values": {
            "issue_type": sorted(ISSUE_TYPE_VALUES),
            "object_part": _allowed_object_parts(claim.claim_object),
            "risk_flags": sorted(RISK_FLAG_VALUES - {"none"}),
        },
        "claim": {
            "claim_object": claim.claim_object,
            "user_claim": claim.user_claim,
        },
        "evidence_requirements": relevant_rules,
        "user_history": _history_context(history),
    }
    template = load_prompt_text(prompt_version)
    return f"{template}\n\nContext JSON:\n{json.dumps(instruction, ensure_ascii=True, indent=2)}"


def build_judge_prompt(
    claim: ClaimRow,
    draft_output: dict[str, object],
    prompt_version: str,
) -> str:
    payload = {
        "claim": {
            "claim_object": claim.claim_object,
            "user_claim": claim.user_claim,
        },
        "draft_output": draft_output,
        "expected_fields": {
            "needs_revision": "boolean",
            "critique": "short string",
            "confidence": "0 to 1 float",
            "suggested_changes": {
                "risk_flags": "list of allowed risk flags",
                "claim_status": "supported|contradicted|not_enough_information",
                "issue_type": "allowed value",
                "object_part": "allowed value",
                "supporting_image_ids": "list of image ids",
                "justification": "short string",
            },
        },
    }
    template = load_prompt_text(prompt_version)
    return f"{template}\n\nContext JSON:\n{json.dumps(payload, ensure_ascii=True, indent=2)}"


def load_prompt_text(prompt_version: str) -> str:
    prompt_path = PROMPTS_ROOT / f"{prompt_version}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt file: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()
