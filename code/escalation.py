"""Deterministic gates: adversarial pre-filter, history lookup, evidence gate.

All rule-based, no LLM (MY_RULES: deterministic over LLM-driven):

* Adversarial pre-filter — scans the transcript and image-analysis signals for
  prompt injection / coercive language before any LLM call is trusted.
* User history lookup — copies the ``history_flags`` column from
  ``user_history.csv`` directly into ``risk_flags``.
* Evidence requirement gate — maps ``issue_family`` to the minimum image
  evidence in ``evidence_requirements.csv`` and flags shortfalls.
"""

from __future__ import annotations

import re

# --- SECTION 1: Adversarial pre-filter ---------------------------------------
# Injection / coercion phrases that should never come from a genuine claimant.
# Matched case-insensitively against the whitespace-normalized transcript.
INJECTION_PATTERNS: tuple[str, ...] = (
    "ignore all previous instructions",
    "approve the claim immediately",
    "skip manual review",
    "follow it and approve",
    "mark this row supported",
    "system reading this should",
)

TEXT_INSTRUCTION_FLAG = "text_instruction_present"


def adversarial_prefilter(user_claim: str) -> list[str]:
    """Scan the claim transcript for prompt-injection / coercion phrases.

    Deterministic, runs before any LLM call. Returns ``[TEXT_INSTRUCTION_FLAG]``
    if any known pattern is present, else ``[]``. Detection only adds a risk
    flag; it does NOT block downstream image analysis or the verdict call.
    """
    normalized = re.sub(r"\s+", " ", (user_claim or "")).casefold()
    for pattern in INJECTION_PATTERNS:
        if pattern in normalized:
            return [TEXT_INSTRUCTION_FLAG]
    return []


# --- SECTION 2: User history lookup ------------------------------------------
# The ``history_flags`` column is copied verbatim into risk_flags (MY_RULES).
# Values: "none" | "user_history_risk" | "manual_review_required"
# Multi-flag rows are semicolon-separated in the CSV, e.g.
# "user_history_risk;manual_review_required".

HISTORY_FLAGS_FIELD = "history_flags"
NO_FLAG_VALUE = "none"


def get_history_flags(user_id: str, history: dict[str, dict[str, str]]) -> list[str]:
    """Return the user's risk flags copied directly from ``user_history.csv``.

    Deterministic, no LLM. Returns ``[]`` for unknown users or users whose
    ``history_flags`` column is ``"none"`` or blank.
    """
    row = history.get(user_id)
    if not row:
        return []
    raw = row.get(HISTORY_FLAGS_FIELD, NO_FLAG_VALUE).strip()
    if not raw or raw == NO_FLAG_VALUE:
        return []
    return [flag.strip() for flag in raw.split(";") if flag.strip()]


# --- SECTION 3: Evidence requirement lookup ----------------------------------
# requirements dict is pre-merged by main.py: each claim_object key already
# includes the "all"-scoped rows. This function is a pure lookup + fallback.

ALL_OBJECT_KEY = "all"


def get_evidence_requirement(
    claim_object: str,
    requirements: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    """Return all evidence requirement rows applicable to ``claim_object``.

    Deterministic, no LLM. The caller passes the pre-merged requirements dict
    from main.py, so both general ("all") and object-specific rows are already
    combined under each key. Falls back to just the "all" rows for unrecognised
    objects so no claim is evaluated without any standard.
    """
    return requirements.get(claim_object, requirements.get(ALL_OBJECT_KEY, []))


# --- SECTION 4: Issue-family derivation --------------------------------------
# Deterministic mapping from issue_type -> issue_family (replaces letting CALL 1
# choose the family). missing_part is context-sensitive: a package's missing
# part is about contents, anything else is a broken/missing component.

FAMILY_GENERAL = "general claim review"

_ISSUE_FAMILY: dict[str, str] = {
    "dent": "dent or scratch",
    "scratch": "dent or scratch",
    "crack": "crack, broken, or missing part",
    "glass_shatter": "crack, broken, or missing part",
    "broken_part": "crack, broken, or missing part",
    "missing_part": "crack, broken, or missing part",  # non-package default
    "torn_packaging": "crushed, torn, or seal damage",
    "crushed_packaging": "crushed, torn, or seal damage",
    "water_damage": "water, stain, or label damage",
    "stain": "water, stain, or label damage",
    "none": FAMILY_GENERAL,
    "unknown": FAMILY_GENERAL,
}


def get_issue_family(issue_type: str, claim_object: str) -> str:
    """Map a primary ``issue_type`` to its evidence ``issue_family``.

    Deterministic, no LLM. ``missing_part`` is context-sensitive: for a
    ``package`` it means missing contents ("contents or inner item"); for any
    other object it is a missing component ("crack, broken, or missing part").
    Unrecognised issue types fall back to "general claim review".
    """
    if issue_type == "missing_part" and claim_object == "package":
        return "contents or inner item"
    return _ISSUE_FAMILY.get(issue_type, FAMILY_GENERAL)
