"""Pydantic models shared across the pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ClaimRow(BaseModel):
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str


class HistoryRow(BaseModel):
    user_id: str
    past_claim_count: int
    accept_claim: int
    manual_review_claim: int
    rejected_claim: int
    last_90_days_claim_count: int
    history_flags: str
    history_summary: str


class RequirementRow(BaseModel):
    requirement_id: str
    claim_object: str
    applies_to: str
    minimum_image_evidence: str


class ImageAsset(BaseModel):
    image_id: str
    source_path: str
    resolved_path: Path
    mime_type: str
    encoded_bytes: bytes
    blurry: bool
    dark_or_glary: bool

    model_config = {"arbitrary_types_allowed": True}


class ImageObservation(BaseModel):
    image_id: str
    relevant: bool
    visible_object: str = "unknown"
    visible_part: str = "unknown"
    visible_issue: str = "unknown"
    supports_claim: bool = False
    quality_flags: list[str] = Field(default_factory=list)
    summary: str


class LlmDecision(BaseModel):
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: list[str] = Field(default_factory=list)
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: list[str] = Field(default_factory=list)
    valid_image: bool
    severity: str
    confidence: float = 0.0


class ClaimAnalysis(BaseModel):
    extracted_claim: str
    image_observations: list[ImageObservation]
    decision: LlmDecision


class FinalPrediction(BaseModel):
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: str
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: str
    severity: str
