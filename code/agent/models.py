"""Pydantic data models for the evidence review pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, computed_field


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class ClaimRow(BaseModel):
    user_id: str
    image_paths: str  # raw semicolon-separated string from CSV
    user_claim: str
    claim_object: Literal["car", "laptop", "package"]

    @computed_field
    @property
    def image_path_list(self) -> list[str]:
        return [p.strip() for p in self.image_paths.split(";") if p.strip()]

    @computed_field
    @property
    def image_ids(self) -> list[str]:
        return [Path(p).stem for p in self.image_path_list]


class MediaFile(BaseModel):
    """Represents one submitted file after loading and normalisation."""
    original_path: str      # relative path string exactly as in CSV
    image_id: str           # stem of original filename, e.g. "img_1"
    actual_format: str      # "JPEG", "PNG", "WEBP", "MP4", "MISSING", "UNKNOWN"
    # Normalised JPEG bytes for each usable visual (1 for static, 0-3 for video)
    usable_frames: list[bytes]
    # Human-readable label per frame: "img_1, frame 0/1, format JPEG"
    frame_labels: list[str] = []

    @computed_field
    @property
    def has_visual_content(self) -> bool:
        return len(self.usable_frames) > 0

    model_config = {"arbitrary_types_allowed": True}


class HistoryRecord(BaseModel):
    user_id: str
    past_claim_count: int
    accept_claim: int
    manual_review_claim: int
    rejected_claim: int
    last_90_days_claim_count: int
    history_flags: str   # "none" or semicolon-separated flag names
    history_summary: str

    @computed_field
    @property
    def flag_set(self) -> set[str]:
        if self.history_flags.strip().lower() == "none":
            return set()
        return {f.strip() for f in self.history_flags.split(";") if f.strip()}


class EvidenceRule(BaseModel):
    requirement_id: str
    claim_object: str      # "car", "laptop", "package", or "all"
    applies_to: str
    minimum_image_evidence: str


# ---------------------------------------------------------------------------
# Pipeline intermediate / output models
# ---------------------------------------------------------------------------

class ModelOutput(BaseModel):
    """Raw structured output from the VLM before deterministic validation."""
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: list[str]
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: list[str]
    valid_image: bool
    severity: str


class OutputRow(BaseModel):
    """Final CSV row — 14 columns in exact required order."""
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: str          # "true" / "false"
    evidence_standard_met_reason: str
    risk_flags: str                     # semicolon-sep or "none"
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str           # semicolon-sep or "none"
    valid_image: str                    # "true" / "false"
    severity: str


class RowStats(BaseModel):
    """Per-row operational statistics — never contains secrets."""
    user_id: str
    strategy: str
    input_tokens: int = 0
    output_tokens: int = 0
    images_submitted: int = 0
    frames_extracted: int = 0
    latency_ms: float = 0.0
    cache_hit: bool = False
    api_attempts: int = 0   # incremented before every SDK call including retries
    retries: int = 0
    error: str | None = None
    provider: str = ""
    model: str = ""
    # Estimated list-price cost from pricing snapshot; None if not computable
    estimated_input_cost_usd: float | None = None
    estimated_output_cost_usd: float | None = None
