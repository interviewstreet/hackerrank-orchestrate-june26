"""Single-row pipeline: media → (maybe) VLM → validate → OutputRow.

run_row() is the public API:
    - resolves media paths against challenge/dataset/
    - checks for zero-media short-circuit (no API call)
    - checks cache; calls VLM on miss
    - on ModelCallError: builds a safe NEI row (valid_image=true if media
      existed), and does NOT write to cache
    - validates and merges deterministic rules
    - returns (OutputRow, RowStats)
"""
from __future__ import annotations

from pathlib import Path

from code.agent.cache import CacheStore, make_cache_key
from code.agent.evidence import EvidenceLoader
from code.agent.history import HistoryLoader
from code.agent.media import load_row_media
from code.agent.models import ClaimRow, OutputRow, RowStats
from code.agent.prompt import STRATEGY_B, build_system_prompt, build_user_message
from code.agent.validator import validate_and_merge, zero_media_output
from code.agent.vision_client import ModelCallError, VisionClient


def _history_text(claim: ClaimRow, history_loader: HistoryLoader) -> str | None:
    h = history_loader.get(claim.user_id)
    if h is None:
        return None
    return (
        f"summary={h.history_summary}; flags={h.history_flags}; "
        f"past={h.past_claim_count}; accepted={h.accept_claim}; "
        f"rejected={h.rejected_claim}; last90={h.last_90_days_claim_count}; "
        f"manual_review={h.manual_review_claim}"
    )


def _failure_row(claim: ClaimRow, history, has_media: bool) -> OutputRow:
    """Deterministic NEI row for transient model failure.

    valid_image reflects whether usable media existed (a transient API
    failure does not make images invalid).
    """
    from code.agent.validator import _list_to_csv, _HISTORY_FLAG_TRIGGERS
    risk_flags = ["manual_review_required"]
    if history is not None and (history.flag_set & _HISTORY_FLAG_TRIGGERS):
        risk_flags.append("user_history_risk")
    return OutputRow(
        user_id=claim.user_id,
        image_paths=claim.image_paths,
        user_claim=claim.user_claim,
        claim_object=claim.claim_object,
        evidence_standard_met="false",
        evidence_standard_met_reason="Model call failed; manual review required.",
        risk_flags=_list_to_csv(risk_flags),
        issue_type="unknown",
        object_part="unknown",
        claim_status="not_enough_information",
        claim_status_justification="Automated review unavailable due to model error.",
        supporting_image_ids="none",
        valid_image=str(has_media).lower(),
        severity="unknown",
    )


def run_row(
    claim: ClaimRow,
    repo_root: Path,
    history_loader: HistoryLoader,
    evidence_loader: EvidenceLoader,
    cache: CacheStore,
    client: VisionClient,
    strategy: str,
) -> tuple[OutputRow, RowStats]:
    """Process one ClaimRow end-to-end and return the final row + stats.

    Image paths in the CSV are relative to challenge/dataset/, so we resolve
    them against repo_root / "dataset".
    """
    stats = RowStats(user_id=claim.user_id, strategy=strategy)
    dataset_root = repo_root / "dataset"

    # --- media loading ---
    media_files = load_row_media(claim.image_path_list, dataset_root)
    total_frames = sum(len(mf.usable_frames) for mf in media_files)
    stats.images_submitted = len(media_files)
    stats.frames_extracted = total_frames

    # --- zero-media short-circuit ---
    if total_frames == 0:
        history = history_loader.get(claim.user_id)
        return zero_media_output(claim, history), stats

    # --- gather context strings for cache key and prompt ---
    history = history_loader.get(claim.user_id)
    evidence_text = evidence_loader.format_for_prompt(claim.claim_object) if strategy == STRATEGY_B else None
    hist_text = _history_text(claim, history_loader) if strategy == STRATEGY_B else None

    # --- cache lookup ---
    key = make_cache_key(
        provider=client.provider,
        model=client.model,
        strategy=strategy,
        claim=claim,
        evidence_text=evidence_text,
        history_text=hist_text,
        media_files=media_files,
        endpoint=client.base_url,
    )
    cached = cache.get(key)
    if cached is not None:
        stats.cache_hit = True
        stats.provider = client.provider
        stats.model = client.model
        output_row = validate_and_merge(cached, claim, history, claim.image_ids)
        return output_row, stats

    # --- VLM call ---
    system_prompt = build_system_prompt()
    user_content = build_user_message(
        claim=claim,
        media_files=media_files,
        history=history,
        evidence_text=evidence_text,
        strategy=strategy,
    )
    try:
        raw_output = client.call(system_prompt, user_content, stats)
    except ModelCallError:
        # Transient failure — do NOT cache; return safe NEI row
        return _failure_row(claim, history, has_media=True), stats

    # --- persist to cache only on success ---
    cache.set(key, raw_output)

    # --- deterministic validation & merge ---
    output_row = validate_and_merge(raw_output, claim, history, claim.image_ids)
    return output_row, stats
