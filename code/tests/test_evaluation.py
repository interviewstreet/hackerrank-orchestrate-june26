"""Tests for evaluation ground-truth loading and scoring logic (P0.5)."""
import csv
import io
import pytest
from pathlib import Path

from code.evaluation.metrics import compute_metrics, format_metrics
from code.evaluation.main import _extract_ground_truth, _align, OUTPUT_COLS


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sample_gt_row(user_id: str = "u1", claim_status: str = "supported") -> dict:
    return {
        "user_id": user_id,
        "image_paths": "images/sample/case_001/img_1.jpg",
        "user_claim": "My car is damaged.",
        "claim_object": "car",
        "evidence_standard_met": "true",
        "evidence_standard_met_reason": "Damage visible.",
        "risk_flags": "none",
        "issue_type": "dent",
        "object_part": "door",
        "claim_status": claim_status,
        "claim_status_justification": "Image shows dent.",
        "supporting_image_ids": "img_1",
        "valid_image": "true",
        "severity": "medium",
    }


# --- _extract_ground_truth ---

def test_extract_ground_truth_from_sample_csv(sample_csv):
    """sample_claims.csv must contain all 10 output columns."""
    with open(sample_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    gt = _extract_ground_truth(rows)
    assert len(gt) == len(rows)
    for row in gt:
        for col in OUTPUT_COLS:
            assert col in row, f"Missing column {col!r} in extracted GT"


def test_extract_ground_truth_raises_on_missing_columns():
    rows = [{"user_id": "u1", "image_paths": "x.jpg", "user_claim": "claim", "claim_object": "car"}]
    with pytest.raises(ValueError, match="missing expected output columns"):
        _extract_ground_truth(rows)


# --- _align ---

def test_align_matching_rows():
    gt = [_sample_gt_row("u1"), _sample_gt_row("u2")]
    pred = [_sample_gt_row("u1"), _sample_gt_row("u2")]
    aligned_pred, aligned_gt = _align(pred, gt)
    assert len(aligned_pred) == 2
    assert len(aligned_gt) == 2


def test_align_missing_prediction_skips_row(capsys):
    gt = [_sample_gt_row("u1"), _sample_gt_row("u2")]
    pred = [_sample_gt_row("u1")]
    aligned_pred, aligned_gt = _align(pred, gt)
    assert len(aligned_pred) == 1
    captured = capsys.readouterr()
    assert "u2" in captured.err


def test_align_extra_prediction_warns(capsys):
    gt = [_sample_gt_row("u1")]
    pred = [_sample_gt_row("u1"), _sample_gt_row("u99")]
    aligned_pred, aligned_gt = _align(pred, gt)
    assert len(aligned_pred) == 1
    captured = capsys.readouterr()
    assert "u99" in captured.err


# --- compute_metrics end-to-end ---

def test_compute_metrics_perfect_score():
    row = _sample_gt_row("u1", "supported")
    metrics = compute_metrics([row], [row])
    exact = metrics["exact"]
    assert exact["claim_status"].accuracy == 1.0
    assert exact["valid_image"].accuracy == 1.0


def test_compute_metrics_all_wrong():
    pred = [_sample_gt_row("u1", "contradicted")]
    gold = [_sample_gt_row("u1", "supported")]
    metrics = compute_metrics(pred, gold)
    assert metrics["exact"]["claim_status"].accuracy == 0.0


def test_compute_metrics_mismatch_raises():
    with pytest.raises(ValueError, match="Row count mismatch"):
        compute_metrics([_sample_gt_row("u1")], [_sample_gt_row("u1"), _sample_gt_row("u2")])


def test_format_metrics_returns_string():
    row = _sample_gt_row("u1")
    metrics = compute_metrics([row], [row])
    text = format_metrics(metrics)
    assert "claim_status" in text
    assert "F1" in text


def test_sample_csv_ground_truth_has_20_rows(sample_csv):
    """sample_claims.csv should have 20 labelled rows."""
    with open(sample_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20, f"Expected 20 rows, got {len(rows)}"
