"""Evaluation entry point: compare output.csv(s) against ground truth.

Ground truth defaults to ``dataset/sample_claims.csv`` which contains
all 14 columns (the last 10 are the expected output labels).

Usage:
    # Score strategy_a output vs sample labels
    python -m code.evaluation.main --strategy strategy_a

    # Score strategy_b output vs sample labels
    python -m code.evaluation.main --strategy strategy_b

    # Score both
    python -m code.evaluation.main --strategy both

Output prediction CSVs are expected at:
    challenge/output_strategy_a.csv   (strategy_a)
    challenge/output_strategy_b.csv   (strategy_b)

Workflow:
    1. Run: python -m code.main --strategy strategy_a \\
               --claims-csv dataset/sample_claims.csv \\
               --output-csv output_strategy_a.csv
    2. Then: python -m code.evaluation.main --strategy strategy_a
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_CHALLENGE_ROOT = Path(__file__).parent.parent.parent
if str(_CHALLENGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHALLENGE_ROOT))

# Input columns from claims.csv / sample_claims.csv
INPUT_COLS = {"user_id", "image_paths", "user_claim", "claim_object"}

# Expected-output columns (last 10 in sample_claims.csv)
OUTPUT_COLS = [
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
    "issue_type", "object_part", "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _extract_ground_truth(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return only the output columns from sample_claims.csv rows."""
    gt = []
    for row in rows:
        missing = [c for c in OUTPUT_COLS if c not in row]
        if missing:
            raise ValueError(
                f"Ground-truth CSV is missing expected output columns: {missing}. "
                "Expected sample_claims.csv with all 14 columns."
            )
        gt.append({c: row[c] for c in ["user_id"] + OUTPUT_COLS})
    return gt


def _align(predictions: list[dict], ground_truth: list[dict]) -> tuple[list, list]:
    """Align by user_id order from ground truth; warn on missing/duplicate."""
    gt_by_uid: dict[str, dict] = {}
    for row in ground_truth:
        uid = row["user_id"]
        if uid in gt_by_uid:
            print(f"WARNING: duplicate user_id in ground truth: {uid}", file=sys.stderr)
        gt_by_uid[uid] = row

    pred_by_uid: dict[str, dict] = {}
    for row in predictions:
        uid = row["user_id"]
        if uid in pred_by_uid:
            print(f"WARNING: duplicate user_id in predictions: {uid}", file=sys.stderr)
        pred_by_uid[uid] = row

    aligned_gt, aligned_pred = [], []
    for uid, gt_row in gt_by_uid.items():
        if uid not in pred_by_uid:
            print(f"WARNING: user_id {uid!r} in ground truth but not in predictions", file=sys.stderr)
            continue
        aligned_gt.append(gt_row)
        aligned_pred.append(pred_by_uid[uid])

    for uid in pred_by_uid:
        if uid not in gt_by_uid:
            print(f"WARNING: user_id {uid!r} in predictions but not in ground truth", file=sys.stderr)

    return aligned_pred, aligned_gt


def _evaluate_one(strategy: str, gt_path: Path) -> None:
    from code.evaluation.metrics import compute_metrics, format_metrics
    from code.evaluation.report import write_report

    pred_path = _CHALLENGE_ROOT / f"output_{strategy}.csv"
    print(f"\n--- Evaluating {strategy} ---")
    print(f"  predictions : {pred_path}")
    print(f"  ground truth: {gt_path}")

    gt_raw = _read_csv(gt_path)
    ground_truth = _extract_ground_truth(gt_raw)
    predictions = _read_csv(pred_path)

    aligned_pred, aligned_gt = _align(predictions, ground_truth)
    if not aligned_pred:
        print(f"  No aligned rows — cannot compute metrics.", file=sys.stderr)
        return

    metrics = compute_metrics(aligned_pred, aligned_gt)
    print(format_metrics(metrics))

    report_path = _CHALLENGE_ROOT / "docs" / f"eval_report_{strategy}.txt"
    write_report(metrics, strategy, report_path, aligned_pred, aligned_gt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline output against sample ground truth."
    )
    parser.add_argument(
        "--strategy",
        choices=["strategy_a", "strategy_b", "both"],
        default="strategy_b",
        help="Which output CSV(s) to evaluate (default: strategy_b)",
    )
    parser.add_argument(
        "--ground-truth",
        default=str(_CHALLENGE_ROOT / "dataset" / "sample_claims.csv"),
        help="Path to ground-truth CSV (default: dataset/sample_claims.csv)",
    )
    args = parser.parse_args(argv)

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(
            f"ERROR: Ground-truth CSV not found at {gt_path}.",
            file=sys.stderr,
        )
        return 1

    strategies = (
        ["strategy_a", "strategy_b"] if args.strategy == "both" else [args.strategy]
    )
    rc = 0
    for strat in strategies:
        try:
            _evaluate_one(strat, gt_path)
        except FileNotFoundError as exc:
            print(f"SKIP {strat}: {exc}", file=sys.stderr)
            rc = 1

    return rc


if __name__ == "__main__":
    sys.exit(main())
