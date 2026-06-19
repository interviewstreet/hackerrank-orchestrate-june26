"""Evaluation entry point: compare output.csv(s) against ground truth.

Ground truth defaults to ``dataset/sample_claims.csv`` which contains
all 14 columns (the last 10 are the expected output labels).

Usage:
    # Score one strategy
    python -m code.evaluation.main --strategy strategy_a
    python -m code.evaluation.main --strategy strategy_b
    python -m code.evaluation.main --strategy strategy_c

    # Score all three
    python -m code.evaluation.main --strategy all

Output prediction CSVs are expected at:
    challenge/output_strategy_a.csv   (strategy_a)
    challenge/output_strategy_b.csv   (strategy_b)
    challenge/output_strategy_c.csv   (strategy_c)

Workflow:
    1. Run: python -m code.main --strategy strategy_c \\
               --claims-csv dataset/sample_claims.csv \\
               --output-csv output_strategy_c.csv \\
               --cache-dir code/.cache/eval_strategy_c_v1
    2. Then: python -m code.evaluation.main --strategy strategy_c
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
    """Strict positional alignment — same length and same user_id order required.

    Fails loudly on length mismatch or user_id mismatch at any position.
    This correctly handles rows with repeated user_ids (e.g. user_004 appears
    twice in test data) that a dict-based approach would silently drop.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Row count mismatch: predictions has {len(predictions)} rows, "
            f"ground truth has {len(ground_truth)} rows. "
            "Ensure inference was run on the exact same claims CSV as ground truth."
        )
    for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
        if pred["user_id"] != gt["user_id"]:
            raise ValueError(
                f"Row {i}: user_id mismatch — predictions has {pred['user_id']!r}, "
                f"ground truth has {gt['user_id']!r}. "
                "Output CSV must preserve the same row order as the input claims CSV."
            )
    return predictions, ground_truth


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
        choices=["strategy_a", "strategy_b", "strategy_c", "all"],
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
        ["strategy_a", "strategy_b", "strategy_c"] if args.strategy == "all" else [args.strategy]
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
