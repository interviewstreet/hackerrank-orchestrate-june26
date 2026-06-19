"""Evaluation entry point: compare output.csv(s) against ground truth.

Usage:
    python -m code.evaluation.main --strategy strategy_a
    python -m code.evaluation.main --strategy strategy_b
    python -m code.evaluation.main --strategy both

Ground truth is challenge/dataset/output_gt.csv (must be provided separately;
if absent, evaluation prints a warning and exits cleanly).

Output CSVs expected at:
    challenge/output_strategy_a.csv  (for strategy_a)
    challenge/output_strategy_b.csv  (for strategy_b)
    challenge/output.csv             (default, used when --strategy is omitted)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_CHALLENGE_ROOT = Path(__file__).parent.parent.parent
if str(_CHALLENGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHALLENGE_ROOT))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _evaluate_one(strategy: str, gt_path: Path) -> None:
    from code.evaluation.metrics import compute_metrics, format_metrics
    from code.evaluation.report import write_report

    if strategy == "strategy_a":
        pred_path = _CHALLENGE_ROOT / "output_strategy_a.csv"
    elif strategy == "strategy_b":
        pred_path = _CHALLENGE_ROOT / "output_strategy_b.csv"
    else:
        pred_path = _CHALLENGE_ROOT / "output.csv"

    print(f"\n--- Evaluating {strategy} ---")
    print(f"  predictions : {pred_path}")
    print(f"  ground truth: {gt_path}")

    preds = _read_csv(pred_path)
    gt = _read_csv(gt_path)

    metrics = compute_metrics(preds, gt)
    print(format_metrics(metrics))

    report_path = _CHALLENGE_ROOT / "docs" / f"eval_report_{strategy}.txt"
    write_report(metrics, strategy, report_path, preds, gt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline output against ground truth."
    )
    parser.add_argument(
        "--strategy",
        choices=["strategy_a", "strategy_b", "both"],
        default="strategy_b",
        help="Which output CSV(s) to evaluate",
    )
    parser.add_argument(
        "--ground-truth",
        default=str(_CHALLENGE_ROOT / "dataset" / "output_gt.csv"),
        help="Path to ground-truth CSV",
    )
    args = parser.parse_args(argv)

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(
            f"WARNING: Ground-truth CSV not found at {gt_path}. "
            "Evaluation skipped.\n"
            "Supply a ground-truth file with --ground-truth <path> to enable scoring.",
            file=sys.stderr,
        )
        return 0

    strategies = (
        ["strategy_a", "strategy_b"] if args.strategy == "both" else [args.strategy]
    )
    for strat in strategies:
        try:
            _evaluate_one(strat, gt_path)
        except FileNotFoundError as exc:
            print(f"SKIP {strat}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
