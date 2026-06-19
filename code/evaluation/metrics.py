"""Evaluation metrics: field-level accuracy, micro-F1 for set fields,
and confusion matrix for claim_status.

Columns graded:
  Exact-match (string equality, case-insensitive):
    evidence_standard_met, valid_image, issue_type, object_part,
    claim_status, severity

  Set micro-F1 (semicolon-separated multi-value columns):
    risk_flags, supporting_image_ids

  Not graded (free-text):
    evidence_standard_met_reason, claim_status_justification
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FieldMetrics:
    name: str
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def record(self, pred: str, gold: str) -> None:
        self.total += 1
        if pred.strip().lower() == gold.strip().lower():
            self.correct += 1


@dataclass
class SetMetrics:
    """Micro-averaged F1 for multi-label semicolon-separated columns."""
    name: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def record(self, pred: str, gold: str) -> None:
        pred_set = {v.strip().lower() for v in pred.split(";") if v.strip()}
        gold_set = {v.strip().lower() for v in gold.split(";") if v.strip()}
        self.tp += len(pred_set & gold_set)
        self.fp += len(pred_set - gold_set)
        self.fn += len(gold_set - pred_set)


@dataclass
class ConfusionMatrix:
    """3-class confusion matrix for claim_status."""
    classes: list[str] = field(default_factory=lambda: [
        "supported", "contradicted", "not_enough_information"
    ])
    matrix: dict[tuple[str, str], int] = field(default_factory=dict)

    def record(self, pred: str, gold: str) -> None:
        p = pred.strip().lower()
        g = gold.strip().lower()
        key = (g, p)
        self.matrix[key] = self.matrix.get(key, 0) + 1

    def display(self) -> str:
        header = f"{'':>25}" + "".join(f"{c:>25}" for c in self.classes)
        lines = [header]
        for gold in self.classes:
            row = f"{'gold:' + gold:>25}"
            for pred in self.classes:
                row += f"{self.matrix.get((gold, pred), 0):>25}"
            lines.append(row)
        return "\n".join(lines)


# Exact-match graded columns
EXACT_COLS = [
    "evidence_standard_met",
    "valid_image",
    "issue_type",
    "object_part",
    "claim_status",
    "severity",
]

# Set micro-F1 graded columns
SET_COLS = [
    "risk_flags",
    "supporting_image_ids",
]


def compute_metrics(
    predictions: list[dict[str, str]],
    ground_truth: list[dict[str, str]],
) -> dict:
    """Compute all metrics.

    Both lists must be the same length with matching user_id ordering.
    Returns a dict with all metric objects and a summary string.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Row count mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truth)} ground truth"
        )

    exact: dict[str, FieldMetrics] = {c: FieldMetrics(c) for c in EXACT_COLS}
    sets: dict[str, SetMetrics] = {c: SetMetrics(c) for c in SET_COLS}
    cm = ConfusionMatrix()

    for pred, gold in zip(predictions, ground_truth):
        for col in EXACT_COLS:
            exact[col].record(pred.get(col, ""), gold.get(col, ""))
        for col in SET_COLS:
            sets[col].record(pred.get(col, ""), gold.get(col, ""))
        cm.record(pred.get("claim_status", ""), gold.get("claim_status", ""))

    return {"exact": exact, "sets": sets, "confusion_matrix": cm}


def format_metrics(metrics: dict) -> str:
    exact: dict[str, FieldMetrics] = metrics["exact"]
    sets: dict[str, SetMetrics] = metrics["sets"]
    cm: ConfusionMatrix = metrics["confusion_matrix"]

    lines = ["=== Exact-Match Accuracy ==="]
    for col, m in exact.items():
        lines.append(f"  {col:<35} {m.correct:3d}/{m.total:3d}  {m.accuracy:.1%}")

    lines += ["", "=== Set Micro-F1 ==="]
    for col, m in sets.items():
        lines.append(
            f"  {col:<35} P={m.precision:.1%}  R={m.recall:.1%}  F1={m.f1:.1%}"
        )

    lines += ["", "=== Confusion Matrix (claim_status) ===", cm.display()]
    return "\n".join(lines)
