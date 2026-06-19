"""Report writer: saves metrics text + per-row diff to docs/eval_report.txt."""
from __future__ import annotations

from pathlib import Path

from code.evaluation.metrics import format_metrics


def write_report(
    metrics: dict,
    strategy: str,
    output_path: Path,
    predictions: list[dict[str, str]],
    ground_truth: list[dict[str, str]],
) -> None:
    lines = [
        f"# Evaluation Report — {strategy}",
        "",
        format_metrics(metrics),
        "",
        "=== Per-Row Differences (claim_status) ===",
    ]
    for i, (pred, gold) in enumerate(zip(predictions, ground_truth), 1):
        p_status = pred.get("claim_status", "")
        g_status = gold.get("claim_status", "")
        if p_status.lower() != g_status.lower():
            uid = pred.get("user_id", f"row_{i}")
            lines.append(f"  [{i:02d}] {uid}: pred={p_status!r}  gold={g_status!r}")

    report_text = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    print(f"Report written to {output_path}")
