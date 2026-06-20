"""Evaluation helpers for the sample set."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from .config import DATASET_ROOT
from .constants import OUTPUT_COLUMNS
from .models import FinalPrediction

FINAL_SCORE_WEIGHTS: dict[str, float] = {
    "claim_status_accuracy": 0.35,
    "issue_type_accuracy": 0.20,
    "object_part_accuracy": 0.15,
    "evidence_standard_met_accuracy": 0.10,
    "valid_image_accuracy": 0.08,
    "risk_flags_exact_match": 0.07,
    "severity_accuracy": 0.03,
    "claim_status_justification_non_empty": 0.02,
}


def score_metrics(metrics: dict[str, object]) -> float:
    """Compute a weighted final score in the range [0, 1]."""

    score = 0.0
    for key, weight in FINAL_SCORE_WEIGHTS.items():
        score += float(metrics[key]) * weight
    return round(score, 4)


def evaluate_predictions(predictions: list[FinalPrediction], expected_csv: Path) -> dict[str, object]:
    predicted = pd.DataFrame([prediction.model_dump() for prediction in predictions], columns=OUTPUT_COLUMNS)
    expected = pd.read_csv(expected_csv)
    join_columns = ["user_id", "image_paths", "user_claim", "claim_object"]
    predicted["_eval_key"] = predicted[join_columns].astype(str).agg("||".join, axis=1)
    expected["_eval_key"] = expected[join_columns].astype(str).agg("||".join, axis=1)
    expected = expected.set_index("_eval_key").reindex(predicted["_eval_key"]).reset_index(drop=True)
    predicted = predicted.drop(columns="_eval_key")

    metrics: dict[str, object] = {"row_count": len(predicted)}
    comparable_columns = [
        "evidence_standard_met",
        "issue_type",
        "object_part",
        "claim_status",
        "valid_image",
        "severity",
    ]
    for column in comparable_columns:
        metrics[f"{column}_accuracy"] = round(
            float(
                (
                    predicted[column].astype(str).str.lower()
                    == expected[column].astype(str).str.lower()
                ).mean()
            ),
            4,
        )

    metrics["risk_flags_exact_match"] = round(
        float((predicted["risk_flags"] == expected["risk_flags"].astype(str)).mean()),
        4,
    )
    metrics["claim_status_justification_non_empty"] = round(
        float(predicted["claim_status_justification"].astype(str).str.len().gt(0).mean()),
        4,
    )
    metrics["final_score"] = score_metrics(metrics)
    return metrics


def write_evaluation_report(
    strategy_metrics: dict[str, dict[str, object]],
    selected_strategy: str,
    report_path: Path,
    sample_count: int,
    sample_seed: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Evaluation Report",
        "",
        f"Selected strategy: `{selected_strategy}`",
        f"Overall final score: `{strategy_metrics[selected_strategy]['final_score']}`",
        "",
        f"Sample count: `{sample_count}`",
        f"Sample seed: `{sample_seed}`",
        "",
        "## Strategy comparison",
    ]
    for strategy_name, metrics in strategy_metrics.items():
        lines.append("")
        lines.append(f"### {strategy_name}")
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Operational analysis",
            "",
            "- Runtime traces are written to `code/artifacts/runtime_traces.jsonl`.",
            "- Gemini call traces include cache keys, prompt hashes, image counts, latency, and usage metadata when available.",
            "- Cost can be estimated from the trace file after a real run using the selected Gemini pricing assumptions.",
            "- Judge pass is gated so only higher-risk or lower-confidence rows incur the second model call.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_html_report(predictions: list[FinalPrediction], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for prediction in predictions:
        image_paths = prediction.image_paths.split(";")
        thumbnails = "<br>".join(
            f"<code>{html.escape(image_path)}</code>" for image_path in image_paths
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(prediction.user_id)}</td>"
            f"<td>{html.escape(prediction.claim_object)}</td>"
            f"<td>{thumbnails}</td>"
            f"<td>{html.escape(prediction.claim_status)}</td>"
            f"<td>{html.escape(prediction.issue_type)}</td>"
            f"<td>{html.escape(prediction.object_part)}</td>"
            f"<td>{html.escape(prediction.risk_flags)}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; vertical-align: top; text-align: left; }}
    code {{ white-space: nowrap; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Dataset root: <code>{html.escape(str(DATASET_ROOT))}</code></p>
  <table>
    <thead>
      <tr>
        <th>User</th>
        <th>Object</th>
        <th>Images</th>
        <th>Claim status</th>
        <th>Issue</th>
        <th>Part</th>
        <th>Risk flags</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8", newline="\n")
