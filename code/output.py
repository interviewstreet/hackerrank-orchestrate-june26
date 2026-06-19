"""Output writer.

Serializes per-claim predictions to a CSV in the column order and schema
expected by the evaluator (see ``dataset/sample_claims.csv`` for the reference
shape). Secrets are never written here; this module contains no LLM calls.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


def write(predictions: list[dict[str, Any]], path: str | Path) -> None:
    """Write ``predictions`` to ``path`` as UTF-8 CSV with no BOM.

    Column order matches ``dataset/sample_claims.csv`` exactly. Keys absent
    from a prediction dict are written as empty strings. Extra keys are silently
    ignored (``extrasaction='ignore'``).
    """
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in predictions:
            writer.writerow({col: row.get(col, "") for col in COLUMNS})
