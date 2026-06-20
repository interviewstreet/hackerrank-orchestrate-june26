"""CSV loading helpers."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from .config import DATASET_ROOT
from .models import ClaimRow, HistoryRow, RequirementRow


class DataLoadError(RuntimeError):
    """Raised when input files cannot be read or validated."""


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataLoadError(f"Missing required CSV: {path}")

    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise DataLoadError(f"Failed to read CSV: {path}") from exc


def load_claims(
    sample: bool = False,
    sample_count: int | None = None,
    sample_seed: int = 42,
) -> list[ClaimRow]:
    file_name = "sample_claims.csv" if sample else "claims.csv"
    frame = _load_csv(DATASET_ROOT / file_name)
    rows: list[ClaimRow] = []
    for record in frame.to_dict(orient="records"):
        rows.append(
            ClaimRow(
                user_id=str(record["user_id"]),
                image_paths=str(record["image_paths"]),
                user_claim=str(record["user_claim"]),
                claim_object=str(record["claim_object"]).strip().lower(),
            )
        )

    if sample and sample_count is not None and sample_count < len(rows):
        randomized = rows[:]
        random.Random(sample_seed).shuffle(randomized)
        return randomized[:sample_count]
    return rows


def load_history() -> dict[str, HistoryRow]:
    frame = _load_csv(DATASET_ROOT / "user_history.csv")
    rows: dict[str, HistoryRow] = {}
    for record in frame.to_dict(orient="records"):
        history = HistoryRow(**record)
        rows[history.user_id] = history
    return rows


def load_requirements() -> list[RequirementRow]:
    frame = _load_csv(DATASET_ROOT / "evidence_requirements.csv")
    return [RequirementRow(**record) for record in frame.to_dict(orient="records")]
