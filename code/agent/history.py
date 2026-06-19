"""User history loader — reads user_history.csv once and serves lookups."""
from __future__ import annotations

import csv
from pathlib import Path

from code.agent.models import HistoryRecord


class HistoryLoader:
    def __init__(self, csv_path: Path) -> None:
        self._records: dict[str, HistoryRecord] = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rec = HistoryRecord(
                    user_id=row["user_id"],
                    past_claim_count=int(row["past_claim_count"]),
                    accept_claim=int(row["accept_claim"]),
                    manual_review_claim=int(row["manual_review_claim"]),
                    rejected_claim=int(row["rejected_claim"]),
                    last_90_days_claim_count=int(row["last_90_days_claim_count"]),
                    history_flags=row["history_flags"],
                    history_summary=row["history_summary"],
                )
                self._records[rec.user_id] = rec

    def get(self, user_id: str) -> HistoryRecord | None:
        return self._records.get(user_id)
