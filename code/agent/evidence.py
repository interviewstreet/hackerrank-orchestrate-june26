"""Evidence requirements loader.

Strategy B includes ALL rules that apply to the row's claim_object
(object-specific rules + global 'all' rules), so the single VLM call
can select the most relevant one without a preliminary classifier.
"""
from __future__ import annotations

import csv
from pathlib import Path

from code.agent.models import EvidenceRule


class EvidenceLoader:
    def __init__(self, csv_path: Path) -> None:
        self.rules: list[EvidenceRule] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rules.append(EvidenceRule(**row))

    def get_all_for_object(self, claim_object: str) -> list[EvidenceRule]:
        """Return all rules where claim_object matches or equals 'all'."""
        return [
            r for r in self.rules
            if r.claim_object == claim_object or r.claim_object == "all"
        ]

    def format_for_prompt(self, claim_object: str) -> str:
        """Return a compact multi-line string of all applicable rules for the prompt."""
        rules = self.get_all_for_object(claim_object)
        lines = []
        for r in rules:
            lines.append(f"- [{r.requirement_id}] ({r.applies_to}): {r.minimum_image_evidence}")
        return "\n".join(lines)
