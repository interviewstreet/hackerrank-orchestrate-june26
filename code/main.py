"""Terminal entry point.

Reads ``dataset/claims.csv``, runs each row through the agent pipeline, and
writes structured predictions to ``output.csv``. See AGENTS.md §6 for the
evaluable-submission contract.

``load_dotenv()`` is called before importing :mod:`agent` / :mod:`output` so
that ``ANTHROPIC_API_KEY`` is present in the environment before any module that
may construct an Anthropic client at import time.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()  # must run before importing modules that read ANTHROPIC_API_KEY

import csv
from pathlib import Path
from typing import Any

import agent
import output

# --- Paths (resolved from this file, not the current working directory) ------
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
CLAIMS_CSV = DATASET_DIR / "claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV = DATASET_DIR / "evidence_requirements.csv"
OUTPUT_CSV = ROOT / "output.csv"

# Requirement rows whose ``claim_object`` is this value apply to every claim.
ALL_OBJECTS = "all"


def load_user_history(path: Path) -> dict[str, dict[str, str]]:
    """Load ``user_history.csv`` into a dict keyed by ``user_id``."""
    history: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            history[row["user_id"]] = row
    return history


def load_requirements(path: Path) -> dict[str, list[dict[str, str]]]:
    """Load ``evidence_requirements.csv`` keyed by ``claim_object``.

    Rows scoped to ``all`` are folded into every object's requirement list so
    that the evidence gate sees both the general and object-specific rules.
    """
    by_object: dict[str, list[dict[str, str]]] = {}
    general: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            obj = row["claim_object"]
            if obj == ALL_OBJECTS:
                general.append(row)
            else:
                by_object.setdefault(obj, []).append(row)
    for obj in by_object:
        by_object[obj] = general + by_object[obj]
    by_object[ALL_OBJECTS] = general
    return by_object


def resolve_image_paths(raw: str, dataset_dir: Path) -> list[str]:
    """Split the ``;``-separated image paths and prepend ``dataset/``.

    Blank entries are dropped; an empty/blank field yields an empty list so the
    evidence gate can fail the claim deterministically instead of erroring.
    """
    paths: list[str] = []
    for part in (raw or "").split(";"):
        rel = part.strip()
        if rel:
            paths.append(str(dataset_dir / rel))
    return paths


def main() -> None:
    history = load_user_history(USER_HISTORY_CSV)
    requirements = load_requirements(EVIDENCE_REQUIREMENTS_CSV)

    predictions: list[dict[str, Any]] = []
    with CLAIMS_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row["resolved_image_paths"] = resolve_image_paths(
                row.get("image_paths", ""), DATASET_DIR
            )
            predictions.append(agent.process_claim(row, history, requirements))

    output.write(predictions, OUTPUT_CSV)


if __name__ == "__main__":
    main()
