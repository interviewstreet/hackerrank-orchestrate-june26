"""Standalone smoke test for CALL 2 (image_analyzer.analyze_images).

NOT a pytest test - it makes a real Anthropic API call and sends a real image.
Run it manually to confirm vision + structured JSON output works end-to-end
before building CALL 3:

    cd code && python3 tests/smoke_image_analyzer.py

Requires ANTHROPIC_API_KEY in the environment (or a .env loaded below).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
load_dotenv(CODE_DIR / ".env")
load_dotenv()  # fall back to repo-root / ambient env

import image_analyzer

ROOT = CODE_DIR.parent
IMAGE_PATH = str(ROOT / "dataset/images/test/case_001/img_1.jpg")

CLAIM_OBJECT = "car"
CLAIMED_PARTS = [{"object_part": "front_bumper", "issue_type": "dent"}]
CLAIM_SUMMARY = "Customer claims front bumper and headlight damage"
EVIDENCE_REQUIREMENTS = [
    {
        "requirement_id": "REQ_CAR_BODY_PANEL",
        "minimum_image_evidence": (
            "The claimed car panel or bumper should be visible from an angle "
            "where surface marks or deformation can be assessed."
        ),
    }
]


def main() -> None:
    print(f"Image: {IMAGE_PATH}")
    print(f"Exists: {Path(IMAGE_PATH).exists()}\n")

    result = image_analyzer.analyze_images(
        image_paths=[IMAGE_PATH],
        claim_object=CLAIM_OBJECT,
        claimed_parts=CLAIMED_PARTS,
        claim_summary=CLAIM_SUMMARY,
        evidence_requirements=EVIDENCE_REQUIREMENTS,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
