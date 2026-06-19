"""Standalone smoke test for CALL 1 (extractor.extract_claim).

NOT a pytest test - it makes a real Anthropic API call. Run it manually to
confirm ANTHROPIC_API_KEY works and the prompt returns valid JSON before we
build CALL 2:

    cd code && python3 tests/smoke_extractor.py

Requires ANTHROPIC_API_KEY in the environment (or a .env loaded below).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Make code/ importable and load the API key before importing extractor.
CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
load_dotenv(CODE_DIR / ".env")
load_dotenv()  # fall back to repo-root / ambient env

import extractor

USER_CLAIM = (
    "Customer: Need to file a car damage claim. "
    "| Agent: What part of the car? | Customer: Door. "
    "| Agent: Scratch, dent, or paint issue? "
    "| Customer: A deep dent on the door panel."
)
CLAIM_OBJECT = "car"


def main() -> None:
    result = extractor.extract_claim(USER_CLAIM, CLAIM_OBJECT)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
