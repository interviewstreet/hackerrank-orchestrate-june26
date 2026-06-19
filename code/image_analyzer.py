"""CALL 2 — image analysis.

Inspects the claim's images (max 3 per claim) with a VLM and reports what
damage is visible, image quality, and any embedded-instruction / tampering
signals (e.g. text_instruction_present). Image paths from the CSV are resolved
by prepending ``dataset/`` (MY_RULES image path fix).

LLM/VLM client is wired in here once the provider is chosen (env-var key only).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# TODO: VLM client import — provider not yet decided (anthropic | openai).
