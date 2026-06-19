"""Pipeline orchestrator.

Wires the deterministic gates and the 3-call LLM split into a single per-claim
flow:

    adversarial pre-filter  ->  CALL 1 claim extraction
                            ->  CALL 2 image analysis
                            ->  evidence + history gates
                            ->  CALL 3 verdict

Deterministic steps run before and around the LLM calls so that as much of the
verdict as possible is reproducible (MY_RULES: deterministic over LLM-driven).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import escalation
import extractor
import image_analyzer
