"""Output writer.

Serializes per-claim predictions to ``output.csv`` in the column order and
schema expected by the evaluator (see ``dataset/sample_claims.csv`` for the
reference shape).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable
