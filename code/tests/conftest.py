"""Pytest fixtures / path setup for the code/ test suite.

Puts the ``code/`` directory on ``sys.path`` so tests can ``import escalation``
(and the other modules) regardless of the directory pytest is invoked from.
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
