"""Small logging helpers for runtime traces and diagnostics."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ARTIFACTS_ROOT

_TRACE_PATH = ARTIFACTS_ROOT / "runtime_traces.jsonl"
_LLM_CALLS_PATH = ARTIFACTS_ROOT / "llm_calls.jsonl"


def configure_logging() -> logging.Logger:
    """Configure a simple console logger once."""

    logger = logging.getLogger("claim_verifier")
    if logger.handlers:
        return logger

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logger


def append_jsonl(file_path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON object per line for later analysis."""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def trace_event(name: str, payload: dict[str, Any]) -> None:
    """Write a structured runtime trace to the active trace file."""

    event = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "event": name,
        **payload,
    }
    append_jsonl(_TRACE_PATH, event)


def set_trace_output_path(path: Path) -> None:
    """Set the trace file location for the current run."""

    global _TRACE_PATH
    _TRACE_PATH = path


def log_llm_call(payload: dict[str, Any]) -> None:
    """Write one raw LLM input/output record for a real API call."""

    event = {
        "timestamp": datetime.now().astimezone().isoformat(),
        **payload,
    }
    append_jsonl(_LLM_CALLS_PATH, event)


def set_llm_calls_output_path(path: Path) -> None:
    """Set the LLM call log location for the current run."""

    global _LLM_CALLS_PATH
    _LLM_CALLS_PATH = path
