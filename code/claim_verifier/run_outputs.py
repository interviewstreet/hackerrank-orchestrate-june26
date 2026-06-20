"""Helpers for creating per-run output folders."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    traces_path: Path
    llm_calls_path: Path


def create_run_paths(output_dir: Path, run_kind: str, config_paths: list[Path]) -> RunPaths:
    """Create a timestamped folder for one run and snapshot its config files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{run_kind}_{timestamp}"
    suffix = 2
    while run_dir.exists():
        run_dir = output_dir / f"{run_kind}_{timestamp}_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    for config_path in config_paths:
        shutil.copy2(config_path, config_dir / config_path.name)

    return RunPaths(
        run_dir=run_dir,
        traces_path=run_dir / "runtime_traces.jsonl",
        llm_calls_path=run_dir / "llm_calls.jsonl",
    )
