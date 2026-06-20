"""Runtime configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
DATASET_ROOT = REPO_ROOT / "dataset"
ARTIFACTS_ROOT = CODE_ROOT / "artifacts"
PROMPTS_ROOT = CODE_ROOT / "prompts"
CONFIGS_ROOT = CODE_ROOT / "configs"


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    resize_max_edge: int = 1200
    judge_confidence_threshold: float = 0.7
    prediction_config_path: Path = CONFIGS_ROOT / "prediction_config.yaml"
    evaluation_config_path: Path = CONFIGS_ROOT / "evaluation_config.yaml"


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    model: str
    analysis_prompt_version: str
    include_user_history: bool = False
    use_judge: bool = False
    judge_model: str | None = None
    judge_prompt_version: str | None = None


@dataclass(frozen=True)
class PredictionConfig:
    output_dir: Path
    prediction: StrategyConfig


@dataclass(frozen=True)
class EvaluationConfig:
    output_dir: Path
    sample_count: int | None
    sample_seed: int
    strategies: list[StrategyConfig]


def load_settings() -> Settings:
    """Load environment variables for runtime execution."""

    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(CODE_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError(
            "Missing GEMINI_API_KEY. Add it to your environment, repo-root .env, or code/.env."
        )

    return Settings(
        gemini_api_key=api_key,
        resize_max_edge=int(os.getenv("RESIZE_MAX_EDGE", "1200")),
        judge_confidence_threshold=float(os.getenv("JUDGE_CONFIDENCE_THRESHOLD", "0.7")),
        prediction_config_path=Path(
            os.getenv("PREDICTION_CONFIG_PATH", str(CONFIGS_ROOT / "prediction_config.yaml")).strip()
            or str(CONFIGS_ROOT / "prediction_config.yaml")
        ),
        evaluation_config_path=Path(
            os.getenv("EVALUATION_CONFIG_PATH", str(CONFIGS_ROOT / "evaluation_config.yaml")).strip()
            or str(CONFIGS_ROOT / "evaluation_config.yaml")
        ),
    )


def load_prediction_config(path: Path) -> PredictionConfig:
    """Load the final prediction strategy configuration from YAML."""

    payload = _load_yaml(path)
    output_dir = _resolve_output_dir(str(payload.get("output_dir", ".")).strip() or ".")
    prediction = _parse_strategy(payload.get("prediction"), field_name="prediction")
    return PredictionConfig(output_dir=output_dir, prediction=prediction)


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Load evaluation strategy configuration from YAML."""

    payload = _load_yaml(path)
    output_dir = _resolve_output_dir(str(payload.get("output_dir", ".")).strip() or ".")
    strategies_payload = payload.get("strategies", [])
    if not strategies_payload:
        raise ConfigurationError("Evaluation config must define at least one strategy.")

    sample_count = payload.get("sample_count")
    if sample_count is not None:
        sample_count = int(sample_count)
        if sample_count <= 0:
            raise ConfigurationError("sample_count must be positive when provided.")

    return EvaluationConfig(
        output_dir=output_dir,
        sample_count=sample_count,
        sample_seed=int(payload.get("sample_seed", 42)),
        strategies=[
            _parse_strategy(strategy_payload, field_name=f"strategies[{index}]")
            for index, strategy_payload in enumerate(strategies_payload)
        ],
    )


def _load_yaml(path: Path) -> dict[str, object]:
    """Load a YAML file into a dictionary."""

    if not path.exists():
        raise ConfigurationError(f"Missing config file: {path}")

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in config file: {path}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError(f"Config file must contain a top-level mapping: {path}")
    return payload


def _parse_strategy(payload: object, field_name: str) -> StrategyConfig:
    if not isinstance(payload, dict):
        raise ConfigurationError(f"{field_name} must be an object.")

    name = str(payload.get("name", "")).strip()
    model = str(payload.get("model", "")).strip()
    analysis_prompt_version = str(payload.get("analysis_prompt_version", "")).strip()
    if not name or not model or not analysis_prompt_version:
        raise ConfigurationError(
            f"{field_name} must define name, model, and analysis_prompt_version."
        )

    use_judge = bool(payload.get("use_judge", False))
    include_user_history = bool(payload.get("include_user_history", False))
    judge_model = str(payload.get("judge_model", "")).strip() or None
    judge_prompt_version = str(payload.get("judge_prompt_version", "")).strip() or None
    if use_judge:
        judge_model = judge_model or model
        judge_prompt_version = judge_prompt_version or "judge_v1"

    return StrategyConfig(
        name=name,
        model=model,
        analysis_prompt_version=analysis_prompt_version,
        include_user_history=include_user_history,
        use_judge=use_judge,
        judge_model=judge_model,
        judge_prompt_version=judge_prompt_version,
    )


def _resolve_output_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()
