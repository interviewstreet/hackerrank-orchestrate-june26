"""Run sample-set evaluation and compare two strategies."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from claim_verifier.config import REPO_ROOT, load_evaluation_config, load_settings
from claim_verifier.data import load_claims
from claim_verifier.logging_utils import (
    configure_logging,
    set_llm_calls_output_path,
    set_trace_output_path,
)
from claim_verifier.metrics import (
    evaluate_predictions,
    score_metrics,
    write_evaluation_report,
    write_html_report,
)
from claim_verifier.pipeline import ClaimReviewPipeline
from claim_verifier.run_outputs import create_run_paths


def _score(metrics: dict[str, object]) -> float:
    return score_metrics(metrics)


def main() -> None:
    logger = configure_logging()
    try:
        settings = load_settings()
        evaluation_config = load_evaluation_config(settings.evaluation_config_path)
        run_paths = create_run_paths(
            output_dir=evaluation_config.output_dir,
            run_kind="evaluation",
            config_paths=[settings.evaluation_config_path],
        )
        set_trace_output_path(run_paths.traces_path)
        set_llm_calls_output_path(run_paths.llm_calls_path)
        sampled_claims = load_claims(
            sample=True,
            sample_count=evaluation_config.sample_count,
            sample_seed=evaluation_config.sample_seed,
        )
        strategy_predictions = {}
        strategy_metrics = {}
        expected_csv = REPO_ROOT / "dataset" / "sample_claims.csv"

        for strategy in evaluation_config.strategies:
            pipeline = ClaimReviewPipeline(settings=settings, strategy=strategy)
            predictions = pipeline.predict_claims(sampled_claims, dataset_label="sample")
            strategy_predictions[strategy.name] = predictions
            strategy_metrics[strategy.name] = evaluate_predictions(predictions, expected_csv)
            pipeline.write_predictions(
                predictions,
                run_paths.run_dir / "strategies" / strategy.name / "output.csv",
            )

        selected_strategy = max(strategy_metrics, key=lambda name: _score(strategy_metrics[name]))
        artifacts_dir = run_paths.run_dir / "evaluation"

        write_evaluation_report(
            strategy_metrics=strategy_metrics,
            selected_strategy=selected_strategy,
            report_path=artifacts_dir / "evaluation_report.md",
            sample_count=len(sampled_claims),
            sample_seed=evaluation_config.sample_seed,
        )
        write_html_report(
            predictions=strategy_predictions[selected_strategy],
            output_path=artifacts_dir / "report.html",
            title=f"Sample Evaluation - {selected_strategy}",
        )
        logger.info("Selected strategy: %s", selected_strategy)
    except Exception:
        logger.exception("Evaluation run failed.")
        raise


if __name__ == "__main__":
    main()
