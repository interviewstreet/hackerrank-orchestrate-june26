"""Run predictions for dataset/claims.csv and write output.csv at the repo root."""

from __future__ import annotations

from claim_verifier.config import load_prediction_config, load_settings
from claim_verifier.logging_utils import (
    configure_logging,
    set_llm_calls_output_path,
    set_trace_output_path,
)
from claim_verifier.pipeline import ClaimReviewPipeline
from claim_verifier.run_outputs import create_run_paths


def main() -> None:
    logger = configure_logging()
    try:
        settings = load_settings()
        prediction_config = load_prediction_config(settings.prediction_config_path)
        run_paths = create_run_paths(
            output_dir=prediction_config.output_dir,
            run_kind="prediction",
            config_paths=[settings.prediction_config_path],
        )
        set_trace_output_path(run_paths.traces_path)
        set_llm_calls_output_path(run_paths.llm_calls_path)
        pipeline = ClaimReviewPipeline(settings=settings, strategy=prediction_config.prediction)
        predictions = pipeline.predict_dataset(sample=False)
        output_path = run_paths.run_dir / "output.csv"
        pipeline.write_predictions(predictions, output_path)
        logger.info("Wrote %s predictions to %s", len(predictions), output_path)
    except Exception:
        logger.exception("Prediction run failed.")
        raise


if __name__ == "__main__":
    main()
