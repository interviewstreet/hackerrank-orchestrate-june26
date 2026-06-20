"""End-to-end pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from .config import Settings, StrategyConfig
from .constants import OUTPUT_COLUMNS
from .data import load_claims, load_history, load_requirements
from .decision import apply_judge_feedback, normalize_prediction, should_run_judge
from .gemini import GeminiClient
from .images import load_images
from .logging_utils import configure_logging, trace_event
from .models import ClaimRow, FinalPrediction
from .prompts import build_analysis_prompt, build_judge_prompt


class ClaimReviewPipeline:
    """Small orchestration layer around Gemini + deterministic normalization."""

    def __init__(self, settings: Settings, strategy: StrategyConfig) -> None:
        self._settings = settings
        self._strategy = strategy
        self._logger = configure_logging()
        self._history = load_history()
        self._requirements = load_requirements()
        self._gemini = GeminiClient(settings)

    def predict_dataset(
        self,
        sample: bool = False,
        sample_count: int | None = None,
        sample_seed: int = 42,
    ) -> list[FinalPrediction]:
        claims = load_claims(sample=sample, sample_count=sample_count, sample_seed=sample_seed)
        dataset_label = "sample" if sample else "claims"
        return self.predict_claims(claims, dataset_label=dataset_label)

    def predict_claims(self, claims: list[ClaimRow], dataset_label: str) -> list[FinalPrediction]:
        predictions: list[FinalPrediction] = []
        for claim_index, claim in enumerate(claims, start=1):
            predictions.append(
                self.predict_claim(
                    claim,
                    dataset_label=dataset_label,
                    claim_index=claim_index,
                    total_claims=len(claims),
                )
            )
        return predictions

    def predict_claim(
        self,
        claim: ClaimRow,
        dataset_label: str,
        claim_index: int,
        total_claims: int,
    ) -> FinalPrediction:
        started = perf_counter()
        trace_context = self._trace_context(claim, dataset_label, claim_index, total_claims)
        trace_event("claim_started", {**trace_context, "success": None})

        try:
            assets, missing_images = load_images(claim.image_paths, self._settings.resize_max_edge)
            if not assets:
                self._logger.warning("No readable images for user %s", claim.user_id)
                prediction = self._missing_image_prediction(claim, missing_images)
                trace_event(
                    "claim_finished",
                    {
                        **trace_context,
                        "success": True,
                        "status": prediction.claim_status,
                        "missing_images": missing_images,
                        "elapsed_ms": int((perf_counter() - started) * 1000),
                    },
                )
                return prediction

            prompt_history = self._history.get(claim.user_id) if self._strategy.include_user_history else None
            prompt = build_analysis_prompt(
                claim,
                self._requirements,
                self._strategy.analysis_prompt_version,
                history=prompt_history,
            )
            analysis = self._gemini.analyze_claim(claim, self._strategy.model, prompt, assets)
            prediction = normalize_prediction(
                claim=claim,
                analysis=analysis,
                history=self._history.get(claim.user_id),
                missing_images=missing_images,
            )

            if self._strategy.use_judge and should_run_judge(
                prediction,
                analysis.decision.confidence,
                self._settings.judge_confidence_threshold,
            ):
                judge_prompt = build_judge_prompt(
                    claim,
                    prediction.model_dump(),
                    self._strategy.judge_prompt_version or "judge_v1",
                )
                judge_payload = self._gemini.judge_claim(
                    self._strategy.judge_model or self._strategy.model,
                    judge_prompt,
                    assets,
                )
                prediction = apply_judge_feedback(prediction, judge_payload, claim.claim_object)
                trace_event(
                    "judge_applied",
                    {
                        **trace_context,
                        "needs_revision": bool(judge_payload.get("needs_revision")),
                        "confidence": judge_payload.get("confidence"),
                    },
                )

            trace_event(
                "claim_finished",
                {
                    **trace_context,
                    "success": True,
                    "status": prediction.claim_status,
                    "missing_images": missing_images,
                    "elapsed_ms": int((perf_counter() - started) * 1000),
                },
            )
            return prediction
        except Exception as exc:
            trace_event(
                "claim_finished",
                {
                    **trace_context,
                    "success": False,
                    "error": str(exc),
                    "elapsed_ms": int((perf_counter() - started) * 1000),
                },
            )
            raise

    @staticmethod
    def write_predictions(predictions: list[FinalPrediction], output_path: Path) -> None:
        frame = pd.DataFrame([prediction.model_dump() for prediction in predictions], columns=OUTPUT_COLUMNS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)

    @staticmethod
    def _missing_image_prediction(claim: ClaimRow, missing_images: list[str]) -> FinalPrediction:
        reason = "No readable images were available for review."
        if missing_images:
            reason = f"{reason} Missing images: {', '.join(missing_images)}."

        return FinalPrediction(
            user_id=claim.user_id,
            image_paths=claim.image_paths,
            user_claim=claim.user_claim,
            claim_object=claim.claim_object,
            evidence_standard_met="false",
            evidence_standard_met_reason=reason,
            risk_flags="manual_review_required",
            issue_type="unknown",
            object_part="unknown",
            claim_status="not_enough_information",
            claim_status_justification="The claim cannot be reviewed because no readable image evidence was available.",
            supporting_image_ids="none",
            valid_image="false",
            severity="unknown",
        )

    def _trace_context(
        self,
        claim: ClaimRow,
        dataset_label: str,
        claim_index: int,
        total_claims: int,
    ) -> dict[str, object]:
        first_image = claim.image_paths.split(";")[0].strip()
        sample_name = Path(first_image).parent.name if first_image else "unknown"
        return {
            "strategy": self._strategy.name,
            "dataset": dataset_label,
            "claim_index": claim_index,
            "total_claims": total_claims,
            "sample_name": sample_name,
            "user_id": claim.user_id,
            "claim_object": claim.claim_object,
        }
