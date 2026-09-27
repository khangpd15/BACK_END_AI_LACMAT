"""Screening orchestration service for RemiCare Strabismus AI.

Coordinates the end-to-end processing pipeline:
1. Validation (Data Integrity Gate)
2. Preprocessing (Baseline median, relative coords, signed displacement)
3. Feature Extraction (Cycle-by-cycle metrics, inter-cycle consistency)
4. Inference Abstraction (Feature-based ML inference or safe inconclusive)
5. Structured Audit Logging (Zero PII, technical telemetry only)
"""

import logging
from typing import Optional

from app.schemas import (
    AnalysisSummary,
    QualitySummary,
    ScreeningRequest,
    ScreeningResponse,
    ScreeningStatus,
)
from app.services.feature_extraction import extract_screening_features
from app.services.inference import get_strabismus_model
from app.services.preprocessing import preprocess_screening_request
from app.services.validation import validate_screening_request

logger = logging.getLogger("remicare.screening")


def run_screening_pipeline(request: ScreeningRequest) -> ScreeningResponse:
    """Execute complete screening pipeline for a Cover Test request."""
    sample_id = request.sampleId or "unknown"
    cycle_count = len(request.cycles) if request.cycles else 0

    # Step 1: Data Integrity Gate
    val_result = validate_screening_request(request)

    if not val_result.is_valid:
        logger.warning(
            f"[ScreeningGate] sampleId={sample_id} status=INCONCLUSIVE "
            f"cycles={cycle_count} reason={val_result.reason} "
            f"samples={val_result.total_samples} valid={val_result.valid_samples}"
        )
        return ScreeningResponse(
            sampleId=sample_id,
            status=ScreeningStatus.SCREENING_INCONCLUSIVE.value,
            modelVersion=None,
            quality=QualitySummary(
                status="FAIL",
                reason=val_result.reason,
                sampleCount=val_result.total_samples,
                validSampleCount=val_result.valid_samples,
                validRatio=round(val_result.valid_ratio, 4),
                meanTrackingQuality=round(val_result.mean_tracking_quality, 4),
                issues=val_result.issues,
            ),
            analysis=AnalysisSummary(
                cyclesAnalyzed=0,
                cycleFeatures=[],
                aggregatedFeatures={},
                consistency={},
            ),
            reason=val_result.reason or "INSUFFICIENT_OR_INVALID_TIME_SERIES",
            notice="Screening result only — not a diagnosis.",
        )

    # Step 2: Preprocessing
    try:
        processed_data = preprocess_screening_request(request)
    except Exception as e:
        logger.error(f"[PreprocessingError] sampleId={sample_id} error={e}")
        return ScreeningResponse(
            sampleId=sample_id,
            status=ScreeningStatus.SCREENING_INCONCLUSIVE.value,
            modelVersion=None,
            quality=QualitySummary(
                status="FAIL",
                reason="PREPROCESSING_FAILURE",
                sampleCount=val_result.total_samples,
                validSampleCount=val_result.valid_samples,
                validRatio=val_result.valid_ratio,
                meanTrackingQuality=val_result.mean_tracking_quality,
                issues=[f"Preprocessing failed: {str(e)}"],
            ),
            analysis=AnalysisSummary(cyclesAnalyzed=0),
            reason="PREPROCESSING_FAILURE",
            notice="Screening result only — not a diagnosis.",
        )

    # Step 3: Feature Extraction
    try:
        features = extract_screening_features(processed_data)
        feature_extraction_status = "SUCCESS"
    except Exception as e:
        logger.error(f"[FeatureExtractionError] sampleId={sample_id} error={e}")
        return ScreeningResponse(
            sampleId=sample_id,
            status=ScreeningStatus.SCREENING_INCONCLUSIVE.value,
            modelVersion=None,
            quality=QualitySummary(
                status="FAIL",
                reason="FEATURE_EXTRACTION_FAILURE",
                sampleCount=val_result.total_samples,
                validSampleCount=val_result.valid_samples,
                validRatio=val_result.valid_ratio,
                meanTrackingQuality=val_result.mean_tracking_quality,
                issues=[f"Feature extraction failed: {str(e)}"],
            ),
            analysis=AnalysisSummary(cyclesAnalyzed=0),
            reason="FEATURE_EXTRACTION_FAILURE",
            notice="Screening result only — not a diagnosis.",
        )

    # Step 4: ML Inference Abstraction
    model = get_strabismus_model()
    inference_result = model.predict(features)

    # Step 5: Compute technical telemetry metrics for logging
    first_cycle_samples = request.cycles[0].samples if request.cycles and request.cycles[0].samples else []
    duration = 0.0
    sampling_interval = 0.0
    if len(first_cycle_samples) > 1:
        duration = first_cycle_samples[-1].t - first_cycle_samples[0].t
        intervals = [
            first_cycle_samples[i + 1].t - first_cycle_samples[i].t
            for i in range(len(first_cycle_samples) - 1)
        ]
        sampling_interval = sum(intervals) / len(intervals) if intervals else 0.0

    # Privacy-preserving technical log (Zero PII)
    logger.info(
        f"[ScreeningAudit] sampleId={sample_id} cycleCount={cycle_count} "
        f"sampleCount={val_result.total_samples} validSampleCount={val_result.valid_samples} "
        f"durationMs={duration:.1f} meanIntervalMs={sampling_interval:.1f} "
        f"qualityResult=PASS featureStatus={feature_extraction_status} "
        f"modelVersion={inference_result.modelVersion} finalStatus={inference_result.status}"
    )

    quality_summary = QualitySummary(
        status="PASS",
        reason=None,
        sampleCount=val_result.total_samples,
        validSampleCount=val_result.valid_samples,
        validRatio=round(val_result.valid_ratio, 4),
        meanTrackingQuality=round(val_result.mean_tracking_quality, 4),
        issues=[],
    )

    analysis_summary = AnalysisSummary(
        cyclesAnalyzed=features["cyclesAnalyzed"],
        cycleFeatures=features["cycleFeatures"],
        aggregatedFeatures=features["aggregatedFeatures"],
        consistency=features["consistency"],
    )

    return ScreeningResponse(
        sampleId=sample_id,
        status=inference_result.status,
        modelVersion=inference_result.modelVersion,
        quality=quality_summary,
        analysis=analysis_summary,
        reason=inference_result.reason,
        notice="Screening result only — not a diagnosis.",
    )
