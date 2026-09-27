"""Phase 4: Run RemiCare to Korean Shared Model Transfer Experiment.

Pipes RemiCare data/raw/sample.json into:
    sample.json
        ↓
    Validation Gate
        ↓
    Preprocessing
        ↓
    Shared Feature Extraction (no imputation)
        ↓
    data/processed/remicare_shared_sample.json
        ↓
    Transfer Inference with models/korean_shared_model.joblib
        ↓
    Transfer Experiment Output with Domain Shift & Clinical Warnings
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.services.shared_feature_contract import (
    extract_remicare_shared_payload,
    run_shared_transfer_inference,
)


def run_pipeline_and_transfer(
    raw_sample_path: str,
    processed_sample_path: str,
    model_path: str,
) -> Dict[str, Any]:
    """Execute end-to-end RemiCare shared extraction and Korean transfer experiment."""
    # 1. Read raw sample.json
    if not os.path.exists(raw_sample_path):
        raise FileNotFoundError(f"Raw RemiCare sample not found: {raw_sample_path}")

    with open(raw_sample_path, "r", encoding="utf-8") as f:
        raw_payload = json.load(f)

    # 2. Extract RemiCare shared payload
    shared_payload = extract_remicare_shared_payload(raw_payload)

    # 3. Save to data/processed/remicare_shared_sample.json
    Path(processed_sample_path).parent.mkdir(parents=True, exist_ok=True)
    with open(processed_sample_path, "w", encoding="utf-8") as f:
        json.dump(shared_payload, f, indent=2)

    print(f"Saved RemiCare shared sample to: {processed_sample_path}")
    print(f"  Sample ID: {shared_payload['sampleId']}")
    print(f"  Features extracted: {len(shared_payload['features'])}")
    print(f"  Missing features: {len(shared_payload['missingFeatures'])}")

    # 4. Load trained Korean shared model artifact
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    model_artifact = joblib.load(model_path)

    # 5. Run Transfer Inference
    transfer_result = run_shared_transfer_inference(shared_payload, model_artifact)

    # 6. Save results to data/processed/remicare_transfer_results.csv
    import pandas as pd
    results_csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "remicare_transfer_results.csv")
    csv_row = {
        "sampleId": shared_payload["sampleId"],
        "prediction": transfer_result.get("prediction", "N/A"),
        "classProbability": transfer_result.get("classProbability", 0.0),
        "domainShiftWarning": transfer_result.get("domainShiftWarning", False),
        "inputCompatible": transfer_result.get("inputCompatible", False),
    }
    df_results = pd.DataFrame([csv_row])
    df_results.to_csv(results_csv_path, index=False)
    print(f"Saved transfer results summary to: {results_csv_path} (N = {len(df_results)})")

    return transfer_result


def main():
    raw_sample_path = os.path.join(PROJECT_ROOT, "data", "raw", "sample.json")
    processed_sample_path = os.path.join(PROJECT_ROOT, "data", "processed", "remicare_shared_sample.json")
    model_path = os.path.join(PROJECT_ROOT, "models", "korean_shared_model.joblib")

    print("=" * 65)
    print("PHASE 4: REMICARE -> KOREAN SHARED MODEL TRANSFER EXPERIMENT")
    print("=" * 65)
    print(f"Raw Input Sample : {raw_sample_path}")
    print(f"Processed Output : {processed_sample_path}")
    print(f"Model Path       : {model_path}")
    print("-" * 65)

    result = run_pipeline_and_transfer(raw_sample_path, processed_sample_path, model_path)

    print("\n--- TRANSFER EXPERIMENT RESULT ---")
    print(json.dumps(result, indent=2))
    print("-" * 65)

    print("\n[EVALUATION SAMPLE COUNT]")
    print("  N = 1 (Only 1 RemiCare Cover Test raw sample is currently present)")
    print("  NO GENERALIZATION CLAIM: Single-sample transfer evaluation cannot establish")
    print("  performance, sensitivity, or specificity on RemiCare.")

    if result.get("domainShiftWarning"):
        print("\n[WARNING] DOMAIN SHIFT DETECTED:")
        print(f"  Source: {result['domainShift'].get('sourceDomain')}")
        print(f"  Target: {result['domainShift'].get('targetDomain')}")
        print(f"  Notice: {result.get('notice')}")
        print(f"  Constraint: {result.get('productionConstraint')}")


if __name__ == "__main__":
    main()
