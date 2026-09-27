"""Extract feature vector from RemiCare Cover Test sample.json.

Pipes data/raw/sample.json through:
    sample.json
        ↓
    Validation Gate (validate_screening_request)
        ↓
    Preprocessing (preprocess_screening_request)
        ↓
    Feature Extraction (extract_contract_features + Cover Test kinematics)
        ↓
    Compatibility check against Feature Contract
        ↓
    data/processed/remicare_features.json
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.schemas import ScreeningRequest
from app.services.feature_contract import (
    REQUIRED_SHARED_FEATURES,
    extract_contract_features,
    validate_feature_vector,
)
from app.services.feature_extraction import extract_screening_features
from app.services.preprocessing import preprocess_screening_request
from app.services.validation import validate_screening_request

__test__ = False


def extract_remicare_sample(
    sample_json_path: str,
    output_json_path: str,
) -> Dict[str, Any]:
    """Execute complete extraction and contract validation on a RemiCare sample."""
    if not os.path.exists(sample_json_path):
        raise FileNotFoundError(f"RemiCare sample not found at: {sample_json_path}")

    with open(sample_json_path, "r", encoding="utf-8") as f:
        raw_payload = json.load(f)

    # 1. Validation Gate
    request_obj = ScreeningRequest.model_validate(raw_payload)
    val_result = validate_screening_request(request_obj)
    if not val_result.is_valid:
        raise ValueError(f"Sample failed Data Integrity Gate: {val_result.reason}")

    # 2. Preprocessing & Native RemiCare Feature Extraction
    proc_data = preprocess_screening_request(request_obj)
    native_features = extract_screening_features(proc_data)

    # 3. Flatten samples across all cycles for shared contract feature extraction
    all_samples: List[Dict[str, Any]] = []
    sample_index = 0
    for cycle in request_obj.cycles:
        for s in cycle.samples:
            all_samples.append({
                "index": sample_index,
                "t": s.t,
                "phase": s.phase,
                "leftX": s.leftX,
                "leftY": s.leftY,
                "leftValid": s.leftValid,
                "rightX": s.rightX,
                "rightY": s.rightY,
                "rightValid": s.rightValid,
                "cycle": cycle.cycle,
                "trackedEye": cycle.trackedEye,
            })
            sample_index += 1

    # 4. Extract Contract Shared Features
    contract_features = extract_contract_features(all_samples)

    # Combine technical contract features with native Cover Test telemetry
    combined_features: Dict[str, Any] = {}
    combined_features.update(contract_features)

    # Add protocol metadata
    combined_features["cyclesAnalyzed"] = native_features["cyclesAnalyzed"]
    combined_features["cycleConsistency"] = native_features["consistency"].get("cycleConsistency", 0.0)
    combined_features["global_medianDx"] = native_features["aggregatedFeatures"].get("global_medianDx", 0.0)
    combined_features["global_maxPeakAbsDx"] = native_features["aggregatedFeatures"].get("global_maxPeakAbsDx", 0.0)

    # 5. Contract Compatibility Validation
    is_compatible, available_features, missing_features = validate_feature_vector(combined_features)

    output_record = {
        "sampleId": request_obj.sampleId,
        "source": "REMICARE_WEBCAM",
        "deviceType": "MEDIAPIPE_WEBCAM",
        "features": combined_features,
        "availableFeatures": available_features,
        "missingFeatures": missing_features,
        "compatibility": {
            "compatible": is_compatible,
            "sharedFeatureCount": len(available_features),
            "missingFeatureCount": len(missing_features),
        },
    }

    # Save to output file
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as fp:
        json.dump(output_record, fp, indent=2)

    return output_record


def main():
    sample_path = os.path.join(PROJECT_ROOT, "data", "raw", "sample.json")
    out_path = os.path.join(PROJECT_ROOT, "data", "processed", "remicare_features.json")

    print("=" * 60)
    print("RemiCare Feature Extraction & Contract Compatibility")
    print("=" * 60)
    print(f"Input Sample : {sample_path}")
    print(f"Output File  : {out_path}")
    print("-" * 60)

    record = extract_remicare_sample(sample_path, out_path)

    compat = record["compatibility"]
    print(f"Sample ID            : {record['sampleId']}")
    print(f"Source               : {record['source']}")
    print(f"Device Type          : {record['deviceType']}")
    print(f"Contract Compatible  : {compat['compatible']}")
    print(f"Shared Features Count: {compat['sharedFeatureCount']}")
    print(f"Missing Features     : {compat['missingFeatureCount']}")
    if record["missingFeatures"]:
        print(f"Missing List         : {record['missingFeatures']}")
    print("-" * 60)
    print(f"Extracted successfully and saved to {out_path}")


if __name__ == "__main__":
    main()
